"""
cogops/agents/orchestrator.py

The Orchestrator for the GovOps Agent.

- Loads config, builds tool registry + system prompt.
- Per-request: reconstructs context from Redis, truncates to token budget,
  runs the reasoning loop, persists results, kicks off async summarizer.
- Session recovery: on startup, loads existing Redis turns so conversations
  survive server restarts.
"""

import os
import json
import asyncio
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import AsyncGenerator, Dict, Any, List, Tuple, Optional

import yaml
from dotenv import load_dotenv

from cogops.config.loader import _load_endpoint_config
from cogops.llm.clients import AsyncLLMService
from cogops.llm.reasoning_loop import stream_with_tool_calls
from cogops.prompts.messages import SERVER_LOAD_FALLBACK_BN
from cogops.prompts.system import get_system_prompt
from cogops.session.redis_store import RedisSessionStore
from cogops.session.summarizer import run_summarizer_task
from cogops.tools.registry import build_tool_registry, bind_tools, ToolContext
from cogops.utils.tokenizer import Tokenizer
from cogops.utils.truncate import truncate_messages_to_budget

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Orchestrator:
    _cached_system_prompt: Optional[str] = None

    def __init__(self, config_path: str = "configs/config.yml"):
        logger.info("Initializing GovOps Orchestrator...")
        self.config = self._load_config(config_path)
        self._load_call_config()

        self.agent_name = self.config.get('agent', {}).get('name', 'Gov Assistant')
        self.agent_story = self.config.get('agent', {}).get('story', '')

        self.llm_service = self._initialize_llm()

        # --- Redis session store ---
        session_cfg = self.config.get('session', {})
        redis_url = os.getenv(
            session_cfg.get('redis_url_env', 'REDIS_URL'),
            session_cfg.get('redis_url_default', "redis://localhost:6379/0"),
        )
        ttl = int(os.getenv(
            session_cfg.get('ttl_seconds_env', 'REDIS_SESSION_TTL_SECONDS'),
            str(session_cfg.get('ttl_default', 86400)),
        ))
        self.redis_store = RedisSessionStore(url=redis_url, ttl_seconds=ttl)

        # --- Tool registry ---
        self.tools_schema, self.raw_tool_map = build_tool_registry()
        self.tools_desc_str = json.dumps(self.tools_schema, indent=2, ensure_ascii=False)

        # --- System prompt (cached) ---
        if Orchestrator._cached_system_prompt is None:
            Orchestrator._cached_system_prompt = get_system_prompt(
                agent_name=self.agent_name,
                agent_story=self.agent_story,
                tools_description=self.tools_desc_str,
                max_concurrent_query=self.max_concurrent_query,
            )
        self.system_prompt = Orchestrator._cached_system_prompt

        # --- Tokenizer ---
        tm_cfg = self.config.get('token_management', {})
        self._tokenizer_model = os.getenv(
            tm_cfg.get('tokenizer_model_env', 'TOKENIZER_MODEL_NAME'),
            tm_cfg.get('tokenizer_model_default', '')
        ) or os.getenv('LLM_MODEL_NAME', '')
        self.tokenizer = Tokenizer(model_name=self._tokenizer_model)

        self.system_prompt_reservation = int(tm_cfg.get('system_prompt_reservation', 500))
        self.history: List[Tuple[str, str]] = []

        # --- Input validation ---
        reasoning_cfg = self.config.get('reasoning', {})
        self.max_input_chars = reasoning_cfg.get('max_input_chars', 1000)
        self.large_input_error = reasoning_cfg.get('large_input_error', SERVER_LOAD_FALLBACK_BN)

        logger.info("GovOps Orchestrator Ready.")

    def _load_config(self, path: str) -> Dict:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error("Config load failed: %s", e)
            raise

    def _load_call_config(self):
        self.llm_call_config = self.config.get('llm_call_parameters', {})
        reasoning_cfg = self.config.get('reasoning', {})
        self.max_turns = reasoning_cfg.get('max_turns', 10)
        self.max_concurrent_query = reasoning_cfg.get('max_concurrent_query', 2)
        self.summarizer_max_tokens = int(os.getenv(
            self.config.get('summarizer', {}).get('max_tokens_env', 'SUMMARIZER_MAX_TOKENS'),
            str(self.config.get('summarizer', {}).get('max_tokens_default', 300)),
        ))

    def _initialize_llm(self) -> AsyncLLMService:
        config_llm = _load_endpoint_config(self.config, 'llm')
        config_secondary = _load_endpoint_config(self.config, 'secondary')
        return AsyncLLMService(
            config_llm=config_llm,
            config_secondary=config_secondary,
        )

    def _build_tool_context(self, user_id: Optional[str]) -> ToolContext:
        return ToolContext(
            user_id=user_id,
            store=self.redis_store,
            secondary_client=getattr(self.llm_service, 'client_secondary', None),
            secondary_model=(self.llm_service.llm_config.model
                             if (self.llm_service.llm_config and hasattr(self.llm_service, 'llm_config')) else ""),
            tool_map=self.raw_tool_map,
        )

    async def process_query(
        self,
        user_query: str,
        user_id: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Main pipeline:
        1. Reconstruct context from Redis (summary + recent turns).
        2. Truncate to token budget.
        3. Run the reasoning loop.
        4. Persist turn + kick off summarizer.
        """
        logger.info("Processing Query: %s", user_query)
        original_query = user_query

        try:
            turn_id = str(uuid.uuid4())[:8]

            # --- Build date line ---
            now_bdt = datetime.now(timezone(timedelta(hours=6)))
            date_line = f"\n\nCurrent date (Bangladesh time): {now_bdt.strftime('%d %B %Y, %A')}"

            # --- Rolling summary ---
            summary = ""
            if user_id and self.redis_store.available:
                summary = self.redis_store.get_summary(user_id)
            summary_delta = f"\n\nRecent conversation summary:\n{summary}" if summary else ""

            # --- Recent turns from Redis ---
            history_messages: List[Dict[str, str]] = []
            if user_id and self.redis_store.available:
                turns = self.redis_store.get_recent_turns(user_id, n=4)
                for turn in reversed(turns):  # oldest first
                    user_text = turn.get("user", "")
                    assistant_text = turn.get("assistant", "")
                    if user_text:
                        history_messages.append({"role": "user", "content": user_text})
                    if assistant_text:
                        # Strip <thinking> tags and UI expanders
                        import re as _re
                        clean = _re.sub(r"<thinking>.*?</thinking>\s*", "", assistant_text, flags=_re.DOTALL)
                        clean = _re.sub(r"(Reasoning|Tool Logs).*?(?=\n\n|\Z)", "", clean, flags=_re.DOTALL | _re.IGNORECASE)
                        clean = _re.sub(r"\n{3,}", "\n\n", clean).strip()
                        history_messages.append({"role": "assistant", "content": clean})

            # --- Build messages ---
            messages = [
                {"role": "system", "content": self.system_prompt},
            ]
            messages.extend(history_messages)
            messages.append({
                "role": "user",
                "content": date_line + summary_delta + "\n\n" + user_query,
            })

            # --- Truncate to token budget ---
            max_ctx = self.llm_service.max_context_tokens or 32000
            budget = max(max_ctx - self.system_prompt_reservation, 2048)
            messages = truncate_messages_to_budget(
                messages, max_tokens=budget, keep_system=True, model_name=self._tokenizer_model,
            )

            # --- Extra body ---
            extra_body: Dict[str, Any] = {'max_tokens': self.llm_call_config.get('max_tokens', 2048)}
            cfg = self.llm_call_config
            if cfg:
                tg = cfg.get('thinking_general', {})
                extra_body['temperature'] = tg.get('temperature', 1.0)
                extra_body['top_p'] = tg.get('top_p', 0.95)
                extra_body['top_k'] = tg.get('top_k', 20)
                extra_body['min_p'] = tg.get('min_p', 0.0)
                extra_body['presence_penalty'] = tg.get('presence_penalty', 1.5)
                extra_body['repetition_penalty'] = tg.get('repetition_penalty', 1.0)
                if self.llm_service.llm_config and self.llm_service.llm_config.thinking:
                    budget = self.llm_service.max_context_tokens // 2
                    extra_body['thinking'] = {'type': 'enabled', 'budget_tokens': budget}

            # --- Bind tools ---
            ctx = self._build_tool_context(user_id)
            bound_tool_map = bind_tools(self.raw_tool_map, ctx)

            # --- Run reasoning loop ---
            answer_accumulator: List[str] = []

            try:
                # Get post_tool_refine config
                threshold_tokens = 800
                try:
                    from cogops.config.loader import load_config
                    cfg = load_config()
                    pt_refine = cfg.get("post_tool_refine", {})
                    threshold_tokens = pt_refine.get("threshold_tokens", 800) if pt_refine.get("enabled", True) else 0
                except Exception:
                    pass

                stream_gen = stream_with_tool_calls(
                    client_llm=self.llm_service.client_llm,
                    model=self.llm_service.model,
                    messages=messages,
                    tools_schema=self.tools_schema,
                    available_tools=bound_tool_map,
                    max_turns=self.max_turns,
                    extra_body=extra_body,
                    client_secondary=getattr(self.llm_service, 'client_secondary', None),
                    secondary_model=(self.llm_service.llm_config.model
                                     if self.llm_service.llm_config else ""),
                    threshold_tokens=threshold_tokens,
                )

                async for event in stream_gen:
                    yield event
                    if event.get("type") == "answer_chunk":
                        answer_accumulator.append(event.get("content", ""))

                final_answer = "".join(answer_accumulator).strip()

                yield {"type": "answer_complete", "channel": "both", "turn_id": turn_id}

                # --- Persist to Redis ---
                if user_id and final_answer:
                    self.redis_store.store_turn(user_id, {
                        "turn_id": turn_id,
                        "user": original_query,
                        "assistant": final_answer,
                    })

                    self.redis_store.set_last_assistant_meta(user_id, {
                        "assistant_text": final_answer,
                        "turn_id": turn_id,
                    })

                    # Async summarizer
                    asyncio.create_task(run_summarizer_task(
                        secondary_client=self.llm_service.client_secondary,
                        secondary_model=(self.llm_service.llm_config.model
                                         if self.llm_service.llm_config else ""),
                        user_id=user_id,
                        store=self.redis_store,
                        user_turn=original_query,
                        assistant_turn=final_answer,
                        max_tokens=self.summarizer_max_tokens,
                    ))

            except Exception as e:
                logger.error("Error in reasoning loop: %s", e, exc_info=True)
                yield {
                    "type": "answer_chunk",
                    "content": SERVER_LOAD_FALLBACK_BN,
                    "channel": "both",
                }
                yield {"type": "answer_complete", "channel": "both", "turn_id": turn_id}

        except Exception as e:
            logger.critical("Unhandled error in Orchestrator.process_query: %s", e, exc_info=True)
            yield {
                "type": "answer_chunk",
                "content": SERVER_LOAD_FALLBACK_BN,
                "channel": "both",
            }
            yield {"type": "answer_complete", "channel": "both"}

    def clear_session(self):
        self.history: List[Tuple[str, str]] = []
        logger.info("Session cleared.")
