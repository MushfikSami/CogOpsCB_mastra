"""
cogops/agents/orchestrator.py

The Orchestrator for the GovOps Agent.

- Loads config, builds tool registry + system prompt.
- Per-request: binds tools with a ToolContext, handles short ambiguous
  follow-ups, truncates messages to the token budget, runs the reasoning
  loop, persists turn + last-assistant meta to Redis, kicks off the async
  summarizer.
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
        self._load_llm_call_config()

        self.agent_name = self.config.get('agent_name', 'Gov Assistant')
        self.agent_story = self.config.get('agent_story', '')

        self.llm_service = self._initialize_llm()

        session_config = self.config.get('session', {})
        redis_url = os.getenv(session_config.get('redis_url_env', 'REDIS_URL'),
                              session_config.get('redis_url_default', "redis://localhost:6379/0"))
        ttl = int(os.getenv(session_config.get('ttl_seconds_env', 'REDIS_SESSION_TTL_SECONDS'),
                            str(session_config.get('ttl_default', 86400))))
        self.redis_store = RedisSessionStore(url=redis_url, ttl_seconds=ttl)

        self.tools_schema, self.raw_tool_map = build_tool_registry()
        self.tools_desc_str = json.dumps(self.tools_schema, indent=2, ensure_ascii=False)

        if Orchestrator._cached_system_prompt is None:
            Orchestrator._cached_system_prompt = get_system_prompt(
                agent_name=self.agent_name,
                agent_story=self.agent_story,
                tools_description=self.tools_desc_str,
                max_concurrent_query=self.max_concurrent_query,
            )
        self.system_prompt = Orchestrator._cached_system_prompt

        self.history: List[Tuple[str, str]] = []

        # Shared tokenizer: used for active context truncation
        tm_config = self.config.get('token_management', {})
        self._tokenizer_model_name = os.getenv(
            tm_config.get('tokenizer_model_env', 'TOKENIZER_MODEL_NAME'),
            "",
        ) or tm_config.get('tokenizer_model_default', '') \
             or os.getenv('LLM_MODEL_NAME', '')
        self.tokenizer = Tokenizer(model_name=self._tokenizer_model_name)

        self.system_prompt_reservation = int(tm_config.get('system_prompt_reservation', 3500))

        logger.info("GovOps Orchestrator Ready.")

    def _load_config(self, path: str) -> Dict:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Config load failed: {e}")
            raise

    def _load_llm_call_config(self):
        self.llm_call_config = self.config.get('llm_call_parameters', {})
        reasoning_cfg = self.config.get('reasoning', {})
        self.max_turns = reasoning_cfg.get('max_turns', 10)
        self.max_concurrent_query = reasoning_cfg.get('max_concurrent_query', 2)
        self.max_input_chars = reasoning_cfg.get('max_input_chars', 1000)
        self.large_input_error = reasoning_cfg.get('large_input_error', "প্রশ্নটি খুব বড়। অনুগ্রহ করে সংক্ষিপ্ত প্রশ্ন করুন।")
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
            secondary_client=(self.llm_service.client_secondary
                              if hasattr(self.llm_service, 'client_secondary') else None),
            secondary_model=(self.llm_service.llm_config.model
                             if (self.llm_service.llm_config and hasattr(self.llm_service, 'llm_config')) else ""),
            tool_map=self.raw_tool_map,
            tools_schema=self.tools_schema,
        )

    async def process_query(
        self,
        user_query: str,
        user_id: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Main pipeline:
        1. If the message is short/numeric and the user has a prior assistant
           reply stored, inject that reply as context so the model can resolve
           the reference (e.g. "3" -> third item in the list).
        2. Build [system + rolling summary, user_query], truncate to token
           budget, run the reasoning loop with a per-request bound tool map.
        3. Stream events; persist turn + last-assistant-meta; trigger
           summarizer in the background.
        """
        logger.info(f"Processing Query: {user_query}")
        original_user_query = user_query

        try:
            current_turn_id = str(uuid.uuid4())[:8]

            # --- Build messages -----------------------------------------
            _now_bdt = datetime.now(timezone(timedelta(hours=6)))
            _date_line = (
                f"\n\nCurrent date (Bangladesh time): {_now_bdt.strftime('%d %B %Y, %A')}"
            )
            summary = ""
            if user_id and self.redis_store.available:
                summary = self.redis_store.get_summary(user_id)
            rolling_summary_delta = (
                f"\n\nRecent conversation summary:\n{summary}" if summary else ""
            )

            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": _date_line + rolling_summary_delta + "\n\n" + user_query},
            ]

            # Active token-budget truncation (before the first primary-LLM call).
            max_ctx = self.llm_service.max_context_tokens or 32000
            budget = max(max_ctx - self.system_prompt_reservation, 2048)
            messages = truncate_messages_to_budget(
                messages,
                max_tokens=budget,
                keep_system=True,
                model_name=self._tokenizer_model_name,
            )

            full_answer_accumulator: List[str] = []

            # --- Build extra_body ---------------------
            extra_body: Dict[str, Any] = {
                'max_tokens': self.llm_call_config.get('max_tokens', 2048),
            }

            # --- Bind tools with per-request context --------------------
            ctx = self._build_tool_context(user_id)
            bound_tool_map = bind_tools(self.raw_tool_map, ctx)

            try:
                stream_gen = stream_with_tool_calls(
                    client_llm=self.llm_service.client_llm,
                    model=self.llm_service.model,
                    messages=messages,
                    tools_schema=self.tools_schema,
                    available_tools=bound_tool_map,
                    max_turns=self.max_turns,
                    extra_body=extra_body,
                )

                async for event in stream_gen:
                    # Forward every event to the caller; the API layer filters
                    # by channel based on the debug header. Answer chunks are
                    # also accumulated locally so we can persist the final
                    # response to Redis once the stream completes.
                    yield event
                    if event.get("type") == "answer_chunk":
                        full_answer_accumulator.append(event.get("content", ""))

                final_response = "".join(full_answer_accumulator).strip()

                yield {
                    "type": "answer_complete",
                    "channel": "both",
                    "turn_id": current_turn_id,
                }

                if user_id and final_response:
                    turn = {
                        "turn_id": current_turn_id,
                        "user": original_user_query,
                        "assistant": final_response,
                    }
                    self.redis_store.store_turn(user_id, turn)
                    self.history.append((original_user_query, final_response))

                    self.redis_store.set_last_assistant_meta(user_id, {
                        "assistant_text": final_response,
                        "turn_id": current_turn_id,
                    })

                    # Rolling summary in background via secondary LLM
                    asyncio.create_task(run_summarizer_task(
                        secondary_client=self.llm_service.client_secondary,
                        secondary_model=(self.llm_service.llm_config.model
                                         if self.llm_service.llm_config else ""),
                        user_id=user_id,
                        store=self.redis_store,
                        user_turn=original_user_query,
                        assistant_turn=final_response,
                        max_tokens=self.summarizer_max_tokens,
                    ))

            except Exception as e:
                logger.error(f"Error in reasoning loop: {e}", exc_info=True)
                # The reasoning loop emits its own user-facing fallback chunk
                # before raising; here we just log and signal completion.
                yield {
                    "type": "error",
                    "content": f"Error: {e}",
                    "channel": "debug",
                }
                yield {
                    "type": "answer_complete",
                    "channel": "both",
                    "turn_id": current_turn_id,
                }

        except Exception as e:
            logger.critical(
                f"Unhandled error in Orchestrator.process_query: {e}",
                exc_info=True,
            )
            yield {
                "type": "answer_chunk",
                "content": SERVER_LOAD_FALLBACK_BN,
                "channel": "both",
            }
            yield {
                "type": "answer_complete",
                "channel": "both",
            }

    def clear_session(self):
        self.history = []
        logger.info("Session cleared.")

