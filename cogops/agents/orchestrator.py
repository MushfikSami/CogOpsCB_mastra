"""
cogops/agents/orchestrator.py

The Orchestrator for the GovOps Agent.

- Loads config, builds tool registry + system prompt.
- Per-request: reconstructs context from Redis, runs the reasoning loop,
  persists results.
- Session recovery: on startup, loads existing Redis turns so conversations
  survive server restarts.
"""

import logging
import os
import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncGenerator, Dict, List, Optional

import yaml
from dotenv import load_dotenv

from cogops.config.loader import _load_endpoint_config
from cogops.llm.clients import AsyncLLMService
from cogops.llm.reasoning_loop import stream_with_tool_calls
from cogops.prompts.messages import SERVER_LOAD_FALLBACK_BN
from cogops.prompts.system import get_system_prompt
from cogops.session.redis_store import RedisSessionStore
from cogops.tools.registry import build_tool_registry, bind_tools, ToolContext

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

        # --- System prompt (cached) ---
        if Orchestrator._cached_system_prompt is None:
            Orchestrator._cached_system_prompt = get_system_prompt(
                agent_name=self.agent_name,
                agent_story=self.agent_story,
                max_concurrent_query=self.max_concurrent_query,
            )
        self.system_prompt = Orchestrator._cached_system_prompt

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

    def _initialize_llm(self) -> AsyncLLMService:
        config_llm = _load_endpoint_config(self.config, 'llm')
        return AsyncLLMService(config_llm=config_llm)

    def _build_tool_context(self, user_id: Optional[str]) -> ToolContext:
        return ToolContext(
            user_id=user_id,
            store=self.redis_store,
        )

    async def process_query(
        self,
        user_query: str,
        user_id: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Main pipeline:
        1. Reconstruct context from Redis (recent turns only — no summarizer).
        2. Run the reasoning loop.
        3. Persist result to Redis.
        """
        logger.info("Processing Query: %s", user_query)
        original_query = user_query

        try:
            turn_id = str(uuid.uuid4())[:8]

            # --- Build date line ---
            now_bdt = datetime.now(timezone(timedelta(hours=6)))
            date_line = f"\n\nCurrent date (Bangladesh time): {now_bdt.strftime('%d %B %Y, %A')}"

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
                        clean = re.sub(r"<thinking>.*?</thinking>\s*", "", assistant_text, flags=re.DOTALL)
                        clean = re.sub(r"(Reasoning|Tool Logs).*?(?=\n\n|\Z)", "", clean, flags=re.DOTALL | re.IGNORECASE)
                        clean = re.sub(r"\n{3,}", "\n\n", clean).strip()
                        history_messages.append({"role": "assistant", "content": clean})

            # --- Build messages ---
            messages = [
                {"role": "system", "content": self.system_prompt},
            ]
            messages.extend(history_messages)
            messages.append({
                "role": "user",
                "content": date_line + "\n\n" + user_query,
            })

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

            # --- Bind tools ---
            ctx = self._build_tool_context(user_id)
            bound_tool_map = bind_tools(self.raw_tool_map, ctx)

            # --- Run reasoning loop ---
            answer_accumulator: List[str] = []

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
        logger.info("Session cleared.")
