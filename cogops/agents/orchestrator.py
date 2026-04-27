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
import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import AsyncGenerator, Dict, Any, List, Tuple, Optional

import yaml
from dotenv import load_dotenv

from cogops.config.loader import _load_endpoint_config
from cogops.llm.clients import AsyncLLMService
from cogops.prompts.system import get_system_prompt
from cogops.session.redis_store import RedisSessionStore
from cogops.session.summarizer import run_summarizer_task
from cogops.tools.registry import build_tool_registry, bind_tools, ToolContext
from cogops.utils.tokenizer import Tokenizer
from cogops.utils.truncate import truncate_messages_to_budget
from cogops.llm.reasoning_loop import _make_event

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _is_short_followup(text: str, max_chars: int = 16) -> bool:
    s = text.strip()
    if not s:
        return False
    if len(s) <= max_chars:
        return True
    # Also catch patterns like "number 3", "option 2", "the second one".
    low = s.lower()
    if any(kw in low for kw in ("second one", "third one", "first one", "last one",
                                "tell me more", "more details", "that one")):
        return True
    return bool(re.fullmatch(r"\d+[.)\s]*", s))


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

        thinking = self.config.get('llm', {}).get('thinking', True)
        if Orchestrator._cached_system_prompt is None:
            Orchestrator._cached_system_prompt = get_system_prompt(
                agent_name=self.agent_name,
                agent_story=self.agent_story,
                tools_description=self.tools_desc_str,
                max_concurrent_query=self.max_concurrent_query,
                thinking=thinking,
            )
        self.system_prompt = Orchestrator._cached_system_prompt

        self.feedback_history: List[Dict[str, Any]] = []
        self.history: List[Tuple[str, str]] = []

        # Shared tokenizer: used both for active context truncation AND for the
        # post-tool refine threshold check in the reasoning loop.
        tm_config = self.config.get('token_management', {})
        self._tokenizer_model_name = os.getenv(
            tm_config.get('tokenizer_model_env', 'TOKENIZER_MODEL_NAME'),
            "",
        ) or tm_config.get('tokenizer_model_default', '') \
             or os.getenv('LLM_MODEL_NAME', '')
        self.tokenizer = Tokenizer(model_name=self._tokenizer_model_name)

        self.system_prompt_reservation = int(tm_config.get('system_prompt_reservation', 3500))

        refine_cfg = self.config.get('post_tool_refine', {})
        self.post_refine_enabled = bool(refine_cfg.get('enabled', True))
        self.post_refine_threshold = int(refine_cfg.get('threshold_tokens', 600))

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
        self.response_templates = self.config.get('response_templates', {})
        reasoning_cfg = self.config.get('reasoning', {})
        self.max_turns = reasoning_cfg.get('max_turns', 10)
        self.short_followup_max_chars = reasoning_cfg.get('short_followup_max_chars', 16)
        self.max_concurrent_query = reasoning_cfg.get('max_concurrent_query', 2)
        self.max_input_chars = reasoning_cfg.get('max_input_chars', 1000)
        self.summarizer_max_tokens = int(os.getenv(
            self.config.get('summarizer', {}).get('max_tokens_env', 'SUMMARIZER_MAX_TOKENS'),
            str(self.config.get('summarizer', {}).get('max_tokens_default', 300)),
        ))

    def _initialize_llm(self) -> AsyncLLMService:
        config_llm = _load_endpoint_config(self.config, 'llm')
        config_reranker = _load_endpoint_config(self.config, 'reranker')
        config_secondary = _load_endpoint_config(self.config, 'secondary')
        return AsyncLLMService(
            config_llm=config_llm,
            config_reranker=config_reranker,
            config_secondary=config_secondary,
        )

    def _build_tool_context(self, user_id: Optional[str]) -> ToolContext:
        return ToolContext(
            user_id=user_id,
            store=self.redis_store,
            secondary_client=self.llm_service.client_secondary,
            secondary_model=(self.llm_service.llm_config.model
                             if self.llm_service.llm_config else ""),
            tool_map=self.raw_tool_map,
            tools_schema=self.tools_schema,
        )

    async def process_query(
        self,
        user_query: str,
        debug_mode: bool = False,  # kept for API compatibility; filtering is in api.py
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

            # --- Reject gibberish / oversized input -----------------------
            if len(user_query) > self.max_input_chars:
                logger.warning(f"Query too long ({len(user_query)} chars), rejecting.")
                yield {
                    "type": "error",
                    "content": "প্রশ্নটি খুব বড়। অনুগ্রহ করে সংক্ষিপ্ত প্রশ্ন করুন।",
                    "channel": "user",
                }
                yield {
                    "type": "answer_complete",
                    "channel": "both",
                    "turn_id": current_turn_id,
                }
                return

            # --- Short follow-up resolution ------------------------------
            if user_id and self.redis_store.available and _is_short_followup(original_user_query, self.short_followup_max_chars):
                last = self.redis_store.get_last_assistant_meta(user_id)
                if last and last.get('assistant_text'):
                    opts = _format_options(last.get('options') or [])
                    user_query = (
                        f"Previous assistant reply (for context):\n"
                        f"{last['assistant_text']}\n"
                        + (f"\nEnumerated options:\n{opts}\n" if opts else "")
                        + f"\nUser follow-up: {original_user_query}\n\n"
                          f"If this follow-up refers to an item in the previous "
                          f"reply, resolve it first (call history_query "
                          f"mode='recent' n=2 if you need more context), then "
                          f"call the appropriate information tool."
                    )

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
                {"role": "system", "content": self.system_prompt + _date_line + rolling_summary_delta},
                {"role": "user", "content": user_query},
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

            # --- Build extra_body + thinking toggle ---------------------
            thinking_mode = self.llm_call_config.get('thinking_general', {})
            extra_body: Dict[str, Any] = dict(thinking_mode)
            extra_body['max_tokens'] = self.llm_call_config.get('max_tokens', 2048)

            if self.config.get('llm', {}).get('thinking', False):
                ctk = dict(extra_body.get('chat_template_kwargs', {}))
                ctk['enable_thinking'] = True
                extra_body['chat_template_kwargs'] = ctk

            loop_kwargs: Dict[str, Any] = {}
            for k in ('presence_penalty', 'repetition_penalty', 'top_p', 'top_k'):
                if k in thinking_mode:
                    loop_kwargs[k] = thinking_mode[k]

            # --- Bind tools with per-request context --------------------
            ctx = self._build_tool_context(user_id)
            bound_tool_map = bind_tools(self.raw_tool_map, ctx)

            from cogops.llm.reasoning_loop import stream_with_tool_calls

            try:
                stream_gen = stream_with_tool_calls(
                    client_llm=self.llm_service.client_llm,
                    model=self.llm_service.model,
                    messages=messages,
                    tools_schema=self.tools_schema,
                    available_tools=bound_tool_map,
                    max_turns=self.max_turns,
                    extra_body=extra_body,
                    user_query=original_user_query,
                    secondary_client=(self.llm_service.client_secondary
                                      if self.post_refine_enabled else None),
                    secondary_model=(self.llm_service.llm_config.model
                                     if (self.post_refine_enabled
                                         and self.llm_service.llm_config) else ""),
                    tokenizer=self.tokenizer if self.post_refine_enabled else None,
                    refine_threshold_tokens=(self.post_refine_threshold
                                             if self.post_refine_enabled else 0),
                    **loop_kwargs,
                )

                async for event in stream_gen:
                    yield event

                    if event["type"] == "answer_chunk":
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

                    # Persist the last assistant reply + any options it offered
                    # so the next short follow-up can be resolved.
                    self.redis_store.set_last_assistant_meta(user_id, {
                        "assistant_text": final_response,
                        "options": _extract_enumerated_options(final_response),
                        "turn_id": current_turn_id,
                    })

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
                yield {
                    "type": "error",
                    "content": f"Error: {e}",
                    "channel": "user",
                }

        except Exception as e:
            logger.error(f"Critical Error in Orchestrator: {e}", exc_info=True)
            yield {
                "type": "error",
                "content": self.response_templates.get(
                    'error_fallback', "Error occurred."),
                "channel": "user",
            }

    def clear_session(self):
        self.history = []
        self.feedback_history = []
        logger.info("Session cleared.")

    def add_feedback(self, user_id: str, turn_id: str, rating: str,
                     comment: str = "") -> None:
        entry = {
            "turn_id": turn_id,
            "rating": rating,
            "comment": comment,
            "timestamp": str(uuid.uuid4())[:8],
        }
        self.feedback_history.append(entry)
        if len(self.feedback_history) > 5:
            self.feedback_history.pop(0)

    def get_negative_feedback(self) -> str:
        negatives = [f for f in self.feedback_history
                     if f["rating"] in ("bad", "unhelpful", "wrong")]
        if not negatives:
            return ""
        lines = ["Recent negative feedback:"]
        for f in negatives:
            lines.append(
                f"Turn {f['turn_id']}: rating={f['rating']}, "
                f"comment='{f.get('comment', '')}'"
            )
        return "\n".join(lines)


_ENUMERATED_LINE_RE = re.compile(
    r"^\s*(?:\d+[.)]|[-*•])\s+(.+)$"
)


def _format_options(options: List[str]) -> str:
    if not options:
        return ""
    return "\n".join(f"- {o}" for o in options)


def _extract_enumerated_options(text: str) -> List[str]:
    """Pull bullet/numbered items out of an assistant reply so short follow-ups
    like '3' can resolve to the third item. Works for both numbered (1., 2))
    and bullet (-, *, •) enumerations."""
    if not text:
        return []
    out: List[str] = []
    for line in text.splitlines():
        m = _ENUMERATED_LINE_RE.match(line)
        if m:
            out.append(m.group(1).strip())
    return out
