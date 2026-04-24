"""
cogops/agents/orchestrator.py

The Orchestrator for the GovOps Agent.
Loads config, builds tool registry, initializes LLM/Graphiti, creates rolling summary in Redis.
process_query() delegates to run_reasoning_loop and handles clarification/history/feedback.
"""

import os
import json
import asyncio
import logging
import uuid
import yaml
from typing import AsyncGenerator, Dict, Any, List, Tuple, Optional

from dotenv import load_dotenv

from cogops.config.loader import _load_endpoint_config, load_config
from cogops.llm.clients import AsyncLLMService
from cogops.prompts.system import get_graph_prompt
from cogops.session.redis_store import RedisSessionStore
from cogops.session.summarizer import run_summarizer_task
from cogops.tools.registry import build_tool_registry
from cogops.tools.ask_user import ClarificationRequested
from cogops.utils.tokenizer import Tokenizer

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Orchestrator:
    _cached_system_prompt: Optional[str] = None

    def __init__(self, config_path: str = "configs/config.yml"):
        logger.info("Initializing GovOps Orchestrator...")
        self.config = self._load_config(config_path)
        self._load_llm_call_config()

        # --- Identity ---
        self.agent_name = self.config.get('agent_name', 'Gov Assistant')
        self.agent_story = self.config.get('agent_story', '')

        # --- LLM Service ---
        self.llm_service = self._initialize_llm()

        # --- Session & Redis ---
        session_config = self.config.get('session', {})
        redis_url = os.getenv(session_config.get('redis_url_env', 'REDIS_URL'), "redis://localhost:6379/0")
        ttl = int(os.getenv(session_config.get('ttl_seconds_env', 'REDIS_SESSION_TTL_SECONDS'), '86400'))
        self.redis_store = RedisSessionStore(url=redis_url, ttl_seconds=ttl)

        # --- Tool Registry ---
        self.tools_schema, self.tool_map = build_tool_registry(
            secondary_client=self.llm_service.client_secondary,
            secondary_model=self.llm_service.llm_config.model if self.llm_service.llm_config else "",
        )
        self.tools_desc_str = json.dumps(self.tools_schema, indent=2, ensure_ascii=False)

        # --- System Prompt ---
        # Built once per class (immutable, identical for all users).
        # Each instance references the same string — no per-user allocation.
        if Orchestrator._cached_system_prompt is None:
            Orchestrator._cached_system_prompt = get_graph_prompt(
                agent_name=self.agent_name,
                agent_story=self.agent_story,
                tools_description=self.tools_desc_str,
            )
        self.system_prompt = Orchestrator._cached_system_prompt

        # --- Feedback storage ---
        self.feedback_history: List[Dict[str, Any]] = []  # recent negative feedback

        # --- History ---
        self.history: List[Tuple[str, str]] = []

        # --- Tokenizer ---
        tm_config = self.config['token_management']
        self.tokenizer = Tokenizer(
            model_name=os.getenv(tm_config['tokenizer_model_env'], "Qwen/Qwen2.5-32B-Instruct"),
        )

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
        self.response_templates = self.config['response_templates']
        self.max_turns = self.config.get('reasoning', {}).get('max_turns', 10)
        self.summarizer_max_tokens = int(os.getenv(
            self.config.get('summarizer', {}).get('max_tokens_env', 'SUMMARIZER_MAX_TOKENS'), '300'
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

    async def process_query(self, user_query: str, debug_mode: bool = False,
                            user_id: Optional[str] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Main pipeline:
        1. Build messages: [system_prompt, rolling_summary_delta, user_query]
        2. Handle clarification reply if pending (stored temporarily in Redis)
        3. Run reasoning loop
        4. Buffer answer, update Redis (store turn + re-summarize)

        Clarification state is stored in Redis with a TTL (default 1 day).
        If the user doesn't reply, the key expires automatically — no cleanup needed.
        """
        logger.info(f"Processing Query: {user_query}")

        try:
            # Check for pending clarification reply
            clarification_reply = None
            if user_id and self.redis_store.available:
                pending = self.redis_store.get_clarification(user_id)
                if pending:
                    clarification_reply = pending
                    self.redis_store.clear_clarification(user_id)
                    # Inject clarification Q+A into history
                    self.history.append((pending['question'], user_query))
                    # Start fresh reasoning with clarification context
                    user_query = f"Clarification reply: {user_query}. Context from previous turn:\n{pending['question']}"

            # Build messages
            summary = ""
            if user_id and self.redis_store.available:
                summary = self.redis_store.get_summary(user_id)
            rolling_summary_delta = ""
            if summary:
                rolling_summary_delta = f"\n\nRecent conversation summary:\n{summary}"

            messages = [
                {"role": "system", "content": self.system_prompt + rolling_summary_delta},
                {"role": "user", "content": user_query}
            ]

            full_answer_accumulator = []
            current_turn_id = str(uuid.uuid4())[:8]

            thinking_mode = self.llm_call_config.get('thinking_general', {})
            extra_body = dict(thinking_mode)
            extra_body['max_tokens'] = self.llm_call_config.get('max_tokens', 2048)

            # Pass extra params to reasoning loop
            loop_kwargs = {}
            if 'presence_penalty' in thinking_mode:
                loop_kwargs['presence_penalty'] = thinking_mode['presence_penalty']
            if 'repetition_penalty' in thinking_mode:
                loop_kwargs['repetition_penalty'] = thinking_mode['repetition_penalty']

            from cogops.llm.reasoning_loop import stream_with_tool_calls

            try:
                stream_gen = stream_with_tool_calls(
                    client_llm=self.llm_service.client_llm,
                    model=self.llm_service.model,
                    messages=messages,
                    tools_schema=self.tools_schema,
                    available_tools=self.tool_map,
                    debug_mode=debug_mode,
                    max_turns=self.max_turns,
                    extra_body=extra_body,
                    **loop_kwargs
                )

                clarification_data = None
                is_answer_complete = False

                async for event in stream_gen:
                    # Handle clarification_needed specially
                    if event.get("type") == "clarification_needed":
                        clarification_data = {
                            "question": event.get("question", ""),
                            "options": event.get("options", []),
                            "reason": event.get("reason", ""),
                            "turn_id": event.get("turn_id", current_turn_id),
                        }
                        if user_id and self.redis_store.available:
                            self.redis_store.set_clarification(user_id, clarification_data)
                        # Yield clarification event on user channel
                        yield {
                            "type": "clarification_needed",
                            "channel": "user",
                            "question": clarification_data["question"],
                            "options": clarification_data["options"],
                            "reason": clarification_data.get("reason", ""),
                            "turn_id": clarification_data["turn_id"],
                        }
                        break  # Stream ends

                    # Yield event through
                    yield event

                    if event["type"] == "answer_chunk":
                        full_answer_accumulator.append(event.get("content", ""))

                final_response = "".join(full_answer_accumulator).strip()

                # Mark answer complete
                yield {
                    "type": "answer_complete",
                    "channel": "both",
                    "turn_id": current_turn_id,
                }

                # Update Redis if user_id provided
                if user_id and final_response:
                    turn = {
                        "turn_id": current_turn_id,
                        "user": user_query,
                        "assistant": final_response,
                    }
                    self.redis_store.store_turn(user_id, turn)
                    self.history.append((user_query, final_response))

                    # Background summarizer
                    asyncio.create_task(run_summarizer_task(
                        secondary_client=self.llm_service.client_secondary,
                        secondary_model=self.llm_service.llm_config.model if self.llm_service.llm_config else "",
                        user_id=user_id,
                        store=self.redis_store,
                        user_turn=user_query,
                        assistant_turn=final_response,
                        max_tokens=self.summarizer_max_tokens,
                    ))

            except ClarificationRequested as ce:
                # The ask_user tool raised this
                clarification_data = {
                    "question": ce.question,
                    "options": ce.options,
                    "reason": ce.reason,
                    "turn_id": ce.turn_id,
                }
                if user_id and self.redis_store.available:
                    self.redis_store.set_clarification(user_id, clarification_data)
                yield {
                    "type": "clarification_needed",
                    "channel": "user",
                    "question": ce.question,
                    "options": ce.options,
                    "reason": ce.reason,
                    "turn_id": ce.turn_id,
                }

        except Exception as e:
            logger.error(f"Critical Error in Orchestrator: {e}", exc_info=True)
            yield {"type": "error", "content": self.response_templates.get('error_fallback', "Error occurred."), "channel": "user"}

    def clear_session(self):
        """Resets the memory and Redis store."""
        self.history = []
        self.feedback_history = []
        logger.info("Session cleared.")

    def add_feedback(self, user_id: str, turn_id: str, rating: str, comment: str = "") -> None:
        """Store feedback. Only negative feedback is surfaced to the system."""
        entry = {"turn_id": turn_id, "rating": rating, "comment": comment, "timestamp": str(uuid.uuid4())[:8]}
        self.feedback_history.append(entry)
        # Keep only last 5
        if len(self.feedback_history) > 5:
            self.feedback_history.pop(0)

    def get_negative_feedback(self) -> str:
        """Return formatted negative feedback for system context injection."""
        negatives = [f for f in self.feedback_history if f["rating"] in ("bad", "unhelpful", "wrong")]
        if not negatives:
            return ""
        lines = ["Recent negative feedback:", ]
        for f in negatives:
            lines.append(f"Turn {f['turn_id']}: rating={f['rating']}, comment='{f.get('comment', '')}'")
        return "\n".join(lines)


# Backward compat
GraphitiAgent = Orchestrator
