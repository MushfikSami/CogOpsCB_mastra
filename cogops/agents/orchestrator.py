"""
cogops/agents/orchestrator.py

Per-request façade for the deterministic chatbot pipeline.

Flow per request:

  1. SANITIZE  (cogops.pipeline.sanitize)
     → on failure: static input-invalid refusal, persist, return.

  2. HISTORY   (load recent turns from Redis)

  3. ROUTER    (cogops.pipeline.router.route)
     →  intent ∈ {factual_govt, chitchat, political_refuse}

  4a. political_refuse → static neutral refusal, persist, return.
  4b. chitchat         → canned bilingual greeting, persist, return.
  4c. factual_govt     → run cogops.agents.pipeline.run_factual_pipeline
                          (Jiggasha vector retrieve → LLM relevance filter →
                          composer stream → non-blocking NLI verify).
                          Forward all events; persist the final post-flight
                          answer to Redis.

This orchestrator does NOT do its own ReAct loop, tool routing, or grounding
post-processing — those are all encapsulated inside `run_factual_pipeline`.
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

import yaml

from cogops.agents.pipeline import PipelineConfig, run_factual_pipeline
from cogops.config.loader import _load_endpoint_config
from cogops.llm.clients import AsyncLLMService
from cogops.pipeline.context_resolve import resolve_references
from cogops.pipeline.router import route as router_route
from cogops.pipeline.sanitize import INPUT_INVALID_REFUSAL_BN, sanitize
from cogops.prompts.messages import SERVER_LOAD_FALLBACK_BN
from cogops.session.redis_store import RedisSessionStore

logger = logging.getLogger(__name__)


_DEFAULT_REFUSAL_BN = (
    "দুঃখিত, এই প্রশ্নের জন্য নির্ভরযোগ্য সরকারি তথ্য পাওয়া যায়নি।"
)

_POLITICAL_REFUSAL_BN = (
    "আমি একটি নিরপেক্ষ সরকারি সেবা সহকারী। রাজনৈতিক বা ধর্মীয় বিষয়ে মতামত "
    "দেওয়া আমার পক্ষে সম্ভব নয়। অন্য কোনো সরকারি সেবায় কি আমি সাহায্য করতে পারি?"
)

_PERSONAL_LAW_REFUSAL_BN = (
    "এই প্রশ্নটি ব্যক্তিগত আইন বা ধর্মীয় বিধানের পরামর্শের বিষয় — এতে "
    "নির্ভরযোগ্য সরকারি সেবার তথ্য নেই। সঠিক উত্তরের জন্য সংশ্লিষ্ট বিশেষজ্ঞের "
    "(আইনজীবী, কাজী অফিস, বা মুফতি) সাথে পরামর্শ করুন।"
)

_CHITCHAT_GREETING_BN = (
    "স্বাগতম! আমি বাংলাদেশ সরকারের সেবা সম্পর্কিত প্রশ্নে সাহায্য করতে পারি — "
    "যেমন এনআইডি, পাসপোর্ট, ট্যাক্স, সনদ, ইত্যাদি। আপনি কোন বিষয়ে জানতে চান?"
)


class Orchestrator:
    """One per `user_id`. Holds LLM clients + Redis store + pipeline config."""

    def __init__(self, config_path: str = "configs/config.yml"):
        logger.info("Initializing GovOps Orchestrator…")
        self.config = self._load_config(config_path)

        self.agent_name = self.config.get("agent", {}).get("name", "GovOps সহকারী")

        # Primary LLM (composer) and Secondary LLM (router + rerank + NLI)
        self.llm_service = self._initialize_llm("llm")
        self.secondary_service = self._initialize_llm("secondary")

        # Redis session store
        session_cfg = self.config.get("session", {}) or {}
        redis_url = os.getenv(
            session_cfg.get("redis_url_env", "REDIS_URL"),
            session_cfg.get("redis_url_default", "redis://localhost:6379/0"),
        )
        ttl = int(os.getenv(
            session_cfg.get("ttl_seconds_env", "REDIS_SESSION_TTL_SECONDS"),
            str(session_cfg.get("ttl_default", 86400)),
        ))
        self.redis_store = RedisSessionStore(url=redis_url, ttl_seconds=ttl)

        # Refusal templates (config-overridable)
        retrieval_cfg = self.config.get("retrieval", {}) or {}
        self.refusal_text = retrieval_cfg.get("refusal_text_bn", _DEFAULT_REFUSAL_BN)
        self.political_refusal_text = retrieval_cfg.get(
            "political_refusal_text_bn", _POLITICAL_REFUSAL_BN,
        )
        self.personal_law_refusal_text = retrieval_cfg.get(
            "personal_law_refusal_text_bn", _PERSONAL_LAW_REFUSAL_BN,
        )
        self.chitchat_greeting = retrieval_cfg.get(
            "chitchat_greeting_bn", _CHITCHAT_GREETING_BN,
        )
        self.input_invalid_refusal = retrieval_cfg.get(
            "input_invalid_refusal_bn", INPUT_INVALID_REFUSAL_BN,
        )

        # Pipeline config (knobs surfaced from config.yml under pipeline:)
        self.pipeline_cfg = self._build_pipeline_config()

        logger.info(
            "Orchestrator ready. primary=%s secondary=%s jiggasha=%s verifier=%s",
            self.llm_service.model, self.secondary_service.model,
            self.pipeline_cfg.jiggasha_endpoint, self.pipeline_cfg.verifier_enabled,
        )

    # --------------------------------------------------------------
    # Construction helpers
    # --------------------------------------------------------------
    def _load_config(self, path: str) -> Dict:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error("Config load failed: %s", e)
            raise

    def _initialize_llm(self, section: str) -> AsyncLLMService:
        config = _load_endpoint_config(self.config, section)
        return AsyncLLMService(config_llm=config)

    def _build_pipeline_config(self) -> PipelineConfig:
        pcfg = (self.config.get("pipeline") or {})
        tools_cfg = (self.config.get("tools") or {})
        jcfg = tools_cfg.get("jiggasha") or {}

        # Jiggasha endpoint resolution: env var wins, then config, then default.
        endpoint = (
            os.getenv(jcfg.get("endpoint_env", "JIGGASHA_ENDPOINT"))
            or jcfg.get("endpoint")
            or "http://localhost:10000/search"
        )
        verifier_cfg = (self.config.get("verifier") or {})
        disambig_cfg = (pcfg.get("disambiguation") or {})

        return PipelineConfig(
            jiggasha_endpoint=endpoint,
            jiggasha_timeout=float(jcfg.get("timeout_seconds", 45.0)),
            top_k_fetch=int(jcfg.get("top_k_fetch", jcfg.get("top_k", 50))),
            use_instruction=bool(jcfg.get("use_instruction", True)),
            cosine_threshold=jcfg.get("cosine_threshold"),
            token_budget=jcfg.get("token_budget"),
            composer_temperature=float(pcfg.get("composer", {}).get("temperature", 0.3)),
            composer_top_p=float(pcfg.get("composer", {}).get("top_p", 0.95)),
            composer_max_tokens=int(pcfg.get("composer", {}).get("max_tokens", 2048)),
            agent_name=self.agent_name,
            verifier_enabled=bool(verifier_cfg.get("enabled", False)),
            verifier_timeout=float(verifier_cfg.get("timeout_seconds", 6.0)),
            verifier_policy=str(verifier_cfg.get("policy", "redact")),
            disambig_min_distinct_services=int(
                disambig_cfg.get("min_distinct_services", 2)
            ),
            disambig_short_query_token_cap=int(
                disambig_cfg.get("short_query_token_cap", 6)
            ),
            disambig_candidate_cap=int(
                disambig_cfg.get("candidate_cap", 6)
            ),
            refusal_text_bn=self.refusal_text,
        )

    # --------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------
    async def process_query(
        self,
        user_query: str,
        user_id: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Run one user turn end-to-end and yield events."""
        turn_id = str(uuid.uuid4())[:8]
        original_query = user_query or ""
        logger.info("[turn:%s] Processing: %r", turn_id, original_query[:80])

        try:
            # ----- Stage 0: Sanitize -----
            clean_query, refusal_reason = sanitize(original_query)
            if refusal_reason is not None:
                yield {"type": "sanitize_verdict", "channel": "debug",
                       "reason": refusal_reason, "turn_id": turn_id}
                yield {"type": "answer_chunk", "channel": "both",
                       "content": self.input_invalid_refusal}
                yield {"type": "final_answer", "channel": "both",
                       "content": self.input_invalid_refusal,
                       "turn_id": turn_id, "source_map": {},
                       "reason": f"sanitize_{refusal_reason}"}
                yield {"type": "answer_complete", "channel": "both", "turn_id": turn_id}
                if user_id:
                    self._persist(user_id, turn_id, original_query, self.input_invalid_refusal)
                return

            # ----- History -----
            history_messages = self._load_history(user_id)
            yield {"type": "history_loaded", "channel": "debug",
                   "turns": len(history_messages),
                   "messages": [
                       {"role": m.get("role", ""),
                        "content": (m.get("content", "") or "")[:1200]}
                       for m in history_messages
                   ],
                   "turn_id": turn_id}

            # ----- Stage 0.5: Context resolution (pronouns / references) -----
            resolved_query = await resolve_references(
                query=clean_query,
                history=history_messages,
                secondary_client=self.secondary_service.client_llm,
                secondary_model=self.secondary_service.model,
                timeout=3.0,
            )
            if resolved_query != clean_query:
                yield {"type": "context_resolved", "channel": "debug",
                       "original": clean_query, "resolved": resolved_query,
                       "turn_id": turn_id}

            # ----- Stage 1: Router -----
            try:
                router_result = await router_route(
                    query=resolved_query,
                    secondary_client=self.secondary_service.client_llm,
                    secondary_model=self.secondary_service.model,
                    timeout=5.0,
                )
            except Exception as e:  # noqa: BLE001
                logger.warning("Router raised (%s); defaulting to factual_govt.", e)
                from cogops.pipeline.router import RouterResult
                router_result = RouterResult(
                    intent="factual_govt",
                    sub_queries_bengali=[clean_query],
                    raw_query=resolved_query,
                    notes=[f"router_exception: {e!s}"],
                )
            yield {"type": "router_done", "channel": "debug",
                   "intent": router_result.intent,
                   "sub_queries": router_result.sub_queries_bengali,
                   "notes": router_result.notes,
                   "token_usage": router_result.usage,
                   "turn_id": turn_id}

            # ----- Stage 4a: political_refuse -----
            if router_result.intent == "political_refuse":
                yield {"type": "answer_chunk", "channel": "both",
                       "content": self.political_refusal_text}
                yield {"type": "final_answer", "channel": "both",
                       "content": self.political_refusal_text,
                       "turn_id": turn_id, "source_map": {},
                       "reason": "political_refuse"}
                yield {"type": "answer_complete", "channel": "both", "turn_id": turn_id}
                if user_id:
                    self._persist(user_id, turn_id, original_query, self.political_refusal_text)
                return

            # ----- Stage 4a': personal_law_refuse (religious / family-law judgement) -----
            if router_result.intent == "personal_law_refuse":
                yield {"type": "answer_chunk", "channel": "both",
                       "content": self.personal_law_refusal_text}
                yield {"type": "final_answer", "channel": "both",
                       "content": self.personal_law_refusal_text,
                       "turn_id": turn_id, "source_map": {},
                       "reason": "personal_law_refuse"}
                yield {"type": "answer_complete", "channel": "both", "turn_id": turn_id}
                if user_id:
                    self._persist(user_id, turn_id, original_query, self.personal_law_refusal_text)
                return

            # ----- Stage 4b: chitchat -----
            if router_result.intent == "chitchat":
                yield {"type": "answer_chunk", "channel": "both",
                       "content": self.chitchat_greeting}
                yield {"type": "final_answer", "channel": "both",
                       "content": self.chitchat_greeting,
                       "turn_id": turn_id, "source_map": {},
                       "reason": "chitchat"}
                yield {"type": "answer_complete", "channel": "both", "turn_id": turn_id}
                if user_id:
                    self._persist(user_id, turn_id, original_query, self.chitchat_greeting)
                return

            # ----- Stage 4c: factual intents — run the deterministic pipeline -----
            # Always search the entire unified corpus (wiki + govt_service).
            # The instruction-based retrieval + threshold filter handles cross-corpus
            # relevance without needing intent-based chunk_type filtering.
            self.pipeline_cfg.chunk_type = None
            final_text_for_persist: Optional[str] = None
            try:
                async for event in run_factual_pipeline(
                    raw_query=clean_query,
                    router_result=router_result,
                    history=history_messages,
                    primary_client=self.llm_service.client_llm,
                    primary_model=self.llm_service.model,
                    secondary_client=self.secondary_service.client_llm,
                    secondary_model=self.secondary_service.model,
                    cfg=self.pipeline_cfg,
                ):
                    if event.get("type") == "final_answer":
                        final_text_for_persist = event.get("content", "") or final_text_for_persist
                    yield event
            except Exception as e:  # noqa: BLE001
                logger.error("[turn:%s] pipeline crashed: %s", turn_id, e, exc_info=True)
                yield {"type": "answer_chunk", "channel": "both",
                       "content": SERVER_LOAD_FALLBACK_BN}
                yield {"type": "final_answer", "channel": "both",
                       "content": SERVER_LOAD_FALLBACK_BN,
                       "turn_id": turn_id, "source_map": {},
                       "reason": "pipeline_error"}
                yield {"type": "answer_complete", "channel": "both", "turn_id": turn_id}

            if user_id and final_text_for_persist:
                self._persist(user_id, turn_id, original_query, final_text_for_persist)

        except Exception as e:  # noqa: BLE001
            logger.critical("[turn:%s] Unhandled in Orchestrator: %s", turn_id, e, exc_info=True)
            yield {"type": "answer_chunk", "content": SERVER_LOAD_FALLBACK_BN, "channel": "both"}
            yield {"type": "answer_complete", "channel": "both", "turn_id": turn_id}

    # --------------------------------------------------------------
    # Persistence helpers
    # --------------------------------------------------------------
    def _load_history(self, user_id: Optional[str]) -> List[Dict[str, str]]:
        """Pull recent user/assistant turns from Redis for the composer's context."""
        if not user_id or not self.redis_store.available:
            return []
        out: List[Dict[str, str]] = []
        try:
            turns = self.redis_store.get_recent_turns(user_id, n=4)
        except Exception as e:  # noqa: BLE001
            logger.warning("History load failed for %s: %s", user_id, e)
            return []
        for turn in reversed(turns):
            user_text = turn.get("user", "")
            assistant_text = turn.get("assistant", "")
            if user_text:
                out.append({"role": "user", "content": user_text})
            if assistant_text:
                out.append({"role": "assistant", "content": assistant_text})
        return out

    def _persist(self, user_id: str, turn_id: str, original_query: str, final_answer: str) -> None:
        try:
            self.redis_store.store_turn(user_id, {
                "turn_id": turn_id,
                "user": original_query,
                "assistant": final_answer,
            })
            self.redis_store.set_last_assistant_meta(user_id, {
                "assistant_text": final_answer,
                "turn_id": turn_id,
            })
        except Exception as e:  # noqa: BLE001
            logger.warning("Persist failed for %s/%s: %s", user_id, turn_id, e)

    def clear_session(self) -> None:
        logger.info("Session cleared.")
