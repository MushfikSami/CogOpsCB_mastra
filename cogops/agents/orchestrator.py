"""
cogops/agents/orchestrator.py

Per-request façade for the multi-agent chatbot pipeline.

Agents (in order):
  0. InputGuard       — code-based sanitization
  1. IntentClassifier — intent + guard rails (secondary LLM, JSON)
  2. QueryProcessor   — disambiguate → formalize → fan-out
  3. RetrievalAgent   — Jiggasha ReAct loop
  4. ComposerAgent    — streaming primary LLM
  5. PostFlightVerifier — NLI + policy + Sources block
"""

from __future__ import annotations

import logging
import os
import re
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

import yaml

from cogops.agents.composer_agent import ComposerAgent, PostFlightVerifier
from cogops.agents.input_guard import GuardConfig, InputGuard
from cogops.agents.intent_classifier import IntentClassifier, IntentResult
from cogops.agents.query_processor import QueryProcessor
from cogops.agents.retrieval_agent import RetrievalAgent, RetrievalConfig
from cogops.config.loader import _load_endpoint_config
from cogops.llm.clients import AsyncLLMService
from cogops.prompts.messages import SERVER_LOAD_FALLBACK_BN
from cogops.prompts.time_reminder import build_time_reminder
from cogops.session.redis_store import RedisSessionStore

logger = logging.getLogger(__name__)


_DEFAULT_REFUSAL_BN = (
    "দুঃখিত, এই প্রশ্নের জন্য নির্ভরযোগ্য সরকারি তথ্য পাওয়া যায়নি।"
)

_CHITCHAT_GREETING_BN = (
    "স্বাগতম! আমি বাংলাদেশ সরকারের সেবা সম্পর্কিত প্রশ্নে সাহায্য করতে পারি — "
    "যেমন এনআইডি, পাসপোর্ট, ট্যাক্স, সনদ, ইত্যাদি। আপনি কোন বিষয়ে জানতে চান?"
)

_INPUT_INVALID_REFUSAL_BN = (
    "দুঃখিত, প্রশ্নটি বোঝা গেল না বা সীমার বাইরে। "
    "অনুগ্রহ করে স্পষ্ট, সংক্ষিপ্ত প্রশ্ন করুন।"
)

# Date/time query detector — Bengali + English patterns
_DATE_QUERY_RE = re.compile(
    r"(\bdate\b|\btoday\b|\bhijri\b|\barabic\s+calendar\b|"
    r"আজকের\s+তারিখ|আজ\s+কত\s+তারিখ|আরবি\s+তারিখ|হিজরি\s+তারিখ|"
    r"আজ\s+কি\s+দিন|আজ\s+কোন\s+দিন|সময়\s+কত|বাংলাদেশ\s+সময়)",
    re.IGNORECASE | re.UNICODE,
)


def _build_date_answer(query: str) -> str:
    """Answer date/time queries using the canonical Bangladesh time reminder."""
    lower = query.lower()
    reminder = build_time_reminder()

    # Extract the date line: "Date: 4 June 2026   (৪ জুন ২০২৬)"
    date_line = None
    weekday_line = None
    time_line = None
    for line in reminder.splitlines():
        if line.startswith("- Date:"):
            date_line = " ".join(line.replace("- Date:", "").split())
        elif line.startswith("- Weekday:"):
            weekday_line = " ".join(line.replace("- Weekday:", "").split())
        elif line.startswith("- Time:"):
            time_line = " ".join(line.replace("- Time:", "").split())

    if "hijri" in lower or "arabic" in lower or "আরবি" in lower or "হিজরি" in query.lower():
        return (
            "আমি বর্তমানে হিজরি/আরবি ক্যালেন্ডারের তারিখ দিতে পারছি না। "
            f"বাংলাদেশ সময় অনুযায়ী আজ {weekday_line}। "
            f"আজকের তারিখ {date_line}।"
        )

    return (
        f"আজ {weekday_line}। "
        f"আজকের তারিখ {date_line} এবং বর্তমান সময় {time_line}।"
    )


class Orchestrator:
    """One per user_id. Holds all agents, LLM clients, and Redis store."""

    def __init__(self, config_path: str = "configs/config.yml"):
        logger.info("Initializing CogOpsCB Orchestrator…")
        self.config = self._load_config(config_path)
        self.agent_name = self.config.get("agent", {}).get("name", "আশা")

        # LLM clients
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

        # Response templates (config-overridable)
        responses = self.config.get("responses", {}) or {}
        self.refusal_text = responses.get("refusal_text_bn", _DEFAULT_REFUSAL_BN)
        self.political_refusal_text = responses.get(
            "political_refusal_text_bn",
            "আমি একটি নিরপেক্ষ সরকারি সেবা সহকারী। রাজনৈতিক তুলনা বা মতামত দেওয়া আমার পক্ষে সম্ভব নয়।",
        )
        self.personal_law_refusal_text = responses.get(
            "personal_law_refusal_text_bn",
            "এই প্রশ্নটি ব্যক্তিগত আইন বা ধর্মীয় বিধানের পরামর্শের বিষয়। সঠিক উত্তরের জন্য সংশ্লিষ্ট বিশেষজ্ঞের সাথে পরামর্শ করুন।",
        )
        self.self_harm_refusal_text = responses.get(
            "self_harm_refusal_text_bn",
            "আমি আপনাকে এই মুহূর্তে সাহায্য করতে পারব না। অনুগ্রহ করে ৯৯৯ বা নিকটতম হাসপাতালে যোগাযোগ করুন।",
        )
        self.illegal_refusal_text = responses.get(
            "illegal_refusal_text_bn",
            "এই ধরনের কার্যকলাপ সম্পর্কে তথ্য দেওয়া আমার পক্ষে সম্ভব নয়।",
        )
        self.system_probe_refusal_text = responses.get(
            "system_probe_response_bn",
            "আমি বাংলাদেশ সরকারের ডিজিটাল সহকারী 'আশা'। আমার কাজ নাগরিকদের সরকারি সেবা সংক্রান্ত তথ্য সহজভাবে বাংলায় পৌঁছে দেওয়া।",
        )
        self.chitchat_greeting = responses.get("chitchat_greeting_bn", _CHITCHAT_GREETING_BN)
        self.input_invalid_refusal = responses.get("input_invalid_refusal_bn", _INPUT_INVALID_REFUSAL_BN)

        # Agent 0: InputGuard
        ig_cfg = self.config.get("input_guard", {}) or {}
        self.input_guard = InputGuard(GuardConfig(
            max_chars=ig_cfg.get("max_chars", 4096),
            entropy_threshold=ig_cfg.get("entropy_threshold", 1.5),
        ))

        # Agent 1: IntentClassifier
        ic_cfg = self.config.get("intent_classifier", {}) or {}
        self.intent_classifier = IntentClassifier(
            secondary_client=self.secondary_service.client_llm,
            secondary_model=self.secondary_service.model,
            timeout=ic_cfg.get("timeout_seconds", 5.0),
            max_sub_queries=ic_cfg.get("max_concurrent_query", 3),
        )

        # Agent 2: QueryProcessor
        self.query_processor = QueryProcessor(
            secondary_client=self.secondary_service.client_llm,
            secondary_model=self.secondary_service.model,
            max_concurrent_query=ic_cfg.get("max_concurrent_query", 3),
        )

        # Agent 3: RetrievalAgent
        self.retrieval_agent = RetrievalAgent(
            cfg=self._build_retrieval_config(),
            secondary_client=self.secondary_service.client_llm,
            secondary_model=self.secondary_service.model,
        )

        # Agent 4: ComposerAgent
        pipe_cfg = self.config.get("pipeline", {}) or {}
        comp_cfg = pipe_cfg.get("composer", {})
        self.composer = ComposerAgent(
            primary_client=self.llm_service.client_llm,
            primary_model=self.llm_service.model,
            agent_name=self.agent_name,
            temperature=float(comp_cfg.get("temperature", 0.1)),
            top_p=float(comp_cfg.get("top_p", 0.95)),
            max_tokens=int(comp_cfg.get("max_tokens", 2048)),
        )

        # Agent 5: PostFlightVerifier
        ver_cfg = self.config.get("verifier", {}) or {}
        self.verifier = PostFlightVerifier(
            secondary_client=self.secondary_service.client_llm,
            secondary_model=self.secondary_service.model,
            enabled=bool(ver_cfg.get("enabled", True)),
            timeout=float(ver_cfg.get("timeout_seconds", 6.0)),
            policy=str(ver_cfg.get("policy", "redact")),
            refusal_text_bn=self.refusal_text,
        )

        logger.info(
            "Orchestrator ready. agents=[guard,classifier,processor,retrieval,composer,verifier] "
            "primary=%s secondary=%s",
            self.llm_service.model, self.secondary_service.model,
        )

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------
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

    def _build_retrieval_config(self) -> RetrievalConfig:
        tools_cfg = self.config.get("tools", {}) or {}
        jcfg = tools_cfg.get("jiggasha", {})
        pcfg = self.config.get("pipeline", {}) or {}
        ret_cfg = pcfg.get("retrieval", {}) or {}

        endpoint = (
            os.getenv(jcfg.get("endpoint_env", "JIGGASHA_ENDPOINT"))
            or jcfg.get("endpoint")
            or "http://localhost:10000/search"
        )

        return RetrievalConfig(
            endpoint=endpoint,
            timeout=float(jcfg.get("timeout_seconds", 45.0)),
            top_k_fetch=int(jcfg.get("top_k_fetch", jcfg.get("top_k", 50))),
            use_instruction=bool(jcfg.get("use_instruction", True)),
            cosine_threshold=jcfg.get("cosine_threshold"),
            token_budget=jcfg.get("token_budget"),
            rerank_threshold=jcfg.get("rerank_threshold", 0.50),
            max_react_iterations=int(ret_cfg.get("max_react_iterations", 0)),
            merge_global_cap=int(ret_cfg.get("merge_global_cap", 50)),
        )

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def _load_history(self, user_id: Optional[str]) -> List[Dict[str, str]]:
        if not user_id or not self.redis_store.available:
            return []
        out: List[Dict[str, str]] = []
        try:
            turns = self.redis_store.get_recent_turns(user_id, n=4)
        except Exception:
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
        except Exception as e:
            logger.warning("Persist failed for %s/%s: %s", user_id, turn_id, e)

    # ------------------------------------------------------------------
    # Guard-rail response selector
    # ------------------------------------------------------------------
    def _guard_rail_response(self, result: IntentResult) -> str:
        category = result.guard_rail_category
        if category == "self_harm":
            return self.self_harm_refusal_text
        if category == "illegal":
            return self.illegal_refusal_text
        if category == "political_comparison":
            return self.political_refusal_text
        if category == "personal_attack":
            return self.personal_law_refusal_text
        if category == "system_probe":
            return self.system_probe_refusal_text
        return self.refusal_text

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
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
            # ----- Layer 0: InputGuard -----
            clean_query, refusal_reason = self.input_guard.check(original_query)
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
                   "turns": len(history_messages), "turn_id": turn_id}

            # ----- Layer 1: IntentClassifier -----
            try:
                intent_result = await self.intent_classifier.classify(
                    query=clean_query,
                    history=history_messages,
                )
            except Exception as e:
                logger.warning("IntentClassifier raised (%s); defaulting to factual.", e)
                intent_result = IntentResult(
                    intent="factual",
                    sub_queries=[clean_query],
                    confidence=0.5,
                    notes=[f"classifier_exception: {e!s}"],
                )

            yield {"type": "intent_classified", "channel": "debug",
                   "intent": intent_result.intent,
                   "guard_rail_triggered": intent_result.guard_rail_triggered,
                   "guard_rail_category": intent_result.guard_rail_category,
                   "sub_queries": intent_result.sub_queries,
                   "needs_clarification": intent_result.needs_clarification,
                   "confidence": intent_result.confidence,
                   "notes": intent_result.notes,
                   "token_usage": intent_result.usage,
                   "turn_id": turn_id}

            # ----- Guard-rail branch -----
            if intent_result.should_refuse():
                response_text = self._guard_rail_response(intent_result)
                yield {"type": "answer_chunk", "channel": "both", "content": response_text}
                yield {"type": "final_answer", "channel": "both",
                       "content": response_text,
                       "turn_id": turn_id, "source_map": {},
                       "reason": f"guard_rail_{intent_result.guard_rail_category}"}
                yield {"type": "answer_complete", "channel": "both", "turn_id": turn_id}
                if user_id:
                    self._persist(user_id, turn_id, original_query, response_text)
                return

            # ----- Chitchat branch -----
            if intent_result.intent == "chitchat":
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

            # ----- Ambiguous branch -----
            if intent_result.intent == "ambiguous" and intent_result.needs_clarification:
                clarification = (
                    intent_result.clarification_prompt_bn
                    or "আপনার প্রশ্নটি একাধিক বিষয় জড়িত বোধ হচ্ছে। অনুগ্রহ করে আরও স্পষ্ট করে জানান।"
                )
                yield {"type": "answer_chunk", "channel": "both", "content": clarification}
                yield {"type": "final_answer", "channel": "both",
                       "content": clarification,
                       "turn_id": turn_id, "source_map": {},
                       "reason": "ambiguous_clarification"}
                yield {"type": "answer_complete", "channel": "both", "turn_id": turn_id}
                if user_id:
                    self._persist(user_id, turn_id, original_query, clarification)
                return

            # ----- Date / Time branch -----
            if _DATE_QUERY_RE.search(clean_query):
                date_answer = _build_date_answer(clean_query)
                yield {"type": "answer_chunk", "channel": "both", "content": date_answer}
                yield {"type": "final_answer", "channel": "both",
                       "content": date_answer,
                       "turn_id": turn_id, "source_map": {},
                       "reason": "date_time_direct"}
                yield {"type": "answer_complete", "channel": "both", "turn_id": turn_id}
                if user_id:
                    self._persist(user_id, turn_id, original_query, date_answer)
                return

            # ----- Factual / Multi-question branch -----
            if not intent_result.is_factual():
                # Fallback for unexpected intents
                logger.warning("Unexpected intent %s; treating as factual.", intent_result.intent)
                intent_result.intent = "factual"
                if not intent_result.sub_queries:
                    intent_result.sub_queries = [clean_query]

            # Layer 2: QueryProcessor
            processed = await self.query_processor.process(
                raw_query=clean_query,
                sub_queries=intent_result.sub_queries,
                history=history_messages,
            )
            yield {"type": "queries_processed", "channel": "debug",
                   "queries": processed.queries,
                   "overflow": processed.overflow,
                   "disambiguated": processed.disambiguated,
                   "formalized": processed.formalized,
                   "turn_id": turn_id}

            # Layer 3: RetrievalAgent
            retrieval_result = await self.retrieval_agent.retrieve(processed.queries)
            yield {"type": "retrieval_done", "channel": "debug",
                   "passages_returned": len(retrieval_result.passages),
                   "instructions": retrieval_result.instructions,
                   "elapsed_ms": retrieval_result.elapsed_ms,
                   "errors": retrieval_result.errors,
                   "turn_id": turn_id}

            yield {"type": "source_map_allocated", "channel": "debug",
                   "n_sources": len(retrieval_result.source_map),
                   "tags": list(retrieval_result.source_map.keys()),
                   "turn_id": turn_id}

            # Layer 4: ComposerAgent (streaming)
            final_text_for_persist: Optional[str] = None
            try:
                raw_answer = ""
                async for event in self.composer.compose(
                    raw_query=clean_query,
                    source_map=retrieval_result.source_map,
                    history=history_messages,
                ):
                    if event.get("type") == "composer_done":
                        raw_answer = event.get("raw_answer", "")
                    yield event
            except RuntimeError as e:
                if str(e) == "composer_empty":
                    yield {"type": "answer_chunk", "channel": "both",
                           "content": self.refusal_text}
                    yield {"type": "final_answer", "channel": "both",
                           "content": self.refusal_text,
                           "turn_id": turn_id,
                           "source_map": {},
                           "reason": "composer_empty"}
                    yield {"type": "answer_complete", "channel": "both", "turn_id": turn_id}
                    return
                raise

            # Layer 5: PostFlightVerifier
            final_answer, post_events, sources_block = await self.verifier.verify(
                raw_answer=raw_answer,
                source_map=retrieval_result.source_map,
            )
            for ev in post_events:
                yield ev

            if sources_block:
                yield {"type": "answer_chunk", "channel": "both",
                       "content": "\n\n" + sources_block}

            yield {"type": "final_answer", "channel": "both",
                   "content": final_answer,
                   "source_map": {
                       tag: {k: v for k, v in meta.items() if k != "text"}
                       for tag, meta in retrieval_result.source_map.items()
                   },
                   "turn_id": turn_id}
            yield {"type": "answer_complete", "channel": "both", "turn_id": turn_id}

            final_text_for_persist = final_answer
            if user_id and final_text_for_persist:
                self._persist(user_id, turn_id, original_query, final_text_for_persist)

        except Exception as e:
            logger.critical("[turn:%s] Unhandled in Orchestrator: %s", turn_id, e, exc_info=True)
            yield {"type": "answer_chunk", "content": SERVER_LOAD_FALLBACK_BN, "channel": "both"}
            yield {"type": "answer_complete", "channel": "both", "turn_id": turn_id}

    def clear_session(self) -> None:
        logger.info("Session cleared.")
