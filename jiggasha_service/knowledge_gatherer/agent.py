"""TAO-ReAct loop orchestrator.

State machine:
  THOUGHT -> ACTION(search) -> OBSERVATION -> THOUGHT -> ... (repeat)
  LLM decides when to stop: {"action": "answer", "answer": "..."}
  At max_steps or when LLM decides "answer", rerank all collected passages.
"""

from __future__ import annotations

import json
import logging

from openai import AsyncOpenAI

from .config import Config, get_config
from .embedder import EmbedderClient
from .retriever import Retriever
from .reranker import RerankerClient

logger = logging.getLogger(__name__)

THOUGHT_SYSTEM_PROMPT = (
    "You are a knowledge gatherer agent. You answer questions by iteratively\n"
    "searching for information. At each step produce:\n\n"
    "1. THOUGHT: Analyze what you know, what you still need, and plan your next move.\n"
    "   - If you have enough information to answer, state that clearly.\n"
    "   - If you need more info, specify exactly what you are looking for.\n\n"
    "2. ACTION: A JSON object with one of:\n"
    '   {"action": "search", "query": "your search query here"} to retrieve more info\n'
    '   {"action": "answer", "answer": "your complete answer here"} when done\n\n'
    "Rules:\n"
    "- Think step by step. Be specific about what you know vs what you need.\n"
    "- When searching, frame your query to target specific information gaps.\n"
    "- If you have gathered enough evidence, use action: \"answer\" with your final answer.\n"
    "- Always answer in the same language as the user's query (Bengali if query is Bengali).\n"
    "- Never hallucinate. Only use information you have retrieved.\n"
    '- If no confirmed information exists after gathering, say so explicitly.\n'
    "You MUST include the ACTION JSON in your response."
)

ANSWER_SYSTEM_PROMPT = (
    "You are a knowledge gatherer synthesizing an answer from retrieved evidence.\n\n"
    "Instructions:\n"
    "- Answer only based on the retrieved passages provided.\n"
    "- Be concise and factual.\n"
    "- Answer in the same language as the user's query.\n"
    '- If no retrieved passage supports the answer, say "NO CONFIRMED INFORMATION exists for this query."\n'
    "- Cite which passages support your claims.\n"
)


class TAOAgent:
    def __init__(
        self,
        config: Config | None = None,
        embedder: EmbedderClient | None = None,
        retriever: Retriever | None = None,
        reranker: RerankerClient | None = None,
    ) -> None:
        self._config = config or get_config()
        self._embedder = embedder or EmbedderClient(self._config)
        self._retriever = retriever or Retriever(self._config, self._embedder)
        self._reranker = reranker or RerankerClient(self._config)

        self._llm = AsyncOpenAI(
            api_key=self._config.openai_api_key,
            base_url=self._config.openai_base_url,
        )
        self._llm_model = self._config.llm_model

    async def run(
        self,
        query: str,
        mode: str = "web",
        top_k: int | None = None,
        max_steps: int | None = None,
    ) -> dict:
        max_steps = max_steps or self._config.max_tao_steps
        top_k = top_k or self._config.top_k

        messages = [
            {"role": "system", "content": THOUGHT_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]

        all_retrieved: list[dict] = []
        tao_steps_log: list[dict] = []

        for step_num in range(1, max_steps + 1):
            step_log: dict = {"step": step_num, "phase": "thought"}

            # === THOUGHT ===
            response = await self._llm.chat.completions.create(
                model=self._llm_model,
                messages=messages,
                max_tokens=500,
                temperature=0.0,
            )
            thought = response.choices[0].message.content or ""
            step_log["thought"] = thought.strip()
            messages.append({"role": "assistant", "content": thought})
            logger.info("[TAO step %d] THOUGHT:\n%s", step_num, step_log["thought"])

            # === PARSE ACTION ===
            action, action_val = self._parse_action(thought)

            if action == "answer":
                step_log["phase"] = "answer_ready"
                step_log["answer"] = action_val
                step_log["reasoning_summary"] = thought.strip()
                logger.info("[TAO step %d] LLM decided: answer", step_num)
                return await self._synthesize_and_return(
                    query, thought.strip(), all_retrieved, step_log, mode
                )

            if step_num == max_steps:
                step_log["phase"] = "max_steps_reached"
                step_log["reasoning_summary"] = (
                    f"Reached maximum of {max_steps} steps. "
                    "Synthesizing answer from gathered information."
                )
                logger.info("[TAO step %d] Max steps reached", step_num)
                return await self._synthesize_and_return(
                    query, step_log["reasoning_summary"], all_retrieved, step_log, mode
                )

            # === ACTION: retrieve ===
            step_log["phase"] = "action"
            step_log["action"] = {"type": "search", "query": action_val}

            logger.info("[TAO step %d] ACTION: search '%s'", step_num, action_val)

            passages = await self._retriever.retrieve(action_val, mode, top_k)
            step_log["observations"] = {
                "count": len(passages),
                "top_scores": [p.get("score", 0) for p in passages[:3]],
            }

            # Collect and deduplicate
            all_retrieved.extend(passages)
            seen: set = set()
            unique: list[dict] = []
            for p in all_retrieved:
                pid = str(p["id"])
                if pid not in seen:
                    seen.add(pid)
                    unique.append(p)
            all_retrieved = unique

            # === OBSERVATION ===
            obs_summary = self._summarize_observations(passages)
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"[OBSERVATION]\n"
                        f"Retrieved {len(passages)} passages for query '{action_val}':\n"
                        f"{obs_summary}"
                    ),
                }
            )
            step_log["phase"] = "observation"
            step_log["observation_summary"] = obs_summary
            tao_steps_log.append(step_log)

        # Safety fallback
        return await self._synthesize_and_return(
            query, "Loop completed.", all_retrieved,
            {"step": max_steps, "phase": "loop_complete"}, mode,
        )

    @staticmethod
    def _parse_action(thought: str) -> tuple[str, str]:
        """Extract action JSON from LLM thought response."""
        thought = thought.strip()
        start = thought.find("{")
        end = thought.rfind("}") + 1
        if start == -1 or end == 0:
            return ("search", thought)

        try:
            data = json.loads(thought[start:end])
            action = data.get("action", "search")
            val = data.get("query", data.get("answer", thought))
            return (action, val)
        except json.JSONDecodeError:
            return ("search", thought)

    @staticmethod
    def _summarize_observations(passages: list[dict]) -> str:
        lines = []
        for p in passages[:5]:
            text = (p.get("text", "") or "")[:300]
            node = (p.get("web_node", "") or "[no node]")
            lines.append(f"- [{node}] {text}{'...' if len(text) == 300 else ''}")
        return "\n".join(lines) if lines else "No passages retrieved."

    async def _synthesize_and_return(
        self,
        query: str,
        reasoning: str,
        all_passages: list[dict],
        step_log: dict,
        mode: str,
    ) -> dict:
        if not all_passages:
            return {
                "answer": (
                    "NO CONFIRMED INFORMATION exists for this query in the database."
                ),
                "reasoning": reasoning,
                "tao_steps": [step_log],
                "sources": [],
                "rerank_results": [],
                "relevant_passages": [],
                "status": "no_data",
                "mode": mode,
            }

        # === RERANK ===
        logger.info("Reranking %d collected passages...", len(all_passages))
        rerank_results = await self._reranker.rank(query, all_passages)

        threshold = self._config.rerank_threshold
        relevant_passages = [
            (p, score, reason)
            for p, score, reason in rerank_results
            if score >= threshold
        ]

        if not relevant_passages:
            return {
                "answer": (
                    "NO CONFIRMED INFORMATION exists for this query in the database."
                ),
                "reasoning": reasoning,
                "tao_steps": [step_log],
                "sources": [],
                "rerank_results": rerank_results,
                "relevant_passages": [],
                "status": "no_data",
                "mode": mode,
            }

        # === ANSWER SYNTHESIS ===
        evidence_parts = []
        for i, (passage, score, _) in enumerate(relevant_passages[:10]):
            text = (passage.get("text", "") or "")[:800]
            node = passage.get("web_node", f"source_{i}")
            evidence_parts.append(
                f"[Source {i + 1} ({node}, score={score:.3f})]\n{text}"
            )

        evidence_text = "\n---\n".join(evidence_parts)

        answer_messages = [
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Query: {query}\n\nEvidence:\n{evidence_text}\n\n"
                    "Provide a clear, concise answer based on the evidence above."
                ),
            },
        ]

        answer_resp = await self._llm.chat.completions.create(
            model=self._llm_model,
            messages=answer_messages,
            max_tokens=2000,
            temperature=0.0,
        )
        answer = answer_resp.choices[0].message.content or ""

        # Build sources list
        sources = []
        for passage, score, _ in relevant_passages[:10]:
            sources.append(
                {
                    "id": str(passage.get("id", "")),
                    "text": passage.get("text", ""),
                    "web_node": passage.get("web_node", ""),
                    "url": passage.get("url", ""),
                    "relevance_score": score,
                    "source_collection": passage.get("source_collection", mode),
                }
            )

        return {
            "answer": answer,
            "reasoning": reasoning,
            "tao_steps": [step_log],
            "sources": sources,
            "rerank_results": rerank_results,
            "relevant_passages": relevant_passages,
            "status": "complete",
            "mode": mode,
        }
