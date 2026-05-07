"""Binary relevance reranker using LLM logprobs.

Scores each passage 0/1 via the LLM, then normalizes with softmax.
Batches with asyncio.gather() for high performance.
"""

from __future__ import annotations

import asyncio
import logging
import numpy as np

from openai import AsyncOpenAI

from .config import Config, get_config

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an expert relevance judge for a government service search system.\n"
    "Determine if a passage contains information relevant to answering the user's query.\n"
    "Reply with a single digit: 0 for no, 1 for yes."
)

USER_PROMPT = (
    "Passage: {text}\n\n"
    "Query: {query}\n\n"
    "Does this passage contain information relevant to answering the query?\n"
    "Reply with a single digit: 0 for no, 1 for yes."
)


class RerankerClient:
    def __init__(self, config: Config | None = None) -> None:
        self._config = config or get_config()
        self._client = AsyncOpenAI(
            api_key=self._config.openai_api_key,
            base_url=self._config.openai_base_url,
        )
        self._model = self._config.llm_model

    async def _score_passage(self, passage: dict, query: str) -> tuple:
        text = passage.get("text", "").strip()
        if not text:
            return (passage, 0.0, "Empty text")

        try:
            resp = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT.format(
                        text=text[:1500],
                        query=query,
                    )},
                ],
                max_tokens=self._config.rerank_max_tokens,
                logprobs=True,
                top_logprobs=20,
                temperature=0.0,
            )

            content = resp.choices[0].message.content.strip()
            reason = f"LLM output: {content}"
            logprobs_data = resp.choices[0].logprobs

            score = 0.0
            if logprobs_data and logprobs_data.content:
                top_lp = logprobs_data.content[0].top_logprobs
                lp0, lp1 = -100.0, -100.0
                for tl in top_lp:
                    tok = tl.token.strip()
                    if tok == "0":
                        lp0 = tl.logprob
                    elif tok == "1":
                        lp1 = tl.logprob

                max_lp = max(lp0, lp1)
                score = float(
                    np.exp(lp1 - max_lp)
                    / (np.exp(lp0 - max_lp) + np.exp(lp1 - max_lp))
                )
                score = float(max(0.0, min(1.0, score)))

            elif content == "1":
                score = 0.9
            else:
                score = 0.1

            return (passage, round(score, 4), reason)

        except Exception as e:
            logger.error("Reranker error for passage %s: %s", passage.get("id", "?"), e)
            return (passage, 0.0, f"Error: {e}")

    async def rank(self, query: str, passages: list[dict]) -> list[tuple]:
        if not passages:
            return []

        batch_size = self._config.max_batch_size
        results: list[tuple] = []

        for i in range(0, len(passages), batch_size):
            batch = passages[i : i + batch_size]
            tasks = [self._score_passage(p, query) for p in batch]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)

        results.sort(key=lambda x: x[1], reverse=True)
        return results
