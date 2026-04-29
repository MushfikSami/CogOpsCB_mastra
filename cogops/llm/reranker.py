"""
cogops/llm/reranker.py

Cross-encoder reranker: binary relevance classification via LLM logprobs.
Batched for high performance using asyncio.gather(). 
Determines if a document's summary indicates that the FULL document 
will contain the answer to the user's query.
"""

import logging
import asyncio
import numpy as np
from openai import AsyncOpenAI

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an expert relevance judge for a government service search system.
Your job is to act as a routing filter: determine if a document's summary indicates that the FULL document will contain the answer to the user's query.

IMPORTANT GUIDELINES:
- You are evaluating an "Index-Style Summary". It may not contain exact granular details, but outlines what the full document covers.
- If the user asks for specific steps, rules, or requirements, and the summary indicates that the full document provides these, consider it relevant.
- Be lenient: if the summary strongly suggests the underlying text holds the answer, return 1.
- The service category must match. If the document is irrelevant or about a completely different topic, return 0.
- Reply with a single digit: 0 for no, 1 for yes.\
"""

USER_PROMPT = """\
Document Title: {node}
Document Summary: {text}

Query: {query}

Based on this summary, would reading the full document likely answer the user's query? 
Reply with a single digit: 0 for no, 1 for yes.\
"""

class RerankerClient:
    """Binary-classification reranker using an OpenAI-compatible LLM.

    Evaluates passages concurrently. Captures top_logprobs for
    tokens "0" and "1", then computes softmax to get a 0-1 relevance score.
    """

    def __init__(self, client: AsyncOpenAI, model: str):
        self.client = client
        self.model = model

    async def _score_passage(self, passage: dict, query: str) -> tuple:
        """Helper to score a single passage asynchronously."""
        # Target the generated 'summary' field first, fallback to 'text'
        text = passage.get("summary", passage.get("text", "")).strip()
        node = passage.get("node", "Unknown")
        
        if not text:
            return (passage, 0.0, "Empty text/summary")

        try:
            # We request only 1 token (0 or 1) to maximize vLLM speed
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT.format(node=node, text=text, query=query)},
                ],
                max_tokens=1, 
                logprobs=True,
                top_logprobs=20,
            )

            content = resp.choices[0].message.content.strip()
            reason = f"LLM output: {content}"
            logprobs_data = resp.choices[0].logprobs

            score = 0.0
            if logprobs_data and logprobs_data.content:
                top_lp = logprobs_data.content[0].top_logprobs

                # Initialize logs to a very low value
                lp0, lp1 = -100.0, -100.0
                for tl in top_lp:
                    token = tl.token.strip()
                    if token == "0": lp0 = tl.logprob
                    elif token == "1": lp1 = tl.logprob

                # Stable softmax calculation
                max_lp = max(lp0, lp1)
                score = np.exp(lp1 - max_lp) / (np.exp(lp0 - max_lp) + np.exp(lp1 - max_lp))
                score = float(max(0.0, min(1.0, score)))

            # Fallback if logprobs aren't available
            elif content == "1":
                score = 0.9
            else:
                score = 0.1

            return (passage, round(score, 4), reason)

        except Exception as e:
            logger.error(f"Reranker error for node '{node}': {e}")
            return (passage, 0.0, f"Error: {e}")

    async def rank(self, query: str, passages: list[dict]) -> list[tuple]:
        """Rank passages by relevance to the query concurrently. 
        
        Returns a sorted list of (passage_dict, score, reason).
        """
        if not passages:
            return []

        # Run all scoring tasks in parallel using vLLM continuous batching
        tasks = [self._score_passage(passage, query) for passage in passages]
        results = await asyncio.gather(*tasks)

        # Filter out completely irrelevant ones if desired, or just return sorted
        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results