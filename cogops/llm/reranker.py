"""
cogops/llm/reranker.py

QwenRerankerClient: cross-encoder reranker for binary passage relevance.
"""

import logging
import numpy as np
from openai import AsyncOpenAI
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.llm_client.config import LLMConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class QwenRerankerClient(OpenAIRerankerClient):
    """Binary-classification reranker using Qwen.

    Sends each passage to the LLM asking 0/1. Captures top_logprobs for
    tokens "0" and "1", then computes softmax: P(1) = exp(lp1)/(exp(lp0)+exp(lp1)).
    This gives a proper 0-1 relevance score.
    """

    def __init__(self, client: AsyncOpenAI, config: LLMConfig):
        super().__init__(client=client, config=config)

    async def rank(self, query: str, passages: list[str]) -> list[tuple]:
        """Rank passages by relevance to the query."""
        ranked = []
        for passage in passages:
            try:
                resp = await self.client.chat.completions.create(
                    model=self.config.model,
                    messages=[
                        {"role": "user", "content": (
                            f"Passage: {passage}\n"
                            f"Query: {query}\n"
                            f"Does this passage contain relevant information for the query? "
                            f"Reply with a single digit: 0 for no, 1 for yes."
                        )},
                    ],
                    extra_body={"max_tokens": 1},
                    logprobs=True,
                    top_logprobs=20,
                )

                content = resp.choices[0].message.content.strip()
                logprobs_data = resp.choices[0].logprobs

                if logprobs_data and logprobs_data.content:
                    top_lp = logprobs_data.content[0].top_logprobs

                    # Extract logprobs for just "0" and "1" tokens
                    lp0 = -100.0
                    lp1 = -100.0
                    for tl in top_lp:
                        token = tl.token.strip()
                        if token == "0":
                            lp0 = tl.logprob
                        elif token == "1":
                            lp1 = tl.logprob

                    # Softmax over {0, 1} space → proper probability
                    score = np.exp(lp1) / (np.exp(lp0) + np.exp(lp1))
                    score = float(max(0.0, min(1.0, score)))

                elif content == "1":
                    score = 0.5  # fallback: model output "1" but no logprobs
                else:
                    score = 0.0

                ranked.append((passage, round(score, 4)))

            except Exception as e:
                logger.error(f"Reranker error for passage: {e}")
                ranked.append((passage, 0.0))

        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked
