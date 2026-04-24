"""
cogops/llm/reranker.py

QwenRerankerClient: cross-encoder reranker for binary passage relevance.
Moved from cogops/models/reranker.py.
"""

import logging
from openai import AsyncOpenAI
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.llm_client.config import LLMConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Qwen token IDs: 16 = "1" (answer present), 15 = "0" (absent)
ANSWER_PRESENT_TOKEN = 16
ANSWER_ABSENT_TOKEN = 15


class QwenRerankerClient(OpenAIRerankerClient):
    """Binary-classification reranker using Qwen with logit-bias forcing."""

    def __init__(self, client: AsyncOpenAI, config: LLMConfig):
        super().__init__(client=client, config=config)
        self._logit_bias = {ANSWER_PRESENT_TOKEN: 100, ANSWER_ABSENT_TOKEN: 100}

    async def rank(self, query: str, passages: list[str]) -> list[tuple]:
        """Rank passages by relevance to the query using logit-bias forcing."""
        ranked = []
        for passage in passages:
            try:
                response = await self.client.chat.completions.create(
                    model=self.config.model,
                    messages=[{"role": "user", "content": f"Does this passage contain the answer to '{query}'? Answer 0 or 1."}],
                    extra_body={"logit_bias": self._logit_bias, "max_tokens": 1},
                )
                content = response.choices[0].message.content or ""
                logprobs = response.choices[0].logprobs.content[0].logprobs if response.choices[0].logprobs else {}
                # Convert logprobs dict to probability
                if "1" in content:
                    prob = logprobs.get(16, 0)
                    prob = min(max(prob, 0), 1)
                else:
                    prob = 0.0
                ranked.append((passage, prob))
            except Exception as e:
                logger.error(f"Reranker error for passage: {e}")
                ranked.append((passage, 0.0))
        # Sort by probability descending
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked
