"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import logging
from typing import Any

import numpy as np
import openai

# Import necessary helpers and base classes from graphiti_core
from graphiti_core.helpers import semaphore_gather
from graphiti_core.llm_client import LLMConfig, RateLimitError
from graphiti_core.prompts import Message
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient

logger = logging.getLogger(__name__)

# --- QWEN3.5 TOKEN CONFIGURATION ---
# These IDs correspond to the Qwen3.5-35B tokenizer.
# Using binary 1/0 tokens instead of True/False because:
# - True/False gets interpreted as factual correctness rather than "contains answer"
# - Binary 1/0 forces the model to treat this as a score classification task
# Token '1': 16  (answer present)
# Token '0': 15  (answer not present)
QWEN_ANSWER_PRESENT_ID = 16  # "1"
QWEN_ANSWER_ABSENT_ID = 15   # "0"

# Combine them for the bias dictionary.
# We use a bias of 100 to FORCE the model to pick only these tokens.
QWEN_LOGIT_BIAS = {QWEN_ANSWER_PRESENT_ID: 100, QWEN_ANSWER_ABSENT_ID: 100}

DEFAULT_MODEL = 'qwen3'  # Placeholder, usually overridden by config


class QwenRerankerClient(OpenAIRerankerClient):
    """
    A Reranker client specifically adapted for Qwen3.5 / vLLM.

    It inherits from OpenAIRerankerClient but overrides the `rank` method
    to use Qwen-specific Token IDs for logit_bias.
    """

    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        # 1. Prepare the messages.
        # Using binary 1/0 tokens instead of True/False to avoid the model interpreting
        # the output as factual correctness. Binary scores treat this as a classification
        # task: 1 = passage contains the answer, 0 = passage does not contain the answer.
        openai_messages_list: Any = [
            [
                Message(
                    role='system',
                    content='You are an information retrieval assistant. For each passage, determine if it contains the answer to the query. Reply "1" if the passage contains the answer. Reply "0" if the passage does not contain the answer.',
                ),
                Message(
                    role='user',
                    content=f"""
                        Query: {query}
                        Passage: {passage}

                        Does this passage contain the answer?
                        Reply ONLY with "1" (yes) or "0" (no).
                        """,
                ),
            ]
            for passage in passages
        ]

        try:
            # 2. Send requests concurrently using semaphore_gather.
            responses = await semaphore_gather(
                *[
                    self.client.chat.completions.create(
                        model=self.config.model or DEFAULT_MODEL,
                        messages=openai_messages,
                        temperature=0,
                        max_tokens=1,

                        # --- CHANGE FROM ORIGINAL ---
                        # Original used OpenAI IDs {'6432': 1, '7983': 1}
                        # We use Qwen binary 1/0 token IDs with high bias.
                        # Token 16 = "1" (answer present), Token 15 = "0" (answer absent)
                        logit_bias=QWEN_LOGIT_BIAS,

                        logprobs=True,
                        # We request 2 logprobs. Since we biased only "1" and "0"
                        # extremely heavily, these two should always be the top 2.
                        top_logprobs=2,
                    )
                    for openai_messages in openai_messages_list
                ]
            )

            # 3. Extract top logprobs.
            responses_top_logprobs = [
                response.choices[0].logprobs.content[0].top_logprobs
                if response.choices[0].logprobs is not None
                and response.choices[0].logprobs.content is not None
                else []
                for response in responses
            ]

            scores: list[float] = []

            # 4. Calculate Scores.
            # For binary classification: 1 = high score (answer present), 0 = low score (answer absent)
            for top_logprobs in responses_top_logprobs:
                if len(top_logprobs) == 0:
                    scores.append(0.0) # Safety fallback
                    continue

                # Get the token text and convert logprob to probability
                token_text = top_logprobs[0].token.strip()
                logprob = top_logprobs[0].logprob
                prob = 100 * (2.718281828 ** logprob)  # Convert logprob to percentage

                if token_text == '1':
                    # Model said "1" (answer present) - use the probability directly
                    scores.append(prob / 100.0)
                else:
                    # Model said "0" (answer absent) - score is 1 - probability
                    # If it said "0" with 99% confidence, score = 1 - 0.99 = 0.01 (low relevance)
                    scores.append(1 - (prob / 100.0))

            # 5. Sort and Return Results.
            results = [(passage, score) for passage, score in zip(passages, scores, strict=True)]
            results.sort(reverse=True, key=lambda x: x[1])
            return results

        except openai.RateLimitError as e:
            raise RateLimitError from e
        except Exception as e:
            logger.error(f'Error in generating Qwen Reranker response: {e}')
            # Fallback: return 0 scores rather than crashing the application
            return [(p, 0.0) for p in passages]