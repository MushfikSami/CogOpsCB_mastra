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
# We include both the word itself and the word with a leading space
# to handle potential whitespace variations in generation.
# True:  2434 ("True"), 2912 (" True")
# False: 3913 ("False"), 3439 (" False")
QWEN_TRUE_IDS = [2434, 2912]
QWEN_FALSE_IDS = [3913, 3439]

# Combine them for the bias dictionary.
# We use a bias of 100 to FORCE the model to pick only these tokens.
QWEN_LOGIT_BIAS = {tid: 100 for tid in QWEN_TRUE_IDS + QWEN_FALSE_IDS}

DEFAULT_MODEL = 'qwen3'  # Placeholder, usually overridden by config


class QwenRerankerClient(OpenAIRerankerClient):
    """
    A Reranker client specifically adapted for Qwen3.5 / vLLM.

    It inherits from OpenAIRerankerClient but overrides the `rank` method
    to use Qwen-specific Token IDs for logit_bias.
    """

    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        # 1. Prepare the messages. 
        # This block is identical to the original OpenAIRerankerClient.
        openai_messages_list: Any = [
            [
                Message(
                    role='system',
                    content='You are an expert tasked with determining whether the passage contains the answer to the query',
                ),
                Message(
                    role='user',
                    content=f"""
                           Respond with "True" if PASSAGE contains the answer to QUERY and "False" otherwise.
                           <PASSAGE>
                           {passage}
                           </PASSAGE>
                           <QUERY>
                           {query}
                           </QUERY>
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
                        # We use Qwen IDs with high bias to force strict True/False output.
                        logit_bias=QWEN_LOGIT_BIAS,
                        
                        logprobs=True,
                        # We request 2 logprobs. Since we biased only "True" and "False"
                        # extremely heavily, these two should always be the top 2.
                        top_logprobs=2,
                    )
                    for openai_messages in openai_messages_list
                ]
            )

            # 3. Extract top logprobs.
            # Identical structure to original.
            responses_top_logprobs = [
                response.choices[0].logprobs.content[0].top_logprobs
                if response.choices[0].logprobs is not None
                and response.choices[0].logprobs.content is not None
                else []
                for response in responses
            ]

            scores: list[float] = []
            
            # 4. Calculate Scores.
            # Keeps original logic: check if the top token is "True".
            for top_logprobs in responses_top_logprobs:
                if len(top_logprobs) == 0:
                    scores.append(0.0) # Safety fallback
                    continue
                
                # Convert log-probability to linear probability (0.0 to 1.0)
                norm_logprobs = np.exp(top_logprobs[0].logprob)
                
                # Check the text content. 
                # .strip() handles " True" vs "True".
                # .lower() handles "True" vs "true".
                token_text = top_logprobs[0].token.strip().split(' ')[0].lower()
                
                if token_text == 'true':
                    # If model said True, score is the probability of True
                    scores.append(norm_logprobs)
                else:
                    # If model said False (or anything else), score is (1 - probability)
                    # This works because if it said "False" with 99% confidence,
                    # the score becomes 1 - 0.99 = 0.01 (Low relevance).
                    scores.append(1 - norm_logprobs)

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