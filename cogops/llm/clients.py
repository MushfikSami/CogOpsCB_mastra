"""
cogops/llm/clients.py

AsyncOpenAI client factory for the primary LLM endpoint.
"""

import asyncio
import os
import logging
from typing import Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI

from cogops.config.loader import EndpointConfig

load_dotenv()

logger = logging.getLogger(__name__)


class AsyncLLMService:
    """
    Async LLM service for vLLM-compatible endpoints.
    Single primary client for reasoning.
    """

    def __init__(
        self,
        client_llm: Optional[AsyncOpenAI] = None,
        config_llm: Optional[EndpointConfig] = None,
    ):
        if config_llm:
            self.client_llm = AsyncOpenAI(
                api_key=config_llm.api_key, base_url=config_llm.base_url,
            )
            self.llm_config = config_llm
        else:
            self.client_llm = client_llm
            self.llm_config = config_llm or EndpointConfig("", "", "", 0)

    @property
    def model(self):
        return self.llm_config.model if self.llm_config else ""

    @property
    def max_context_tokens(self):
        if not self.llm_config:
            return 32000
        return self.llm_config.max_context_tokens or 32000

    async def health_check(self, timeout: float = 5.0) -> str:
        """Check if the primary LLM is reachable."""
        try:
            await asyncio.wait_for(
                self.client_llm.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": "ok"}],
                    max_tokens=1,
                ),
                timeout=timeout,
            )
            return "ok"
        except Exception as e:
            logger.warning("LLM health check failed: %s", e)
            return "error"
