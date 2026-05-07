"""
cogops/llm/clients.py

AsyncOpenAI client factories for primary LLM, reranker, secondary endpoints.
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
    Supports primary LLM (reasoning), reranker, and secondary (summarizer).
    """

    def __init__(
        self,
        client_llm: Optional[AsyncOpenAI] = None,
        client_reranker: Optional[AsyncOpenAI] = None,
        client_secondary: Optional[AsyncOpenAI] = None,
        config_llm: Optional[EndpointConfig] = None,
        config_reranker: Optional[EndpointConfig] = None,
        config_secondary: Optional[EndpointConfig] = None,
    ):
        if config_llm:
            self._init_from_configs(config_llm, config_reranker, config_secondary)
        else:
            self.client_llm = client_llm
            self.client_reranker = client_reranker
            self.client_secondary = client_secondary
            self.llm_config = config_llm or EndpointConfig("", "", "", 0)

    def _init_from_configs(self, config_llm, config_reranker, config_secondary):
        self.client_llm = AsyncOpenAI(
            api_key=config_llm.api_key, base_url=config_llm.base_url,
        )
        self.client_reranker = (
            AsyncOpenAI(api_key=config_reranker.api_key, base_url=config_reranker.base_url)
            if config_reranker else None
        )
        self.client_secondary = (
            AsyncOpenAI(api_key=config_secondary.api_key, base_url=config_secondary.base_url)
            if config_secondary else None
        )
        self.llm_config = config_llm

    @property
    def model(self):
        return self.llm_config.model if self.llm_config else ""

    @property
    def max_context_tokens(self):
        return self.llm_config.max_context_tokens if self.llm_config else 32000

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
