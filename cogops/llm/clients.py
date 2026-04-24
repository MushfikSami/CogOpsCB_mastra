"""
cogops/llm/clients.py

AsyncOpenAI client factories: creates clients for primary LLM, reranker, secondary.
Moved from cogops/models/llm.py (the init part).
"""

import os
import logging
from typing import Optional
from dotenv import load_dotenv
from openai import AsyncOpenAI

from cogops.config.loader import EndpointConfig

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AsyncLLMService:
    """
    Asynchronous LLM Service for vLLM with native thinking support
    and multi-endpoint capability (primary LLM, reranker, secondary).

    Client factories live here; the tool-calling loop is in reasoning_loop.py.
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
        """
        Initialize with either client instances or endpoint configs.
        If configs provided, clients are auto-created.
        """
        if config_llm:
            self._init_from_configs(config_llm, config_reranker, config_secondary)
        else:
            self.client_llm = client_llm
            self.client_reranker = client_reranker
            self.client_secondary = client_secondary
            self.llm_config = config_llm or EndpointConfig("", "", "", 0)

        thinking_enabled = self.llm_config.thinking if self.llm_config else False
        logger.info(
            f"AsyncLLMService initialized: "
            f"llm={self.client_llm.base_url if self.client_llm else 'None'}, "
            f"reranker={self.client_reranker.base_url if self.client_reranker else 'None'}, "
            f"secondary={self.client_secondary.base_url if self.client_secondary else 'None'}, "
            f"thinking={thinking_enabled}"
        )

    def _init_from_configs(self, config_llm, config_reranker, config_secondary):
        self.client_llm = AsyncOpenAI(api_key=config_llm.api_key, base_url=config_llm.base_url)
        self.client_reranker = AsyncOpenAI(api_key=config_reranker.api_key, base_url=config_reranker.base_url) if config_reranker else None
        self.client_secondary = AsyncOpenAI(api_key=config_secondary.api_key, base_url=config_secondary.base_url) if config_secondary else None
        self.llm_config = config_llm

    @property
    def model(self):
        return self.llm_config.model if self.llm_config else ""

    @property
    def max_context_tokens(self):
        return self.llm_config.max_context_tokens if self.llm_config else 32000

    @property
    def client_secondary_client(self):
        return self.client_secondary
