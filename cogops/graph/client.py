"""
cogops/graph/client.py

Graphiti client singleton: get_graphiti_client().
Moved from cogops/tools/graphiti_tools.py (the client/init part).
"""

import os
import logging
from typing import Optional

from dotenv import load_dotenv
from graphiti_core import Graphiti
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.driver.neo4j_driver import Neo4jDriver
from openai import AsyncOpenAI

from cogops.embedders.triton import TritonEmbedder, TritonEmbedderConfig
from cogops.llm.reranker import QwenRerankerClient

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global Singleton ---
_GRAPHITI_CLIENT: Optional[Graphiti] = None


async def get_graphiti_client() -> Graphiti:
    global _GRAPHITI_CLIENT
    if _GRAPHITI_CLIENT is None:
        llm_config = LLMConfig(
            api_key=os.getenv("LLM_API_KEY", "sk-placeholder"),
            base_url=os.getenv("LLM_BASE_URL"),
            model=os.getenv("LLM_MODEL_NAME"),
            max_tokens=150000
        )
        llm_client = OpenAIGenericClient(config=llm_config)

        triton_conf = TritonEmbedderConfig(
            url=os.getenv("TRITON_URL", "localhost:6000"),
            model_name=os.getenv("TRITON_MODEL_NAME", "gemma_embedding"),
            tokenizer_path=os.getenv("TRITON_TOKENIZER", "onnx-community/embeddinggemma-300m-ONNX"),
            max_batch_size=8
        )
        embedder = TritonEmbedder(config=triton_conf)

        reranker_llm_config = LLMConfig(
            api_key=os.getenv("RERANKER_API_KEY", "sk-placeholder"),
            base_url=os.getenv("RERANKER_BASE_URL"),
            model=os.getenv("RERANKER_MODEL_NAME"),
            max_tokens=1
        )
        inner_client = AsyncOpenAI(
            api_key=os.getenv("RERANKER_API_KEY", "sk-placeholder"),
            base_url=os.getenv("RERANKER_BASE_URL")
        )
        reranker = QwenRerankerClient(client=inner_client, config=reranker_llm_config)

        neo4j_driver = Neo4jDriver(
            uri=os.getenv("NEO4J_URI"),
            user=os.getenv("NEO4J_USER"),
            password=os.getenv("NEO4J_PASSWORD"),
            database=os.getenv("NEO4J_DATABASE", "neo4j")
        )

        _GRAPHITI_CLIENT = Graphiti(
            graph_driver=neo4j_driver,
            llm_client=llm_client,
            embedder=embedder,
            cross_encoder=reranker
        )
        logger.info("Graphiti client initialized.")
    return _GRAPHITI_CLIENT
