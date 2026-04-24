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

        # Patch Graphiti's node_search cross_encoder branch: it passes node
        # NAMES (e.g. "পাসপোর্ট") to the cross-encoder which is meaningless
        # for relevance classification. A name alone doesn't contain an answer.
        # Fix: use node.summary (the rich description) as the passage text.
        import graphiti_core.search.search as _search_mod
        _orig_node_search = _search_mod.node_search

        async def _patched_node_search(
            driver, cross_encoder, query, _query_vector, group_ids, config,
            search_filter, center_node_uuid, bfs_origin_uuids, limit,
            reranker_min_score,
        ):
            if config.reranker != "cross_encoder":
                return await _orig_node_search(
                    driver, cross_encoder, query, _query_vector, group_ids, config,
                    search_filter, center_node_uuid, bfs_origin_uuids, limit,
                    reranker_min_score,
                )
            # For cross_encoder: re-implement with summary text instead of names
            from graphiti_core.search.search import (
                rrf, maximal_marginal_relevance, get_embeddings_for_nodes,
                node_fulltext_search, node_vector_search,
                episode_mentions_reranker, node_distance_reranker,
            )
            from graphiti_core.search.search_config import NodeReranker
            # Replicate the pre-rerank search logic
            if config.full_text:
                search_results = await node_fulltext_search(
                    driver, query, search_filter, group_ids, 2 * limit,
                )
            else:
                search_results = await node_vector_search(
                    driver, _query_vector, search_filter, group_ids, 2 * limit,
                )

            search_result_uuids = [[n.uuid for n in result] for result in search_results]
            node_uuid_map = {n.uuid: n for result in search_results for n in result}

            reranked_uuids: list = []
            node_scores: list = []

            if config.reranker == NodeReranker.rrf:
                reranked_uuids, node_scores = rrf(search_result_uuids, min_score=reranker_min_score)
            elif config.reranker == NodeReranker.mmr:
                embeddings = await get_embeddings_for_nodes(driver, list(node_uuid_map.values()))
                reranked_uuids, node_scores = maximal_marginal_relevance(
                    _query_vector, embeddings, config.mmr_lambda, reranker_min_score,
                )
            elif config.reranker == NodeReranker.cross_encoder:
                # FIX: rank using summary (rich text) instead of name
                passage_to_uuid = {}
                for node in list(node_uuid_map.values()):
                    text = node.summary or node.name
                    passage_to_uuid[text] = node.uuid
                reranked = await cross_encoder.rank(query, list(passage_to_uuid.keys()))
                reranked_uuids = [
                    passage_to_uuid[p] for p, s in reranked if s >= reranker_min_score
                ]
                node_scores = [s for _, s in reranked if s >= reranker_min_score]
            elif config.reranker == NodeReranker.episode_mentions:
                reranked_uuids, node_scores = await episode_mentions_reranker(
                    driver, search_result_uuids, min_score=reranker_min_score,
                )
            elif config.reranker == NodeReranker.node_distance:
                reranked_uuids, node_scores = await node_distance_reranker(
                    driver,
                    rrf(search_result_uuids, min_score=reranker_min_score)[0],
                    center_node_uuid, min_score=reranker_min_score,
                )

            reranked_nodes = [node_uuid_map[u] for u in reranked_uuids]
            return reranked_nodes[:limit], node_scores[:limit]

        _search_mod.node_search = _patched_node_search

        logger.info("Graphiti client initialized. (cross_encoder node patch applied)")
    return _GRAPHITI_CLIENT
