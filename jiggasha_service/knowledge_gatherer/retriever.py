"""Qdrant retrieval for dual-mode (web vs local).

mode=web   -> jiggasha_webnode collection (web_node only, production)
mode=local -> jiggasha_database collection (full text + web_node, testing)
"""

from __future__ import annotations

import logging

from qdrant_client import QdrantClient

from .config import Config, get_config
from .embedder import EmbedderClient

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(
        self,
        config: Config | None = None,
        embedder: EmbedderClient | None = None,
    ) -> None:
        self._config = config or get_config()
        self._embedder = embedder or EmbedderClient(self._config)
        self._client = QdrantClient(url=self._config.qdrant_url)

    def _resolve_collection(self, mode: str) -> str:
        if mode == "local":
            return self._config.qdrant_collection_local
        return self._config.qdrant_collection_web

    async def retrieve(
        self,
        query: str,
        mode: str,
        top_k: int | None = None,
    ) -> list[dict]:
        top_k = top_k or self._config.top_k
        collection = self._resolve_collection(mode)
        query_vector = await self._embedder.embed_query(query)

        search_result = self._client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=top_k,
        ).points

        passages = []
        for hit in search_result:
            payload = hit.payload or {}
            doc = payload.get("doc", "") or ""
            # doc = "category :: sub :: service :: topic\n<text>"
            # Extract web_node: first 4 " :: " separated tokens
            web_node = ""
            if " :: " in doc:
                parts = doc.split(" :: ")
                if len(parts) >= 4:
                    web_node = " :: ".join(parts[:4])
            passages.append(
                {
                    "id": hit.id,
                    "text": doc,
                    "web_node": web_node,
                    "url": payload.get("url", ""),
                    "score": float(hit.score),
                    "source_collection": collection,
                }
            )

        return passages
