"""Embedding client for qwen3-embed at 172.22.8.106:5001.

Queries are encoded with the instruction prefix format:
  Instruct: {task_desc}
  Query: {query}
Documents are embedded as raw text (no prefix).
"""

from __future__ import annotations

import logging

import aiohttp

from .config import Config, get_config

logger = logging.getLogger(__name__)


def get_detailed_instruct(task_desc: str, query: str) -> str:
    return f"Instruct: {task_desc}\nQuery: {query}"


class EmbedderClient:
    def __init__(self, config: Config | None = None) -> None:
        self._config = config or get_config()
        self._api_url = f"{self._config.embedder_api_url}/embeddings"
        self._model = self._config.embedding_model
        self._api_key = self._config.embedding_api_key
        self._task_instruct = self._config.retrieval_task_instruct

    async def embed_query(self, query: str) -> list[float]:
        formatted = get_detailed_instruct(self._task_instruct, query)
        return (await self._embed([formatted]))[0]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._embed(texts)

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        payload = {
            "model": self._model,
            "input": texts,
            "encoding_format": "float",
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self._api_url, json=payload, headers=headers, timeout=60
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                return [item["embedding"] for item in data["data"]]
