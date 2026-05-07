"""Configuration: .env + config.yml, cached singleton."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
import yaml

_BASE_DIR = Path(__file__).resolve().parent.parent
_ENV_PATH = _BASE_DIR / ".env"
_CONFIG_PATH = _BASE_DIR / "config.yml"


def _load_env() -> None:
    if _ENV_PATH.exists():
        load_dotenv(_ENV_PATH)


_load_env()


@dataclass(frozen=True)
class Config:
    # LLM
    openai_api_key: str = ""
    openai_base_url: str | None = None
    llm_model: str = "qwen36"

    # Embedding
    embedder_api_url: str = "http://172.22.8.106:5001/v1"
    embedding_model: str = "qwen3-embed"
    embedding_api_key: str = "qwen3_emb"

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection_web: str = "jiggasha_webnode"
    qdrant_collection_local: str = "jiggasha_database"

    # Retrieval
    top_k: int = 10

    # TAO-ReAct loop
    max_tao_steps: int = 5
    loop_stop_threshold: float = 0.8
    min_relevant_passages: int = 3

    # Reranker
    rerank_threshold: float = 0.5
    max_batch_size: int = 20
    rerank_max_tokens: int = 1

    # Task instruction for embeddings
    retrieval_task_instruct: str = (
        "Given a web search query, retrieve relevant passages "
        "that answer the query"
    )

    @staticmethod
    def _from_env() -> dict:
        return {
            "openai_api_key": os.environ.get("OPENAI_API_KEY", ""),
            "openai_base_url": os.environ.get("OPENAI_BASE_URL") or None,
            "llm_model": os.environ.get("LLM_MODEL", "qwen36"),
            "embedder_api_url": os.environ.get(
                "EMBEDDER_API_URL", "http://172.22.8.106:5001/v1"
            ),
            "embedding_model": os.environ.get("EMBEDDING_MODEL", "qwen3-embed"),
            "embedding_api_key": os.environ.get("EMBEDDING_API_KEY", "qwen3_emb"),
            "qdrant_url": os.environ.get("QDRANT_URL", "http://localhost:6333"),
            "qdrant_collection_web": os.environ.get(
                "QDRANT_COLLECTION_WEB", "jiggasha_webnode"
            ),
            "qdrant_collection_local": os.environ.get(
                "QDRANT_COLLECTION_LOCAL", "jiggasha_database"
            ),
            "retrieval_task_instruct": os.environ.get(
                "RETRIEVAL_TASK_INSTRUCT",
                "Given a web search query, retrieve relevant passages that answer the query",
            ),
        }

    @staticmethod
    def _from_yaml() -> dict:
        if not _CONFIG_PATH.exists():
            return {}
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        retrieval = data.get("retrieval", {})
        agent_cfg = data.get("agent", {})
        reranker = data.get("reranker", {})
        fact_check = data.get("fact_check", {})
        return {
            "top_k": retrieval.get("top_k", 10),
            "max_tao_steps": agent_cfg.get("max_tao_steps", 5),
            "loop_stop_threshold": agent_cfg.get("loop_stop_threshold", 0.8),
            "min_relevant_passages": agent_cfg.get("min_relevant_passages", 3),
            "rerank_threshold": reranker.get("threshold", 0.5),
            "max_batch_size": reranker.get("max_batch_size", 20),
            "rerank_max_tokens": reranker.get("max_tokens", 1),
        }

    @staticmethod
    def default() -> "Config":
        env_vars = Config._from_env()
        yaml_vars = Config._from_yaml()
        merged = {**env_vars, **yaml_vars}
        return Config(**merged)


_config: Config | None = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = Config.default()
    return _config
