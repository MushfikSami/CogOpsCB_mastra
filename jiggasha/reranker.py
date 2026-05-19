"""Qwen3-Reranker-8B HTTP client via vLLM /v1/rerank."""
import json
import logging
import time

import requests

logger = logging.getLogger(__name__)


class Reranker:
    """Client for the vLLM reranking endpoint."""

    PREFIX = (
        "\n\n</s>\n<|begin_of_text|>system\n"
        "Judge whether the Document meets the requirements based on the Query "
        "and the Instruct provided. Note that the answer can only be "
        '"yes" or "no".\n\nuser\n'
    )
    SUFFIX = "\n\nassistant\n\n<|end|>\n\n"

    def __init__(self, url: str, api_key: str, model: str, top_k: int = 30, timeout: int = 20):
        self.url = url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self.model = model
        self.top_k = top_k
        self.timeout = timeout

    def rerank(self, query: str, documents: list[str], instruction: str | None = None) -> list[dict]:
        """Rerank documents by relevance to query.

        Args:
            query: The search query.
            documents: List of raw document texts.
            instruction: Custom instruction, defaults to retrieval instruction.

        Returns:
            List of dicts with keys: index, relevance_score, document.
            Sorted by relevance_score descending. Caller applies adaptive
            min-score filter and max_keep ceiling.
        """
        if not self.url:
            return []

        instruction = instruction or (
            "Given a web search query, retrieve relevant passages that answer the query"
        )
        query_template = f"{self.PREFIX}<Instruct>: {instruction}\n<Query>: {query}\n"
        doc_template = f"<Document>: {{doc}}{self.SUFFIX}"

        formatted_docs = [doc_template.format(doc=doc) for doc in documents]
        formatted_query = query_template

        t0 = time.time()
        try:
            resp = requests.post(
                self.url,
                json={
                    "model": self.model,
                    "query": formatted_query,
                    "documents": formatted_docs,
                },
                headers=self.headers,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            elapsed = time.time() - t0
            logger.info(f"Reranker responded in {elapsed:.1f}s — {len(data.get('results', []))} docs")

            results = []
            for item in data.get("results", []):
                results.append({
                    "index": item["index"],
                    "relevance_score": item["relevance_score"],
                    "document": item.get("document", {}).get("text", ""),
                })
            results.sort(key=lambda x: x["relevance_score"], reverse=True)
            return results
        except Exception as e:
            logger.warning(f"Reranker unavailable ({self.url}): {e}")
            return []


# --- CLI entry point (standalone reranking) ---

def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Qwen3-Reranker CLI")
    parser.add_argument("--query", required=True)
    parser.add_argument("--input", default=None)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--config", default="config.yml")
    args = parser.parse_args()

    import yaml

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Resolve env
    env_path = "env"
    env: dict[str, str] = {}
    for k in ["RERANKER_URL", "RERANKER_API_KEY", "RERANKER_MODEL", "EMBEDDER_URL"]:
        env[k] = os.environ.get(k, "")

    import os
    if env_path not in os.environ:
        env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
        if os.path.exists(env_file):
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())
    for k in ["RERANKER_URL", "RERANKER_API_KEY", "RERANKER_MODEL"]:
        env[k] = os.environ.get(k, "")

    reranker_cfg = cfg["reranker"]
    r = Reranker(
        url=os.environ.get(reranker_cfg["url_env"], ""),
        api_key=os.environ.get(reranker_cfg.get("api_key_env", ""), ""),
        model=os.environ.get(reranker_cfg.get("model_env", ""), "") or reranker_cfg.get("model", ""),
        top_k=reranker_cfg.get("top_k", 30),
    )

    if args.input:
        with open(args.input, "r", encoding="utf-8") as f:
            documents = json.load(f)
    else:
        documents = sys.stdin.read().strip().split("\n")

    results = r.rerank(args.query, documents)
    for item in results[: args.topk]:
        text = item["document"].replace("<Document>: ", "").split("\n\n")[0][:100]
        print(f"  score={item['relevance_score']:.4f}  [{item['index']}] {text}")


if __name__ == "__main__":
    main()
