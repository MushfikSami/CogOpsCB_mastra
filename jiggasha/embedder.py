"""Qwen3-Embedding-8B HTTP client."""
import logging
import time

import requests

logger = logging.getLogger(__name__)


class Embedder:
    """Client for the vLLM embedding endpoint."""

    def __init__(self, url: str, api_key: str, model: str, batch_size: int = 64, timeout: int = 120):
        self.url = url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self.model = model
        self.batch_size = batch_size
        self.timeout = timeout

    def embed(self, text: str) -> list[float]:
        """Embed a single text, return a 4096-dim float vector."""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts.

        Splits into sub-batches of self.batch_size.
        Returns one vector per input text.

        Raises RuntimeError on persistent failure — never returns zero vectors,
        which would silently corrupt the Qdrant collection.
        """
        all_embs: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            last_err: Exception | None = None
            for attempt in range(3):
                try:
                    resp = requests.post(
                        self.url,
                        json={"model": self.model, "input": batch, "encoding_format": "float"},
                        headers=self.headers,
                        timeout=self.timeout,
                    )
                    resp.raise_for_status()
                    embs = [item["embedding"] for item in resp.json()["data"]]
                    if len(embs) != len(batch):
                        raise RuntimeError(
                            f"Embedder returned {len(embs)} vectors for {len(batch)} inputs"
                        )
                    for j, vec in enumerate(embs):
                        norm_sq = sum(v * v for v in vec)
                        if norm_sq < 1e-6:
                            raise RuntimeError(
                                f"Embedder returned a zero-norm vector at batch offset {i+j}"
                            )
                    all_embs.extend(embs)
                    break
                except Exception as e:
                    last_err = e
                    wait = 2 ** attempt
                    logger.warning(
                        "Embedding batch at offset %d failed (attempt %d/3): %s — retrying in %ds",
                        i, attempt + 1, e, wait,
                    )
                    time.sleep(wait)
            else:
                raise RuntimeError(
                    f"Embedding failed permanently at batch offset {i} after 3 attempts: {last_err}"
                )
        return all_embs
