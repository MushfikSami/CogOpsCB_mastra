"""
cogops/embedders/triton.py

TritonEmbedder: custom Triton inference client for Gemma embeddings.
Moved from cogops/models/embedder.py.
"""

import logging
from typing import Iterable

from graphiti_core.embedder.client import EmbedderClient
from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TritonEmbedderConfig:
    def __init__(self, url: str, model_name: str, tokenizer_path: str, max_batch_size: int = 8):
        self.url = url
        self.model_name = model_name
        self.tokenizer_path = tokenizer_path
        self.max_batch_size = max_batch_size


class TritonEmbedder(EmbedderClient):
    """Embedder that sends texts to a Triton Inference Server for Gemma embeddings."""

    def __init__(self, config: TritonEmbedderConfig):
        self.config = config
        self.client = InferenceServerClient(url=config.url)
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
        logger.info(f"TritonEmbedder initialized: url={config.url}, model={config.model_name}")

    def _embed_text(self, text: str) -> list[float]:
        tokens = self.tokenizer(text, return_tensors="pt")
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]
        inputs = [
            InferInput("input_ids", input_ids.shape, "INT64"),
            InferInput("attention_mask", attention_mask.shape, "INT64"),
        ]
        inputs[0].set_data_from_numpy(input_ids.numpy())
        inputs[1].set_data_from_numpy(attention_mask.numpy())
        outputs = [InferRequestedOutput("sentence_embedding")]
        result = self.client.infer(model_name=self.config.model_name, inputs=inputs, outputs=outputs)
        embedding = result.as_numpy("sentence_embedding").tolist()
        # Triton returns [[...768 floats...]] (batch of 1), flatten to [...768 floats...]
        if isinstance(embedding, list) and len(embedding) > 0:
            if isinstance(embedding[0], list):
                return embedding[0]
            # Already flat (single embedding returned as list of floats)
            return embedding
        return embedding

    async def create(self, input_data: str | list[str]) -> list[float] | list[list[float]]:
        """Create embedding(s) for the given input."""
        if isinstance(input_data, str):
            return self._embed_text(input_data)
        elif isinstance(input_data, list):
            return await self.create_batch(input_data)
        raise ValueError(f"Unsupported input type: {type(input_data)}")

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        """Create embeddings for a batch of texts."""
        results = []
        for i in range(0, len(input_data_list), self.config.max_batch_size):
            batch = input_data_list[i:i + self.config.max_batch_size]
            for text in batch:
                results.append(self._embed_text(text))
        return results
