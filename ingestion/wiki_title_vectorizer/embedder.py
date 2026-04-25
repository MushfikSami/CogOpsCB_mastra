import json
import logging
from typing import Any, Dict, List
import numpy as np
import requests
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

QUERY_PREFIX = "task: search result | query: "
PASSAGE_PREFIX = "title: none | text: "

class GemmaTritonEmbedderConfig(BaseModel):
    """Configuration for the GemmaTritonEmbedder."""
    triton_url: str = Field(description="Base URL for the Triton Inference Server")
    triton_request_timeout: int = Field(default=480, description="Request timeout in seconds.")
    model_name: str = Field(default="gemma_embedding", description="Name of the model in Triton.")
    tokenizer_name: str = Field(default="onnx-community/embeddinggemma-300m-ONNX", description="HF tokenizer name.")
    triton_output_name: str = Field(default="sentence_embedding", description="Name of the output tensor.")
    batch_size: int = Field(default=8, description="Batch size for embedding requests sent to Triton.")

class _SyncGemmaTritonEmbedder:
    """Internal synchronous client that handles communication with Triton."""
    def __init__(self, config: GemmaTritonEmbedderConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    def _build_triton_payload(self, texts: List[str]) -> Dict[str, Any]:
        """Prepares the request payload for Triton."""
        tokens = self.tokenizer(texts, padding=True, truncation=True, max_length=2048, return_tensors="np")
        input_ids = tokens["input_ids"].astype(np.int64)
        attention_mask = tokens["attention_mask"].astype(np.int64)
        payload = {
            "inputs": [
                {"name": "input_ids", "shape": list(input_ids.shape), "datatype": "INT64", "data": input_ids.flatten().tolist()},
                {"name": "attention_mask", "shape": list(attention_mask.shape), "datatype": "INT64", "data": attention_mask.flatten().tolist()},
            ],
            "outputs": [{"name": self.config.triton_output_name}],
        }
        return payload

    def _post_process(self, triton_output: Dict[str, Any]) -> List[List[float]]:
        """Extracts the pooled embeddings from the Triton output."""
        output_data = next((out for out in triton_output["outputs"] if out["name"] == self.config.triton_output_name), None)
        if output_data is None:
            raise ValueError(f"Output '{self.config.triton_output_name}' not in Triton response.")
        
        shape = output_data["shape"]
        embeddings = np.array(output_data["data"], dtype=np.float32).reshape(shape)
        return embeddings.tolist()

    def embed(self, texts: List[str], model_name: str) -> List[List[float]]:
        """Creates embeddings for a list of texts using a synchronous request."""
        if not texts:
            return []
        api_url = f"{self.config.triton_url.rstrip('/')}/v2/models/{model_name}/infer"
        payload = self._build_triton_payload(texts)
        try:
            response = requests.post(
                api_url, 
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"},
                timeout=self.config.triton_request_timeout
            )
            response.raise_for_status()
            response_json = response.json()
            return self._post_process(response_json)
        except requests.exceptions.RequestException as e:
            logger.error(f"Error embedding texts with model {model_name}: {e}", exc_info=True)
            raise

class GemmaTritonEmbedder:
    """A synchronous client for EmbeddingGemma on Triton with separate query and passage embedding via prefixes."""
    def __init__(self, config: GemmaTritonEmbedderConfig):
        self.config = config
        self._client = _SyncGemmaTritonEmbedder(config)
        logger.info(f"Embedder initialized for Triton at {config.triton_url} with batch size {config.batch_size}")

    def embed_queries(self, texts: List[str]) -> List[List[float]]:
        """Embeds a batch of queries using the query prefix."""
        if not isinstance(texts, list) or not texts:
            return []
        texts_with_prefix = [QUERY_PREFIX + t for t in texts]
        all_embeddings = []
        for i in range(0, len(texts_with_prefix), self.config.batch_size):
            batch = texts_with_prefix[i : i + self.config.batch_size]
            logger.info(f"Sending query batch of {len(batch)} to Triton...")
            batch_embeddings = self._client.embed(batch, self.config.model_name)
            all_embeddings.extend(batch_embeddings)
        return all_embeddings

    def embed_passages(self, texts: List[str]) -> List[List[float]]:
        """Embeds a batch of documents/passages using the passage prefix."""
        if not isinstance(texts, list) or not texts:
            return []
        texts_with_prefix = [PASSAGE_PREFIX + t for t in texts]
        all_embeddings = []
        for i in range(0, len(texts_with_prefix), self.config.batch_size):
            batch = texts_with_prefix[i : i + self.config.batch_size]
            logger.info(f"Sending passage batch of {len(batch)} to Triton...")
            batch_embeddings = self._client.embed(batch, self.config.model_name)
            all_embeddings.extend(batch_embeddings)
        return all_embeddings

    def as_chroma_passage_embedder(self) -> EmbeddingFunction:
        """Returns an object that conforms to ChromaDB's EmbeddingFunction protocol."""
        class ChromaPassageEmbedder(EmbeddingFunction):
            def __init__(self, client: 'GemmaTritonEmbedder'):
                self._client = client
            def __call__(self, input: Documents) -> Embeddings:
                return self._client.embed_passages(input)
        return ChromaPassageEmbedder(self)

    def close(self):
        logger.info("Closing embedder (no-op for synchronous requests version).")
        pass