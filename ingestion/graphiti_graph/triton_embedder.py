import numpy as np
import tritonclient.http as httpclient
from transformers import AutoTokenizer
from graphiti_core.embedder.client import EmbedderClient, EmbedderConfig
from typing import List, Union, Iterable

class TritonEmbedderConfig(EmbedderConfig):
    url: str = "localhost:6000"
    model_name: str = "gemma_embedding"
    tokenizer_path: str = "onnx-community/embeddinggemma-300m-ONNX"
    instruction_prefix: str = "title: none | text: "
    # SERVER LIMIT: Strict batch size limit of 8
    max_batch_size: int = 8 

class TritonEmbedder(EmbedderClient):
    """
    Custom Embedder Client with automatic batching to respect Triton limits.
    """
    def __init__(self, config: TritonEmbedderConfig = None):
        if config is None:
            config = TritonEmbedderConfig()
        self.config = config
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        self.client = httpclient.InferenceServerClient(url=self.config.url, verbose=False)
        
        if not self.client.is_server_live():
            raise ConnectionError(f"Triton server at {self.config.url} is not live.")

    def _infer(self, texts: List[str]) -> np.ndarray:
        """Helper to send a specific list of texts to Triton."""
        processed_texts = [self.config.instruction_prefix + t for t in texts]

        tokens = self.tokenizer(
            processed_texts, 
            return_tensors='np', 
            padding=True, 
            truncation=True,
            max_length=2048
        )

        input_ids = tokens['input_ids'].astype(np.int64)
        attention_mask = tokens['attention_mask'].astype(np.int64)

        inputs = [
            httpclient.InferInput('input_ids', input_ids.shape, "INT64"),
            httpclient.InferInput('attention_mask', attention_mask.shape, "INT64")
        ]
        inputs[0].set_data_from_numpy(input_ids)
        inputs[1].set_data_from_numpy(attention_mask)

        response = self.client.infer(model_name=self.config.model_name, inputs=inputs)
        return response.as_numpy('sentence_embedding')

    async def create(
        self, input_data: Union[str, List[str], Iterable[int], Iterable[Iterable[int]]]
    ) -> List[float]:
        if isinstance(input_data, str):
            input_list = [input_data]
        elif isinstance(input_data, list) and isinstance(input_data[0], str):
            input_list = input_data
        else:
            raise NotImplementedError("Raw integer token embedding not implemented")

        embeddings = self._infer(input_list)
        return embeddings[0].tolist()

    async def create_batch(self, input_data_list: List[str]) -> List[List[float]]:
        """
        Generates a batch of embeddings, splitting them into smaller chunks
        to respect the Triton server's max_batch_size limit.
        """
        all_embeddings = []
        batch_size = self.config.max_batch_size
        
        # Loop through data in chunks of 8 (or config limit)
        for i in range(0, len(input_data_list), batch_size):
            chunk = input_data_list[i : i + batch_size]
            
            # Infer just this chunk
            chunk_embeddings = self._infer(chunk)
            
            # Add to results
            all_embeddings.extend(chunk_embeddings.tolist())
            
        return all_embeddings