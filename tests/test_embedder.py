"""test_embedder.py — Phase 1: Triton embedder output shape tests."""
import sys
import pytest


@pytest.fixture(autouse=True)
def _reset_modules():
    for mod in list(sys.modules.keys()):
        if mod.startswith("cogops.embedders"):
            del sys.modules[mod]
    yield


class TestTritonEmbedder:
    """TritonEmbedder output shapes match ABC contract."""

    def test_create_str_returns_flat_list(self):
        """create("str") returns list[float], not list[list[float]]."""
        from cogops.embedders.triton import TritonEmbedder, TritonEmbedderConfig
        triton_conf = TritonEmbedderConfig(
            url="localhost:6000", model_name="gemma_embedding",
            tokenizer_path="onnx-community/embeddinggemma-300m-ONNX",
            max_batch_size=8,
        )
        embedder = TritonEmbedder(config=triton_conf)
        result = asyncio_run(embedder.create(input_data="test"))
        assert isinstance(result, list)
        assert len(result) == 768
        # Should be flat, not nested
        assert isinstance(result[0], float)

    def test_create_list_returns_flat_first_embedding(self):
        """create(["str"]) returns first embedding as list[float]."""
        from cogops.embedders.triton import TritonEmbedder, TritonEmbedderConfig
        triton_conf = TritonEmbedderConfig(
            url="localhost:6000", model_name="gemma_embedding",
            tokenizer_path="onnx-community/embeddinggemma-300m-ONNX",
            max_batch_size=8,
        )
        embedder = TritonEmbedder(config=triton_conf)
        result = asyncio_run(embedder.create(input_data=["test"]))
        assert isinstance(result, list)
        assert len(result) == 768
        assert isinstance(result[0], float)

    def test_create_batch_returns_nested(self):
        """create_batch(["a", "b"]) returns list[list[float]]."""
        from cogops.embedders.triton import TritonEmbedder, TritonEmbedderConfig
        triton_conf = TritonEmbedderConfig(
            url="localhost:6000", model_name="gemma_embedding",
            tokenizer_path="onnx-community/embeddinggemma-300m-ONNX",
            max_batch_size=8,
        )
        embedder = TritonEmbedder(config=triton_conf)
        result = asyncio_run(embedder.create_batch(["test1", "test2"]))
        assert isinstance(result, list)
        assert len(result) == 2
        assert isinstance(result[0], list)
        assert len(result[0]) == 768
        assert isinstance(result[0][0], float)

    def test_no_nan_or_inf(self):
        """Embedding values must be valid floats."""
        from cogops.embedders.triton import TritonEmbedder, TritonEmbedderConfig
        import math
        triton_conf = TritonEmbedderConfig(
            url="localhost:6000", model_name="gemma_embedding",
            tokenizer_path="onnx-community/embeddinggemma-300m-ONNX",
            max_batch_size=8,
        )
        embedder = TritonEmbedder(config=triton_conf)
        result = asyncio_run(embedder.create(input_data="test"))
        for val in result:
            assert not math.isnan(val), f"NaN found in embedding"
            assert not math.isinf(val), f"Inf found in embedding"

    def test_nonzero_norm(self):
        """L2 norm should be non-zero and reasonable."""
        from cogops.embedders.triton import TritonEmbedder, TritonEmbedderConfig
        triton_conf = TritonEmbedderConfig(
            url="localhost:6000", model_name="gemma_embedding",
            tokenizer_path="onnx-community/embeddinggemma-300m-ONNX",
            max_batch_size=8,
        )
        embedder = TritonEmbedder(config=triton_conf)
        result = asyncio_run(embedder.create(input_data="test"))
        norm = sum(x * x for x in result) ** 0.5
        assert norm > 0.01, f"Embedding norm too small: {norm}"


def asyncio_run(coro):
    import asyncio
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop:
        return loop.run_until_complete(coro)
    new_loop = asyncio.new_event_loop()
    try:
        return new_loop.run_until_complete(coro)
    finally:
        new_loop.close()
