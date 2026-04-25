"""test_wiki.py — tests for the Wikipedia tool suite."""
import sys
from unittest.mock import MagicMock, patch

import pytest

# Force clean import of wiki module in every test.
@pytest.fixture(autouse=True)
def _reset_wiki():
    for mod in list(sys.modules.keys()):
        if mod.startswith("cogops.tools.wiki"):
            del sys.modules[mod]
    yield


# ---------------------------------------------------------------------------
# Async helper (mirrors test_tools_pure.py)
# ---------------------------------------------------------------------------

def _arun(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop:
        return loop.run_until_complete(coro)
    import asyncio
    new_loop = asyncio.new_event_loop()
    try:
        return new_loop.run_until_complete(coro)
    finally:
        new_loop.close()


import asyncio


# ---------------------------------------------------------------------------
# wikipedia_title_suggest — ChromaDB semantic search
# ---------------------------------------------------------------------------

class TestWikipediaTitleSuggest:
    """wikipedia_title_suggest: ChromaDB + Triton embedding fallback."""

    def _mock_response(self, titles=None, distances=None, metadatas=None):
        """Return a ChromaDB query() response dict."""
        titles = titles or ["বাংলাদেশ", "ঢাকা", "চট্টগ্রাম"]
        distances = distances or [0.12, 0.34, 0.56]
        metadatas = metadatas or [{"source": "bn_wiki"}] * len(titles)
        return {
            "ids": [["1", "2", "3"]],
            "documents": [titles],
            "distances": [distances],
            "metadatas": [metadatas],
        }

    def _make_embedder_mock(self):
        """Return a mock TritonEmbedder with create_sync."""
        m = MagicMock()
        m.create_sync.return_value = [0.0] * 768  # dummy 768-dim embedding
        m.create_batch_sync.return_value = [[0.0] * 768]
        return m

    def _make_chroma_mock(self):
        """Return a mock ChromaDB HttpClient."""
        client = MagicMock()
        client.heartbeat.return_value = None
        coll = MagicMock()
        coll.query.return_value = self._mock_response()
        client.get_collection.return_value = coll
        return client

    @patch("cogops.tools.wiki.chromadb.HttpClient")
    @patch("cogops.tools.wiki.TritonEmbedder")
    def test_returns_titles(self, EmbedderMock, ChromaMock):
        EmbedderMock.return_value = self._make_embedder_mock()
        ChromaMock.return_value = self._make_chroma_mock()

        result = _arun(
            __import__("cogops.tools.wiki", fromlist=["wikipedia_title_suggest"])
            .wikipedia_title_suggest("বাংলাদেশের রাজধানী", top=3)
        )
        assert "ভেক্টর মিল" in result
        assert "বাংলাদেশ" in result
        assert "ঢাকা" in result
        assert "চট্টগ্রাম" in result
        assert "দূরত্ব:" in result

    @patch("cogops.tools.wiki.chromadb.HttpClient")
    @patch("cogops.tools.wiki.TritonEmbedder")
    def test_empty_collection(self, EmbedderMock, ChromaMock):
        EmbedderMock.return_value = self._make_embedder_mock()
        client = self._make_chroma_mock()
        client.get_collection.return_value.query.return_value = {
            "documents": [[]],
            "distances": [[]],
            "metadatas": [[]],
        }
        ChromaMock.return_value = client

        result = _arun(
            __import__("cogops.tools.wiki", fromlist=["wikipedia_title_suggest"])
            .wikipedia_title_suggest("একদম অজানা বিষয়", top=3)
        )
        assert "মিল পাওয়া যায়নি" in result

    @patch("cogops.tools.wiki.chromadb.HttpClient")
    @patch("cogops.tools.wiki.TritonEmbedder")
    def test_chroma_connection_error(self, EmbedderMock, ChromaMock):
        EmbedderMock.return_value = self._make_embedder_mock()
        ChromaMock.side_effect = Exception("connection refused")

        result = _arun(
            __import__("cogops.tools.wiki", fromlist=["wikipedia_title_suggest"])
            .wikipedia_title_suggest("অনুসন্ধান", top=3)
        )
        assert "ত্রুটি" in result

    @patch("cogops.tools.wiki.chromadb.HttpClient")
    @patch("cogops.tools.wiki.TritonEmbedder")
    def test_embedder_error(self, EmbedderMock, ChromaMock):
        EmbedderMock.return_value = self._make_embedder_mock()
        EmbedderMock.return_value.create_sync.side_effect = Exception(
            "triton timeout"
        )

        result = _arun(
            __import__("cogops.tools.wiki", fromlist=["wikipedia_title_suggest"])
            .wikipedia_title_suggest("অনুসন্ধান", top=3)
        )
        assert "এম্বেডিং ত্রুটি" in result


# ---------------------------------------------------------------------------
# wikipedia_title_suggest — registry integration
# ---------------------------------------------------------------------------

class TestWikiRegistry:
    """Ensure wikipedia_title_suggest is in the registry."""

    def test_in_tools_map(self):
        from cogops.tools.wiki import wikipedia_tools_map
        assert "wikipedia_title_suggest" in wikipedia_tools_map

    def test_in_tools_list(self):
        from cogops.tools.wiki import wikipedia_tools_list
        names = {t["function"]["name"] for t in wikipedia_tools_list}
        assert "wikipedia_title_suggest" in names
