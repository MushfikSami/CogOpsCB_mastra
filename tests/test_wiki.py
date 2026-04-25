"""test_wiki.py — tests for wikipedia_search's OpenSearch + ChromaDB fallback."""
import asyncio
import sys
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _reset_wiki():
    for mod in list(sys.modules.keys()):
        if mod.startswith("cogops.tools.wiki"):
            del sys.modules[mod]
    yield


def _arun(coro):
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


# ---------------------------------------------------------------------------
# wikipedia_search — OpenSearch primary path
# ---------------------------------------------------------------------------

class TestWikipediaSearchOpenSearch:
    """OpenSearch (action=opensearch) returns titles — no fallback needed."""

    def test_returns_title_suggestions(self):
        opensearch_response = [
            "বাংলাদেশ",
            ["বাংলাদেশ", "বাংলাদেশের ইতিহাস"],
            ["দক্ষিণ এশিয়ার রাষ্ট্র", "ঐতিহাসিক বিবরণ"],
            [
                "https://bn.wikipedia.org/wiki/বাংলাদেশ",
                "https://bn.wikipedia.org/wiki/বাংলাদেশের_ইতিহাস",
            ],
        ]
        with patch("cogops.tools.wiki._do_get", return_value=opensearch_response):
            from cogops.tools.wiki import wikipedia_search
            result = _arun(wikipedia_search("বাংলাদেশ", top=2))
        assert "বাংলাদেশ" in result
        assert "বাংলাদেশের ইতিহাস" in result
        assert "Wikipedia শিরোনাম প্রস্তাব" in result
        # Must NOT be the fallback label
        assert "ভেক্টর ফলব্যাক" not in result

    def test_empty_query(self):
        from cogops.tools.wiki import wikipedia_search
        result = _arun(wikipedia_search("", top=3))
        assert "খালি" in result

    def test_api_error_returns_graceful_message(self):
        with patch("cogops.tools.wiki._do_get", return_value=None):
            from cogops.tools.wiki import wikipedia_search
            result = _arun(wikipedia_search("যে কোনো প্রশ্ন", top=3))
        assert "ত্রুটি" in result

    def test_malformed_response(self):
        with patch("cogops.tools.wiki._do_get", return_value={"not": "a list"}):
            from cogops.tools.wiki import wikipedia_search
            result = _arun(wikipedia_search("প্রশ্ন", top=3))
        assert "অপ্রত্যাশিত" in result


# ---------------------------------------------------------------------------
# wikipedia_search — ChromaDB fallback when OpenSearch returns no titles
# ---------------------------------------------------------------------------

class TestWikipediaSearchChromaFallback:
    """OpenSearch returns empty titles → internal ChromaDB lookup kicks in."""

    def _empty_opensearch(self):
        # OpenSearch shape with empty title list
        return ["query", [], [], []]

    def _mock_embedder(self):
        m = MagicMock()
        m.create_sync.return_value = [0.0] * 768
        return m

    def _mock_chroma_client(self, titles=None, distances=None):
        titles = titles if titles is not None else ["বাংলাদেশ", "ঢাকা"]
        distances = distances if distances is not None else [0.11, 0.22]
        client = MagicMock()
        client.heartbeat.return_value = None
        coll = MagicMock()
        coll.query.return_value = {
            "documents": [titles],
            "distances": [distances],
        }
        client.get_collection.return_value = coll
        return client

    @patch("cogops.tools.wiki.chromadb.HttpClient")
    @patch("cogops.tools.wiki.TritonEmbedder")
    @patch("cogops.tools.wiki._do_get")
    def test_falls_back_to_chroma_and_returns_titles(
        self, do_get_mock, EmbedderMock, ChromaMock
    ):
        do_get_mock.return_value = self._empty_opensearch()
        EmbedderMock.return_value = self._mock_embedder()
        ChromaMock.return_value = self._mock_chroma_client()

        from cogops.tools.wiki import wikipedia_search
        result = _arun(wikipedia_search("একদম বিরল বিষয়", top=2))
        assert "ভেক্টর ফলব্যাক" in result
        assert "বাংলাদেশ" in result
        assert "ঢাকা" in result
        assert "দূরত্ব" in result

    @patch("cogops.tools.wiki.chromadb.HttpClient")
    @patch("cogops.tools.wiki.TritonEmbedder")
    @patch("cogops.tools.wiki._do_get")
    def test_falls_back_and_chroma_empty(
        self, do_get_mock, EmbedderMock, ChromaMock
    ):
        do_get_mock.return_value = self._empty_opensearch()
        EmbedderMock.return_value = self._mock_embedder()
        ChromaMock.return_value = self._mock_chroma_client(titles=[], distances=[])

        from cogops.tools.wiki import wikipedia_search
        result = _arun(wikipedia_search("কোনো মিল নেই", top=3))
        assert "পাওয়া যায়নি" in result

    @patch("cogops.tools.wiki.chromadb.HttpClient")
    @patch("cogops.tools.wiki.TritonEmbedder")
    @patch("cogops.tools.wiki._do_get")
    def test_chroma_connection_error(
        self, do_get_mock, EmbedderMock, ChromaMock
    ):
        do_get_mock.return_value = self._empty_opensearch()
        EmbedderMock.return_value = self._mock_embedder()
        ChromaMock.side_effect = Exception("connection refused")

        from cogops.tools.wiki import wikipedia_search
        result = _arun(wikipedia_search("প্রশ্ন", top=3))
        assert "ক্রোমাডিবি সংযোগ ত্রুটি" in result

    @patch("cogops.tools.wiki.chromadb.HttpClient")
    @patch("cogops.tools.wiki.TritonEmbedder")
    @patch("cogops.tools.wiki._do_get")
    def test_embedder_error(
        self, do_get_mock, EmbedderMock, ChromaMock
    ):
        do_get_mock.return_value = self._empty_opensearch()
        bad = MagicMock()
        bad.create_sync.side_effect = Exception("triton timeout")
        EmbedderMock.return_value = bad
        ChromaMock.return_value = self._mock_chroma_client()

        from cogops.tools.wiki import wikipedia_search
        result = _arun(wikipedia_search("প্রশ্ন", top=3))
        assert "ট্রাইটন এম্বেডিং ত্রুটি" in result


# ---------------------------------------------------------------------------
# Registry exposure
# ---------------------------------------------------------------------------

class TestWikiRegistryExposure:
    """wikipedia_title_suggest must NOT be exposed; only the three public tools."""

    def test_public_tool_names(self):
        from cogops.tools.wiki import wikipedia_tools_map
        assert set(wikipedia_tools_map.keys()) == {
            "wikipedia_search",
            "wikipedia_get_summary",
            "wikipedia_get_full_content",
        }

    def test_title_suggest_not_in_schema(self):
        from cogops.tools.wiki import wikipedia_tools_list
        names = {t["function"]["name"] for t in wikipedia_tools_list}
        assert "wikipedia_title_suggest" not in names
