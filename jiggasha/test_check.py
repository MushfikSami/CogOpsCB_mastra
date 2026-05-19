#!/usr/bin/env python3
"""Integration tests for Jiggasha pipeline — tests all components live.

Usage:
    python3 test_check.py
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import requests
import yaml
from embedder import Embedder

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(SCRIPT_DIR, "config.yml"), "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)


def _resolve_env(key: str | None) -> str:
    if not key:
        return ""
    if key not in os.environ:
        env_path = os.path.join(SCRIPT_DIR, ".env")
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        k, v = line.split("=", 1)
                        os.environ.setdefault(k.strip(), v.strip())
    return os.environ.get(key, "")


def test_embedder():
    """Test embedder can produce 4096-dim vectors."""
    print("\n[Test 1] Embedder")
    url = _resolve_env(cfg["embedder"].get("url_env"))
    key = _resolve_env(cfg["embedder"].get("api_key_env"))
    model = _resolve_env(cfg["embedder"].get("model_env")) or cfg["embedder"].get("model", "")
    e = Embedder(url=url, api_key=key, model=model)

    vec = e.embed("এনআইডি কার্ড ডাউনলোড করার পদ্ধতি")
    assert len(vec) == 4096, f"Expected 4096-dim, got {len(vec)}"
    norm = sum(v * v for v in vec) ** 0.5
    assert 0.5 < norm < 2.0, f"Unexpected norm: {norm:.4f}"
    print(f"  PASS — vector dim={len(vec)}, norm={norm:.4f}")


def test_qdrant_health():
    """Test Qdrant is reachable and collection exists."""
    print("\n[Test 3] Qdrant")
    from qdrant_client import QdrantClient

    qd = QdrantClient(_resolve_env(cfg["qdrant"].get("url_env")))
    cnt = qd.count(cfg["qdrant"]["collection"]).count
    assert cnt > 0, "Collection has 0 points"
    print(f"  PASS — collection '{cfg['qdrant']['collection']}' has {cnt:,} points")


def test_service_health():
    """Test FastAPI /health endpoint."""
    print("\n[Test 4] Service health")
    port = cfg.get("port", 10000)
    resp = requests.get(f"http://localhost:{port}/health", timeout=10).json()
    assert resp.get("qdrant", {}).get("status") == "ok", f"Qdrant: {resp.get('qdrant')}"
    assert resp.get("embedder", {}).get("status") == "ok", f"Embedder: {resp.get('embedder')}"
    print(f"  PASS — qdrant={resp['qdrant'].get('points')}, embedder=ok")


def test_search():
    """Test full search pipeline with a real query."""
    print("\n[Test 5] Search")
    port = cfg.get("port", 10000)
    query = "এনআইডি কার্ড ডাউনলোড"
    resp = requests.post(
        f"http://localhost:{port}/search",
        json={"query": query, "top_k": 30},
        timeout=60,
    ).json()
    results = resp.get("results", [])
    assert len(results) > 0, f"No results for query '{query}': {resp}"
    first = results[0]
    assert first["passage_id"] > 0
    assert first["text"].strip() != ""
    print(f"  PASS — {len(results)} results, top: id={first['passage_id']} score={first['score']:.4f}")
    print(f"  First result: {first['text'][:100]}...")


def test_off_topic_returns_low_top_score():
    """Off-topic query should have a low top cosine score.

    With no service-side filtering, the response will still contain top-K hits,
    but the TOP cosine should be noticeably lower than for an in-corpus query.
    This is informational — the chatbot-side LLM relevance filter is the
    actual gate.
    """
    print("\n[Test 6] Off-topic query — top cosine should be low")
    port = cfg.get("port", 10000)
    query = "চাঁদের আলো দেখে আমি কাব্য রচনা করতে চাই"
    resp = requests.post(
        f"http://localhost:{port}/search",
        json={"query": query, "top_k": 20},
        timeout=30,
    ).json()
    results = resp.get("results", [])
    top_score = results[0]["score"] if results else 0.0
    # Informational threshold: in-corpus queries usually score 0.70+ on this model.
    assert top_score < 0.55, (
        f"Off-topic query top cosine={top_score:.4f} — unexpectedly high; "
        f"corpus may be drifting or query isn't actually off-topic"
    )
    print(f"  PASS — top cosine={top_score:.4f} (< 0.55, looks off-topic)")


def test_search_multi_rerank():
    """Multi-query rerank path: sub_queries + rerank=true → vetted bucketed results."""
    print("\n[Test 7] Multi-query rerank (/search with sub_queries)")
    port = cfg.get("port", 10000)
    sub_queries = [
        "এনআইডি সংশোধন কোথায়?",
        "পাসপোর্ট ফি কত?",
    ]
    t0 = time.time()
    resp = requests.post(
        f"http://localhost:{port}/search",
        json={
            "sub_queries": sub_queries,
            "top_k_per_sub": 15,
            "rerank": True,
            "candidate_cap_global": 30,
            "keep_cap": 24,
            "weak_per_sub_cap": 3,
        },
        timeout=60,
    ).json()
    elapsed = time.time() - t0

    assert "passages" in resp, f"Missing 'passages' in response: {resp}"
    assert "rerank" in resp, f"Missing 'rerank' in response: {resp}"
    passages = resp["passages"]
    rerank = resp["rerank"]
    degraded = resp.get("degraded", False)

    assert len(passages) > 0, f"No passages returned: {resp}"
    # rerank keys must be string-1-based and cover every sub-query.
    assert set(rerank.keys()) == {"1", "2"}, f"rerank keys: {list(rerank.keys())}"
    pids_in_passages = {p["passage_id"] for p in passages}
    for sub_key, entries in rerank.items():
        for entry in entries:
            assert isinstance(entry, list) and len(entry) == 2, (
                f"Malformed rerank entry: {entry}"
            )
            pid, cls = entry
            assert pid in pids_in_passages, (
                f"rerank references pid={pid} not in passages"
            )
            assert cls in (0, 1), f"rerank class must be 0 or 1, got {cls}"

    # In a healthy run we expect at least one "yes" verdict and no degraded
    # fallback. Treat degraded as a soft warning, not a hard fail.
    yes_total = sum(1 for v in rerank.values() for _, cls in v if cls == 0)
    print(
        f"  PASS — passages={len(passages)} yes_votes={yes_total} "
        f"degraded={degraded} elapsed={int(elapsed * 1000)}ms"
    )
    if degraded:
        print("  WARN — rerank degraded to cosine safety net (LLM unavailable)")


def main():
    print("=" * 60)
    print("Jiggasha Integration Tests")
    print("=" * 60)

    passed = 0
    failed = 0
    tests = [
        ("embedder", test_embedder),
        ("qdrant_health", test_qdrant_health),
        ("service_health", test_service_health),
        ("search", test_search),
        ("off_topic_low_top_score", test_off_topic_returns_low_top_score),
        ("search_multi_rerank", test_search_multi_rerank),
    ]

    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  FAIL — {e}")

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    print("=" * 60)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
