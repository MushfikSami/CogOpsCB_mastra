#!/usr/bin/env python3
"""Integrity verifier for the Jiggasha Qdrant collection.

Checks:
  - Every point has a non-zero vector (norm > 0.5).
  - Point count matches CSV row count (after dropping empty-text rows).
  - No missing passage_id in [1..csv_count].

Exits non-zero if anything is wrong. Run after process.py to confirm a
clean ingest, or as a periodic health check.

Usage:
    python3 verify.py
    python3 verify.py --config config.yml
"""
import argparse
import csv
import logging
import math
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qdrant_client import QdrantClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _load_config(path: str = "config.yml") -> dict:
    if not os.path.isabs(path):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_env(key: str | None) -> str:
    if not key:
        return ""
    if key not in os.environ:
        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
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


def _count_csv_rows(csv_path: str) -> int:
    with open(csv_path, "r", encoding="utf-8") as f:
        count = 0
        for row in csv.DictReader(f):
            if row.get("text", "").strip():
                count += 1
        return count


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify Jiggasha Qdrant integrity")
    parser.add_argument("--config", default="config.yml")
    parser.add_argument(
        "--min-norm", type=float, default=0.5,
        help="Vectors with L2 norm below this are flagged as zero/corrupt (default: 0.5)",
    )
    parser.add_argument(
        "--max-norm", type=float, default=2.0,
        help="Vectors above this norm are also flagged (cosine-normalized embeddings should be ~1.0)",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, cfg["csv_path"])
    qdrant_url = _resolve_env(cfg["qdrant"].get("url_env"))
    collection = cfg["qdrant"]["collection"]

    # 1. CSV row count
    csv_count = _count_csv_rows(csv_path)
    logger.info("CSV has %d non-empty rows", csv_count)

    # 2. Qdrant connection + collection info
    client = QdrantClient(qdrant_url, timeout=30)
    if not client.collection_exists(collection):
        logger.error("Collection %r does not exist at %s", collection, qdrant_url)
        return 1

    info = client.count(collection)
    points_count = info.count
    logger.info("Qdrant collection %r has %d points", collection, points_count)

    # 3. Scroll all points, check vectors
    zero_norm: list[int] = []
    bad_norm: list[tuple[int, float]] = []
    seen_ids: set[int] = set()
    norms: list[float] = []

    offset = None
    page = 0
    while True:
        points, next_offset = client.scroll(
            collection_name=collection,
            limit=500,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        if not points:
            break
        page += 1
        for p in points:
            pid = int(p.id)
            seen_ids.add(pid)
            vec = p.vector
            if vec is None:
                zero_norm.append(pid)
                continue
            norm = math.sqrt(sum(v * v for v in vec))
            norms.append(norm)
            if norm < args.min_norm:
                zero_norm.append(pid)
            elif norm > args.max_norm:
                bad_norm.append((pid, norm))
        logger.info("  scrolled page %d, total seen=%d", page, len(seen_ids))
        if next_offset is None:
            break
        offset = next_offset

    # 4. Missing / extra passage_ids (assume IDs are 1..csv_count)
    expected = set(range(1, csv_count + 1))
    missing = sorted(expected - seen_ids)
    extra = sorted(seen_ids - expected)

    # 5. Report
    print("=" * 60)
    print("Jiggasha collection integrity report")
    print("=" * 60)
    print(f"  collection:    {collection}")
    print(f"  csv rows:      {csv_count}")
    print(f"  qdrant points: {points_count}")
    print(f"  scrolled:      {len(seen_ids)}")
    if norms:
        print(f"  norm avg:      {sum(norms)/len(norms):.4f}")
        print(f"  norm min/max:  {min(norms):.4f} / {max(norms):.4f}")
    print(f"  zero/low-norm: {len(zero_norm)} (first 10: {zero_norm[:10]})")
    print(f"  high-norm:     {len(bad_norm)} (first 5: {bad_norm[:5]})")
    print(f"  missing ids:   {len(missing)} (first 20: {missing[:20]})")
    print(f"  extra ids:     {len(extra)} (first 20: {extra[:20]})")
    print("=" * 60)

    ok = (
        csv_count == points_count
        and not zero_norm
        and not missing
        and not extra
    )
    if ok:
        print("VERDICT: PASS")
        return 0
    print("VERDICT: FAIL — re-ingest with `python3 process.py --force`")
    return 1


if __name__ == "__main__":
    sys.exit(main())
