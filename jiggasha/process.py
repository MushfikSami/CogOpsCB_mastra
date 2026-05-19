#!/usr/bin/env python3
"""Ingest jiggasha/data.csv into Qdrant with embeddings.

Usage:
    python3 process.py                    # ingest all rows
    python3 process.py --force            # drop & recreate collection
    python3 process.py --limit 100        # ingest first 100 rows
    python3 process.py --dry-run          # show rows without embedding
"""
import argparse
import csv
import logging
import os
import sys
import time

import yaml

# Ensure current dir is on path so embedder/reranker are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from embedder import Embedder
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _load_config(path: str = "config.yml") -> dict:
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


def _parse_meta(raw: str) -> dict[str, str]:
    meta: dict[str, str] = {}
    for part in raw.split("|"):
        part = part.strip()
        if ":" not in part:
            continue
        k, v = part.split(":", 1)
        meta[k.strip()] = v.strip()
    return meta


def _create_collection(client: QdrantClient, cfg: dict) -> None:
    name = cfg["qdrant"]["collection"]
    dist = getattr(Distance, cfg["qdrant"].get("distance", "Cosine").upper(), Distance.COSINE)
    dim = cfg["embedder"]["dimension"]
    if client.collection_exists(name):
        logger.info(f"Deleting existing collection '{name}'")
        client.delete_collection(name)
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=dim, distance=dist),
    )
    for field, schema_type in {
        "passage_id": PayloadSchemaType.INTEGER,
        "category": PayloadSchemaType.KEYWORD,
        "sub_category": PayloadSchemaType.KEYWORD,
        "service": PayloadSchemaType.KEYWORD,
        "topic": PayloadSchemaType.KEYWORD,
        "text": PayloadSchemaType.TEXT,
    }.items():
        client.create_payload_index(
            collection_name=name, field_name=field, field_schema=schema_type,
        )
    logger.info(f"Created Qdrant collection '{name}' (dim={dim}, {dist})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Process: load CSV -> embed -> Qdrant")
    parser.add_argument("--force", action="store_true", help="Drop & recreate collection")
    parser.add_argument("--limit", type=int, default=0, help="Process only first N rows")
    parser.add_argument("--dry-run", action="store_true", help="Show rows without embedding")
    parser.add_argument("--config", default="config.yml", help="Config file path")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, cfg["csv_path"])

    # Step 1: Read CSV
    logger.info(f"Step 1/3: Reading {csv_path}...")
    rows: list[dict] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if args.limit and i >= args.limit:
                break
            meta = _parse_meta(row.get("meta", ""))
            rows.append({
                "passage_id": i + 1,
                "text": row.get("text", "").strip(),
                "meta": meta,
            })
    rows = [r for r in rows if r["text"]]
    logger.info(f"  {len(rows)} valid rows")

    if args.dry_run:
        logger.info("DRY RUN — skipping embedding and upsert")
        for r in rows[:5]:
            print(f"  [{r['passage_id']}] {r['text'][:80]}...")
        print(f"  ... ({len(rows)} total)")
        return

    # Step 2: Embed
    embedder_cfg = cfg["embedder"]
    url = _resolve_env(embedder_cfg.get("url_env"))
    key = _resolve_env(embedder_cfg.get("api_key_env"))
    model = _resolve_env(embedder_cfg.get("model_env")) or embedder_cfg.get("model", "")
    logger.info("Step 2/3: Embedding passages...")
    texts = [r["text"] for r in rows]
    embedder = Embedder(url=url, api_key=key, model=model, batch_size=embedder_cfg.get("batch_size", 64))
    t0 = time.time()
    embeddings = embedder.embed_batch(texts)
    elapsed = time.time() - t0
    logger.info(f"  Embedded {len(embeddings)} passages in {elapsed:.1f}s")

    # Step 3: Upsert to Qdrant
    logger.info("Step 3/3: Upserting to Qdrant...")
    client = QdrantClient(
        url=_resolve_env(cfg["qdrant"].get("url_env")),
        timeout=cfg["qdrant"].get("timeout", 30),
    )
    if args.force or not client.collection_exists(cfg["qdrant"]["collection"]):
        _create_collection(client, cfg)

    BATCH = 100
    name = cfg["qdrant"]["collection"]
    total = 0
    batch: list[PointStruct] = []
    for row, emb in zip(rows, embeddings):
        batch.append(PointStruct(
            id=row["passage_id"],
            vector=emb,
            payload={"passage_id": row["passage_id"], "text": row["text"], **row["meta"]},
        ))
        if len(batch) >= BATCH:
            client.upsert(collection_name=name, points=batch, wait=False)
            total += len(batch)
            batch = []
            logger.info(f"  upserted {total:,} / {len(rows):,}")
    if batch:
        client.upsert(collection_name=name, points=batch, wait=False)
        total += len(batch)

    count = client.count(name).count
    logger.info(f"Done! Qdrant now has {count:,} points ({total:,} inserted)")


if __name__ == "__main__":
    main()
