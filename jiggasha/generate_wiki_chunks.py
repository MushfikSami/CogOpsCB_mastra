#!/usr/bin/env python3
"""
Generate per-row JSON chunk files from data.csv into wiki_chunks/.

Uses Hugging Face AutoTokenizer for both embedder and LLM token counts.
  • embedder_token_count  → Qwen/Qwen3-Embedding-8B
  • llm_token_count       → cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit

Filenames are numeric passage IDs: 00000.json … NNNNN.json.

Schema fields:
  page_title, section, subsection, chunk_index, total_chunks,
  embedder_token_count, llm_token_count, revision_id, chunk_type,
  source_url, text, last_edited

source_url is formatted like the Bengali source descriptor shown in
chat responses: "বিভাগ: X | উপ-বিভাগ: Y | সেবা: Z | বিষয়: W"
(only non-empty fields are included).
"""

import csv
import json
import os
from datetime import datetime, timezone

from transformers import AutoTokenizer

CSV_PATH = "data.csv"
OUT_DIR = "wiki_chunks"
CHUNK_TYPE = "govt_service"
EMBEDDER_TOKENIZER = "Qwen/Qwen3-Embedding-8B"
LLM_TOKENIZER = "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit"

_FALLBACK_RULES = [
    (lambda r: r["service"].startswith("মৃত্যু নিবন্ধন"),
     "মৃত্যু নিবন্ধন ও সনদ", "মৃত্যু নিবন্ধন ও সনদপ্রাপ্তি"),
    (lambda r: not r["category"] and "ই-পাসপোর্ট" in r["text"],
     "পাসপোর্ট", "পাসপোর্ট"),
    (lambda r: r["service"].startswith("নিঃসন্তান প্রত্যয়ন"),
     "জরুরি প্রত্যয়ন ও সনদ", "জরুরি প্রত্যয়ন"),
    (lambda r: r["service"].startswith("ড্রাইভিং লাইসেন্স প্রত্যয়ন"),
     "জরুরি প্রত্যয়ন ও সনদ", "জরুরি প্রত্যয়ন"),
    (lambda r: r["service"].startswith("অনূদিত সনদ প্রত্যয়ন"),
     "জরুরি প্রত্যয়ন ও সনদ", "জরুরি প্রত্যয়ন"),
    (lambda r: r["service"].startswith("সাধারণ ডায়েরি প্রত্যয়ন"),
     "জরুরি প্রত্যয়ন ও সনদ", "জরুরি প্রত্যয়ন"),
    (lambda r: r["service"].startswith("পাসপোর্ট কপি প্রত্যয়ন"),
     "জরুরি প্রত্যয়ন ও সনদ", "জরুরি প্রত্যয়ন"),
    (lambda r: r["service"].startswith("পাসপোর্ট প্রত্যয়ন"),
     "জরুরি প্রত্যয়ন ও সনদ", "জরুরি প্রত্যয়ন"),
    (lambda r: r["service"].startswith("মাসিক/বাৎসরিক আয়ের হিসাব প্রত্যয়ন"),
     "জরুরি প্রত্যয়ন ও সনদ", "জরুরি প্রত্যয়ন"),
    (lambda r: r["service"].startswith("উত্তরাধিকার/সাকসেশন সনদ"),
     "জরুরি প্রত্যয়ন ও সনদ", "জরুরি প্রত্যয়ন"),
    (lambda r: r["service"].startswith("স্বর্ণ (কারিগরি)"),
     "ব্যবসায় সংক্রান্ত সেবা", "লাইসেন্স নবায়ন"),
]


def parse_meta(meta_str: str) -> dict:
    result = {"category": "", "sub_category": "", "service": "", "topic": ""}
    if not meta_str:
        return result
    for part in meta_str.split("|"):
        part = part.strip()
        if ":" in part:
            key, val = part.split(":", 1)
            key = key.strip()
            val = val.strip()
            if key in result:
                result[key] = val
    return result


def apply_fallback(row: dict) -> dict:
    if row["category"] and row["sub_category"]:
        return row
    for predicate, cat, sub in _FALLBACK_RULES:
        if predicate(row):
            row["category"] = cat
            row["sub_category"] = sub
            return row
    if not row["category"]:
        row["category"] = "অনির্দিষ্ট বিভাগ"
    if not row["sub_category"]:
        row["sub_category"] = "অনির্দিষ্ট উপ-বিভাগ"
    return row


def build_source_url(category: str, sub_category: str, service: str, topic: str) -> str:
    """Format source_url like the Bengali descriptor in response sources."""
    parts = []
    if category:
        parts.append(f"বিভাগ: {category}")
    if sub_category:
        parts.append(f"উপ-বিভাগ: {sub_category}")
    if service:
        parts.append(f"সেবা: {service}")
    if topic:
        parts.append(f"বিষয়: {topic}")
    return " | ".join(parts) if parts else ""


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading tokenizers …")
    embed_tok = AutoTokenizer.from_pretrained(EMBEDDER_TOKENIZER, trust_remote_code=True)
    llm_tok = AutoTokenizer.from_pretrained(LLM_TOKENIZER, trust_remote_code=True)
    print("Tokenizers ready.")

    rows = []
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            meta = parse_meta(row.get("meta", ""))
            rows.append(apply_fallback({
                "idx": idx,
                "text": (row.get("text") or "").strip(),
                "category": meta["category"],
                "sub_category": meta["sub_category"],
                "service": meta["service"],
                "topic": meta["topic"],
            }))

    # Compute total_chunks per (category, sub_category)
    group_totals = {}
    for r in rows:
        key = (r["category"], r["sub_category"])
        group_totals[key] = group_totals.get(key, 0) + 1

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    group_indices = {}
    written = 0

    for r in rows:
        page_title = r["category"]
        section = r["sub_category"]
        key = (page_title, section, CHUNK_TYPE)

        chunk_index = group_indices.get(key, 0)
        group_indices[key] = chunk_index + 1
        total_chunks = group_totals[(page_title, section)]

        text = r["text"]
        embedder_token_count = len(embed_tok.encode(text, add_special_tokens=False))
        llm_token_count = len(llm_tok.encode(text, add_special_tokens=False))

        payload = {
            "page_title": page_title,
            "section": section,
            "subsection": r["service"],
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "embedder_token_count": embedder_token_count,
            "llm_token_count": llm_token_count,
            "revision_id": f"govt-{r['idx']:05d}",
            "chunk_type": CHUNK_TYPE,
            "source_url": build_source_url(
                r["category"], r["sub_category"], r["service"], r["topic"]
            ),
            "text": text,
            "last_edited": today,
        }

        fname = f"{r['idx']:05d}.json"
        fpath = os.path.join(OUT_DIR, fname)
        with open(fpath, "w", encoding="utf-8") as out:
            json.dump(payload, out, ensure_ascii=False, indent=2)
        written += 1

    print(f"Wrote {written} JSON files to {OUT_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
