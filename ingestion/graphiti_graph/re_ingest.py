import os
import csv
import json
import asyncio
import logging
import argparse
import uuid
import re
from datetime import datetime, timezone
from dotenv import load_dotenv

# Graphiti Core Imports
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from graphiti_core.driver.neo4j_driver import Neo4jDriver
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.llm_client.config import LLMConfig

from openai import AsyncOpenAI
from cogops.embedders.triton import TritonEmbedder, TritonEmbedderConfig
from cogops.llm.reranker import QwenRerankerClient

# ==========================================
# LOGGING SETUP
# ==========================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("re_ingest")

# Silence noisy logs
logging.getLogger("neo4j").setLevel(logging.CRITICAL)
logging.getLogger("graphiti_core").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)


def parse_failed_indices(log_file_path: str) -> set:
    """
    Parses the log file to find row indices marked with failure indicators.
    Expects format: "X [Row 123] Failed: ..." or "❌ [Row 123] Failed: ..."
    """
    failed_indices = set()
    # Match both "X" (clean format) and "❌" (old format with special char)
    pattern = re.compile(r"(?:X|❌)\s*\[Row\s+(\d+)\]")

    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    idx = int(match.group(1))
                    failed_indices.add(idx)
    except FileNotFoundError:
        logger.error(f"Log file not found: {log_file_path}")
        return set()

    return failed_indices


async def process_single_row(graphiti: Graphiti, row: dict, original_index: int):
    """Process one row."""
    row_json = json.dumps(row, ensure_ascii=False)
    episode_name = row.get('id', str(uuid.uuid4()))

    try:
        await graphiti.add_episode(
            name=episode_name,
            episode_body=row_json,
            source=EpisodeType.json,
            source_description=f"CSV Row {original_index} (Retry)",
            reference_time=datetime.now(timezone.utc)
        )
        logger.info(f"OK [Row {original_index}] Retry Success")
        return True
    except Exception as e:
        logger.error(f"X [Row {original_index}] Retry Failed: {str(e)}")
        return False


async def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Re-ingest failed rows from a previous log.")
    parser.add_argument("csv_file", type=str, help="Path to original CSV file")
    parser.add_argument("log_file", type=str, help="Path to the log file containing failures (prev_run.log)")
    args = parser.parse_args()

    if not os.path.exists(args.csv_file):
        logger.error(f"CSV file not found: {args.csv_file}")
        return

    # 1. Parse Log for Failures
    logger.info(f"Parsing log file: {args.log_file}...")
    failed_indices = parse_failed_indices(args.log_file)

    if not failed_indices:
        logger.info("No failed rows found in the log file! Exiting.")
        return

    retry_list = []
    with open(args.csv_file, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i in failed_indices:
                retry_list.append((i, row))

    total_rows = len(retry_list)

    # 2. Initialize Drivers
    logger.info("Initializing drivers...")

    llm_config = LLMConfig(
        api_key=os.getenv("LLM_API_KEY", "sk-placeholder"),
        base_url=os.getenv("LLM_BASE_URL"),
        model=os.getenv("LLM_MODEL_NAME"),
        max_tokens=150000
    )
    llm_client = OpenAIGenericClient(config=llm_config)
    llm_client.MAX_RETRIES=5

    triton_conf = TritonEmbedderConfig(
        url=os.getenv("TRITON_URL", "localhost:6000"),
        model_name=os.getenv("TRITON_MODEL_NAME", "gemma_embedding"),
        tokenizer_path=os.getenv("TRITON_TOKENIZER", "onnx-community/embeddinggemma-300m-ONNX"),
        max_batch_size=8
    )
    embedder = TritonEmbedder(config=triton_conf)

    reranker_llm_config = LLMConfig(
        api_key=os.getenv("RERANKER_API_KEY", "sk-placeholder"),
        base_url=os.getenv("RERANKER_BASE_URL"),
        model=os.getenv("RERANKER_MODEL_NAME"),
        max_tokens=1
    )
    reranker_client = AsyncOpenAI(
        api_key=os.getenv("RERANKER_API_KEY", "sk-placeholder"),
        base_url=os.getenv("RERANKER_BASE_URL")
    )
    reranker = QwenRerankerClient(client=reranker_client, config=reranker_llm_config)

    neo4j_driver = Neo4jDriver(
        uri=os.getenv("NEO4J_URI"),
        user=os.getenv("NEO4J_USER"),
        password=os.getenv("NEO4J_PASSWORD"),
        database=os.getenv("NEO4J_DATABASE", "neo4j")
    )

    graphiti = Graphiti(
        graph_driver=neo4j_driver,
        llm_client=llm_client,
        embedder=embedder,
        cross_encoder=reranker
    )

    # 3. Synchronous Batch Loop
    batch_limit = int(os.getenv("SEMAPHORE_LIMIT", 2))
    logger.info(f"Starting Retry of {total_rows} rows. Processing {batch_limit} at a time.")

    success_count = 0

    for i in range(0, len(retry_list), batch_limit):
        current_batch = retry_list[i : i + batch_limit]
        batch_tasks = []

        for original_idx, row_data in current_batch:
            batch_tasks.append(process_single_row(graphiti, row_data, original_idx))

        results = await asyncio.gather(*batch_tasks)
        success_count += sum(1 for r in results if r)

        batch_num = i // batch_limit + 1
        logger.info(f"--- Batch {batch_num} Done ---")

    # Final Summary
    logger.info("=" * 30)
    logger.info(f"RETRY COMPLETE. Success: {success_count} / {total_rows}")
    logger.info("=" * 30)

    # Cleanup
    try:
        if hasattr(neo4j_driver, 'close'):
            await neo4j_driver.close()
        if hasattr(graphiti, 'close'):
            await graphiti.close()
    except:
        pass


if __name__ == "__main__":
    asyncio.run(main())
