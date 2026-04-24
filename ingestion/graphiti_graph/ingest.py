import os
import csv
import json
import asyncio
import logging
import argparse
import uuid
from datetime import datetime, timezone
from dotenv import load_dotenv

# Graphiti Core Imports
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from graphiti_core.driver.neo4j_driver import Neo4jDriver
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.llm_client.config import LLMConfig

from openai import AsyncOpenAI
from cogops.models.embedder import TritonEmbedder, TritonEmbedderConfig
from cogops.models.reranker import QwenRerankerClient

# ==========================================
# LOGGING SETUP
# ==========================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ingest")

# Silence noisy logs
logging.getLogger("neo4j").setLevel(logging.CRITICAL)
logging.getLogger("graphiti_core").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)


async def process_single_row(graphiti: Graphiti, row: dict, index: int):
    """Process one row."""
    row_json = json.dumps(row, ensure_ascii=False)
    episode_name = row.get('id', str(uuid.uuid4()))

    try:
        await graphiti.add_episode(
            name=episode_name,
            episode_body=row_json,
            source=EpisodeType.json,
            source_description=f"CSV Row {index}",
            reference_time=datetime.now(timezone.utc)
        )
        logger.info(f"OK [Row {index}] Success")
        return True
    except Exception as e:
        logger.error(f"X [Row {index}] Failed: {str(e)}")
        return False


async def main():
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file", type=str)
    args = parser.parse_args()

    if not os.path.exists(args.csv_file):
        logger.error(f"File not found: {args.csv_file}")
        return

    # --- Setup ---
    logger.info("Initializing drivers...")

    llm_config = LLMConfig(
        api_key=os.getenv("LLM_API_KEY", "sk-placeholder"),
        base_url=os.getenv("LLM_BASE_URL"),
        model=os.getenv("LLM_MODEL_NAME"),
        max_tokens=256000
    )
    llm_client = OpenAIGenericClient(config=llm_config)

    triton_conf = TritonEmbedderConfig(
        url=os.getenv("TRITON_URL", "localhost:6000"),
        model_name=os.getenv("TRITON_MODEL_NAME", "gemma_embedding"),
        tokenizer_path=os.getenv("TRITON_TOKENIZER", "onnx-community/embeddinggemma-300m-ONNX"),
        max_batch_size=8
    )
    embedder = TritonEmbedder(config=triton_conf)

    # Reranker uses the RERANKER_* endpoint
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

    # --- Read CSV ---
    with open(args.csv_file, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    total_rows = len(rows)

    batch_limit = int(os.getenv("SEMAPHORE_LIMIT", 2))
    logger.info(f"Starting ingestion of {total_rows} rows. Processing {batch_limit} at a time.")

    # --- SYNCHRONOUS BATCH LOOP ---
    success_count = 0

    for i in range(0, len(rows), batch_limit):
        current_batch = rows[i : i + batch_limit]
        batch_tasks = []

        for j, row in enumerate(current_batch):
            global_index = i + j
            batch_tasks.append(process_single_row(graphiti, row, global_index))

        results = await asyncio.gather(*batch_tasks)
        success_count += sum(1 for r in results if r)

        batch_num = i // batch_limit + 1
        logger.info(f"--- Batch {batch_num} Done ---")

    logger.info(f"Ingestion Complete. Success: {success_count} / {total_rows}")

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
