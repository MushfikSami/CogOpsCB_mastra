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

# Local Imports
try:
    from triton_embedder import TritonEmbedder, TritonEmbedderConfig
    from qwen3_reranker import QwenRerankerClient
except ImportError as e:
    raise ImportError(f"Could not import custom modules: {e}")

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
    """
    Process one row. 
    """
    try:
        row_json = json.dumps(row, ensure_ascii=False)
        episode_name = row.get('id', str(uuid.uuid4()))
        
        await graphiti.add_episode(
            name=episode_name,
            episode_body=row_json,
            source=EpisodeType.json,
            source_description=f"CSV Row {index}",
            reference_time=datetime.now(timezone.utc)
        )
        logger.info(f"✅ [Row {index}] Success")
        return True
    except Exception as e:
        logger.error(f"❌ [Row {index}] Failed: {str(e)}")
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
        api_key=os.getenv("VLLM_API_KEY", "sk-placeholder"),
        base_url=os.getenv("VLLM_BASE_URL"),
        model=os.getenv("VLLM_MODEL_NAME"),
        max_tokens=150000
    )
    llm_client = OpenAIGenericClient(config=llm_config)

    # Embedder Config (Now respects batch size 8)
    triton_conf = TritonEmbedderConfig(
        url=os.getenv("TRITON_URL", "localhost:6000"),
        model_name=os.getenv("TRITON_MODEL_NAME", "gemma_embedding"),
        tokenizer_path=os.getenv("TRITON_TOKENIZER", "onnx-community/embeddinggemma-300m-ONNX"),
        max_batch_size=8 # Ensure this matches your new TritonEmbedder
    )
    embedder = TritonEmbedder(config=triton_conf)
    
    reranker = QwenRerankerClient(client=llm_client, config=llm_config)

    neo4j_driver = Neo4jDriver(
        uri=os.getenv("NEO4J_URI"),
        user=os.getenv("NEO4J_USER"),
        password=os.getenv("NEO4J_PASSWORD")
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
    
    # Use ENV for batch size, default to 2
    batch_limit = int(os.getenv("SEMAPHORE_LIMIT", 2))
    logger.info(f"Starting ingestion of {len(rows)} rows. Processing {batch_limit} at a time.")

    # --- SYNCHRONOUS BATCH LOOP ---
    # We slice the list into chunks of 'batch_limit' (e.g., 2)
    # and await them completely before moving to the next chunk.
    success_count = 0
    
    for i in range(0, len(rows), batch_limit):
        # 1. Get the next 2 rows
        current_batch = rows[i : i + batch_limit]
        batch_tasks = []
        
        # 2. Create tasks for just these 2
        for j, row in enumerate(current_batch):
            global_index = i + j
            batch_tasks.append(process_single_row(graphiti, row, global_index))
            
        # 3. Wait for these 2 to finish completely
        results = await asyncio.gather(*batch_tasks)
        
        success_count += sum(1 for r in results if r)
        logger.info(f"--- Batch {i//batch_limit + 1} Done ---")

    logger.info(f"Ingestion Complete. Success: {success_count} / {len(rows)}")

    # Cleanup
    try:
        if hasattr(neo4j_driver, 'close'): await neo4j_driver.close()
        if hasattr(graphiti, 'close'): await graphiti.close()
    except: pass

if __name__ == "__main__":
    asyncio.run(main())