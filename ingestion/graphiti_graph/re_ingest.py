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
logger = logging.getLogger("re_ingest")

# Silence noisy logs
logging.getLogger("neo4j").setLevel(logging.CRITICAL)
logging.getLogger("graphiti_core").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)

def parse_failed_indices(log_file_path: str) -> set:
    """
    Parses the log file to find row indices marked with ❌.
    Expects format: "❌ [Row 123] Failed: ..."
    """
    failed_indices = set()
    # Regex to capture the number inside [Row X] following a red cross
    # Matches: ❌ [Row 5] ...
    pattern = re.compile(r"❌\s*\[Row\s+(\d+)\]")
    
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if "❌" in line:
                    match = pattern.search(line)
                    if match:
                        idx = int(match.group(1))
                        failed_indices.add(idx)
    except FileNotFoundError:
        logger.error(f"Log file not found: {log_file_path}")
        return set()
        
    return failed_indices

async def process_single_row(graphiti: Graphiti, row: dict, original_index: int):
    """
    Process one row. 
    """
    try:
        row_json = json.dumps(row, ensure_ascii=False)
        # Use existing ID or generate new one. 
        # Ideally, use the SAME ID as the failed attempt if it was generated based on content,
        # but here we rely on the CSV 'id' column or generate a fresh UUID.
        episode_name = row.get('id', str(uuid.uuid4()))
        
        await graphiti.add_episode(
            name=episode_name,
            episode_body=row_json,
            source=EpisodeType.json,
            source_description=f"CSV Row {original_index} (Retry)",
            reference_time=datetime.now(timezone.utc)
        )
        logger.info(f"✅ [Row {original_index}] Retry Success")
        return True
    except Exception as e:
        logger.error(f"❌ [Row {original_index}] Retry Failed: {str(e)}")
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

    logger.info(f"Found {len(failed_indices)} failed rows to retry.")

    # 2. Initialize Drivers (Same as ingest.py)
    logger.info("Initializing drivers...")
    
    llm_config = LLMConfig(
        api_key=os.getenv("VLLM_API_KEY", "sk-placeholder"),
        base_url=os.getenv("VLLM_BASE_URL"),
        model=os.getenv("VLLM_MODEL_NAME"),
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

    # 3. Read CSV and Filter Rows
    retry_list = [] # List of tuples: (original_index, row_dict)
    
    with open(args.csv_file, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i in failed_indices:
                retry_list.append((i, row))

    # 4. Synchronous Batch Loop
    batch_limit = int(os.getenv("SEMAPHORE_LIMIT", 2))
    logger.info(f"Starting Retry of {len(retry_list)} rows. Processing {batch_limit} at a time.")

    success_count = 0
    
    for i in range(0, len(retry_list), batch_limit):
        # Slice the list of (index, row) tuples
        current_batch = retry_list[i : i + batch_limit]
        batch_tasks = []
        
        for original_idx, row_data in current_batch:
            # We pass the *original* index for logging consistency
            batch_tasks.append(process_single_row(graphiti, row_data, original_idx))
            
        # Wait for batch
        results = await asyncio.gather(*batch_tasks)
        
        success_count += sum(1 for r in results if r)
        
    # Final Summary
    logger.info("="*30)
    logger.info(f"RETRY COMPLETE. Success: {success_count} / {len(retry_list)}")
    logger.info("="*30)

    # Cleanup
    try:
        if hasattr(neo4j_driver, 'close'): await neo4j_driver.close()
        if hasattr(graphiti, 'close'): await graphiti.close()
    except: pass

if __name__ == "__main__":
    asyncio.run(main())