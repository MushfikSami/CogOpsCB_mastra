import os
import yaml
import argparse
import json
from datetime import datetime
from loguru import logger
import sys
from dotenv import load_dotenv
from tqdm import tqdm
import chromadb

# --- Custom Module Imports ---
# Ensure these paths are correct for your project structure
from cogops.retriver.db import SQLDatabaseManager
from cogops.utils.db_config import get_postgres_config
from cogops.models.embGemma_embedder import GemmaTritonEmbedder, GemmaTritonEmbedderConfig

# Load environment variables from .env file
load_dotenv()

# --- Infrastructure Configuration (from Environment Variables) ---
TRITON_URL = os.environ.get("TRITON_EMBEDDER_URL", "http://localhost:6000")
CHROMA_HOST = os.environ.get("CHROMA_DB_HOST", "localhost")
CHROMA_PORT = int(os.environ.get("CHROMA_DB_PORT", 8443))
POSTGRES_CONFIG = get_postgres_config()

def load_agent_config(config_path: str) -> dict:
    """Loads the agent's YAML configuration file."""
    logger.info(f"Loading agent configuration from: {config_path}")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info("Agent configuration loaded successfully.")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found at: {config_path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading YAML configuration: {e}")
        sys.exit(1)

def load_json_files(json_folder_path: str) -> list:
    """Loads all JSON files from a directory."""
    if not os.path.isdir(json_folder_path):
        logger.error(f"JSON folder not found at: {json_folder_path}")
        sys.exit(1)
        
    all_json_data = []
    logger.info(f"Loading JSON files from '{json_folder_path}'...")
    file_list = [f for f in os.listdir(json_folder_path) if f.endswith('.json')]
    
    for filename in tqdm(file_list, desc="Reading JSON files"):
        filepath = os.path.join(json_folder_path, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                all_json_data.append(json.load(f))
        except json.JSONDecodeError:
            logger.warning(f"Could not parse JSON from {filename}. Skipping.")
        except Exception as e:
            logger.error(f"Failed to read {filename}: {e}")
            
    logger.info(f"Successfully loaded {len(all_json_data)} JSON files.")
    return all_json_data

def ingest_to_postgres(db_manager: SQLDatabaseManager, all_json_data: list):
    """Prepares and upserts structured data into PostgreSQL."""
    logger.info("--- Starting PostgreSQL Ingestion ---")
    
    postgres_records = []
    # These are the columns for the 'passages' table
    postgres_columns = ['passage_id', 'category', 'sub_category', 'service', 'topic', 'text', 'url', 'date']

    for data in all_json_data:
        # Create a record with only the keys relevant to PostgreSQL
        record = {key: data.get(key) for key in postgres_columns}
        # Ensure date is in the correct format if it's a string
        if isinstance(record['date'], str):
            record['date'] = datetime.fromisoformat(record['date']).date()
        postgres_records.append(record)

    if not postgres_records:
        logger.warning("No records to insert into PostgreSQL. Skipping.")
        return

    # All columns except the primary key should be updated on conflict
    update_columns = [col for col in postgres_columns if col != 'passage_id']
    
    logger.info(f"Upserting {len(postgres_records)} records into the 'passages' table...")
    db_manager.upsert_passages(insert_data=postgres_records, update_columns=update_columns)
    logger.info("✅ PostgreSQL ingestion complete.")


def ingest_to_chroma(chroma_client: chromadb.Client, embedder: GemmaTritonEmbedder, config: dict, all_json_data: list):
    """Prepares, embeds, and ingests data into multiple ChromaDB collections."""
    logger.info("--- Starting ChromaDB Ingestion ---")

    # Mapping from config collection name to the key in our JSON files
    collection_key_map = {
        "PropositionsDB": "propositions",
        "SummariesDB": "summaries",
        "QuestionsDB": "question_patterns"
    }

    collections_to_process = config['vector_retriever']['collections']
    passage_id_meta_key = config['vector_retriever']['passage_id_meta_key']
    passage_embedding_function = embedder.as_chroma_passage_embedder()

    for collection_name in collections_to_process:
        json_key = collection_key_map.get(collection_name)
        if not json_key:
            logger.warning(f"No mapping found for collection '{collection_name}'. Skipping.")
            continue

        logger.info(f"\nProcessing collection: '{collection_name}'")

        # 1. Clear the collection for a fresh start
        try:
            chroma_client.delete_collection(name=collection_name)
            logger.info(f"Successfully deleted existing collection '{collection_name}'.")
        except Exception:
            logger.info(f"Collection '{collection_name}' does not exist. A new one will be created.")

        # 2. Create the collection with our custom embedder
        collection = chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=passage_embedding_function
        )
        
        # 3. Prepare data from all JSON files for this collection
        documents, metadatas, ids = [], [], []
        for data in all_json_data:
            passage_id = data['passage_id']
            doc_list = data.get(json_key, [])
            for i, doc_text in enumerate(doc_list):
                documents.append(doc_text)
                metadatas.append({passage_id_meta_key: passage_id})
                ids.append(f"{collection_name}_{passage_id}_{i}")

        if not documents:
            logger.warning(f"No documents found for collection '{collection_name}'. Skipping ingestion.")
            continue

        # 4. Ingest data in batches
        batch_size = 8  # A sensible default batch size
        for i in tqdm(range(0, len(documents), batch_size), desc=f"Ingesting to {collection_name}"):
            collection.add(
                documents=documents[i:i + batch_size],
                metadatas=metadatas[i:i + batch_size],
                ids=ids[i:i + batch_size]
            )
        
        logger.info(f"✅ Successfully ingested {collection.count()} documents into '{collection_name}'.")


def main():
    """Main function to orchestrate the entire ingestion pipeline."""
    parser = argparse.ArgumentParser(description="Ingest processed data into PostgreSQL and ChromaDB.")
    parser.add_argument("--config", type=str, required=True, help="Path to the agent's v1.yaml file.")
    parser.add_argument("--json_folder", type=str, required=True, help="Path to the folder with processed JSON files.")
    args = parser.parse_args()

    # --- Initialization ---
    config = load_agent_config(args.config)
    all_json_data = load_json_files(args.json_folder)

    if not all_json_data:
        logger.error("No JSON data loaded. Exiting.")
        sys.exit(1)

    db_manager = SQLDatabaseManager(POSTGRES_CONFIG)
    
    logger.info(f"Initializing embedder with Triton at: {TRITON_URL}")
    embedder_config = GemmaTritonEmbedderConfig(triton_url=TRITON_URL)
    embedder = GemmaTritonEmbedder(config=embedder_config)

    logger.info(f"Connecting to ChromaDB at {CHROMA_HOST}:{CHROMA_PORT}")
    chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

    try:
        # --- Run Ingestion Pipelines ---
        ingest_to_postgres(db_manager, all_json_data)
        ingest_to_chroma(chroma_client, embedder, config, all_json_data)
    except Exception as e:
        logger.error(f"A critical error occurred during the ingestion process: {e}", exc_info=True)
    finally:
        embedder.close()
        logger.info("\n--- Data Ingestion Script Finished ---")

if __name__ == "__main__":
    main()