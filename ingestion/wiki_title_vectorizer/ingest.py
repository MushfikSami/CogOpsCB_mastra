import os
import yaml
import argparse
import pandas as pd
import chromadb
import logging
from dotenv import load_dotenv
from tqdm import tqdm  # <-- Import tqdm for the progress bar
import time
# Import your custom embedder module
from embedder import GemmaTritonEmbedder, GemmaTritonEmbedderConfig
load_dotenv()
# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Infrastructure Configuration (from Environment Variables or Defaults) ---
TRITON_URL = os.environ.get("TRITON_EMBEDDER_URL", "http://localhost:6000")
CHROMA_HOST = os.environ.get("CHROMA_DB_HOST", "localhost")
CHROMA_PORT = int(os.environ.get("CHROMA_DB_PORT", 8443)) # Default Chroma port is 8443

print(TRITON_URL)

def load_data_config(config_path: str) -> dict:
    """Loads the YAML data configuration file."""
    logging.info(f"Loading data configuration from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    logging.info("Data configuration loaded successfully.")
    return config

def main(config_path: str):
    """Main function to run the data ingestion process."""
    load_dotenv()
    data_config = load_data_config(config_path)
    collection_name = data_config['collection_name']

    # 1. Initialize our custom embedder using infrastructure config
    logging.info(f"Initializing embedder with Triton at: {TRITON_URL}")
    embedder_config = GemmaTritonEmbedderConfig(triton_url=TRITON_URL)
    embedder = GemmaTritonEmbedder(config=embedder_config)

    try:
        # 2. Connect to ChromaDB server
        logging.info(f"Connecting to ChromaDB at {CHROMA_HOST}:{CHROMA_PORT}")
        chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        heartbeat = chroma_client.heartbeat()
        logging.info(f"✅ Connection successful! Server heartbeat: {heartbeat}ns")

        # 3. Clear the collection if it already exists
        try:
            logging.info(f"Checking for existing collection '{collection_name}'...")
            chroma_client.delete_collection(name=collection_name)
            logging.info(f"Successfully deleted existing collection '{collection_name}'.")
        except ValueError:
            logging.info(f"Collection '{collection_name}' does not exist. A new one will be created.")
        except Exception as e:
            logging.error(f"An error occurred while trying to delete collection: {e}")
            #raise

        # 4. Get or create the collection
        passage_embedding_function = embedder.as_chroma_passage_embedder()
        collection = chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=passage_embedding_function
        )
        logging.info(f"Collection '{collection_name}' is ready.")

        # 5. Read and Process CSV Data
        logging.info(f"Reading CSV file from: {data_config['csv_file_path']}")
        df = pd.read_csv(data_config['csv_file_path'])

        # Validate required columns
        content_col = data_config['content_column']
        metadata_cols = data_config['metadata_columns']
        # ... (validation logic can be added here)

        # Process data in batches
        ingestion_batch_size = data_config.get('batch_size', 64)
        total_rows = len(df)
        
        # <-- Wrap the loop with tqdm for a progress bar
        for i in tqdm(range(0, total_rows, ingestion_batch_size), desc="Ingesting Batches"):
            batch_df = df.iloc[i:i + ingestion_batch_size]
            
            # Prepare data for ChromaDB based on config mapping
            documents = batch_df[content_col].astype(str).tolist()
            metadatas = batch_df[metadata_cols].to_dict('records') if metadata_cols else [None] * len(batch_df)
            ids = [f"row_{j}" for j in range(i, i + len(batch_df))]
            
            # Add the batch to the collection. ChromaDB will call the embedder.
            collection.add(documents=documents, metadatas=metadatas, ids=ids)
            
            #time.sleep(5)

        logging.info("✅ All batches processed and added successfully!")
        logging.info(f"Collection now contains {collection.count()} documents.")

    except FileNotFoundError:
        logging.error(f"Error: The file '{data_config['csv_file_path']}' was not found.")
    except Exception as e:
        logging.error(f"An error occurred during the ingestion process: {e}", exc_info=True)
    finally:
        embedder.close()
        logging.info("--- Data Insertion Script Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest data into ChromaDB from a CSV file.")
    parser.add_argument("--config", type=str, default="chroma_ingestion_config.yaml", help="Path to data config YAML file.")
    args = parser.parse_args()
    main(args.config)