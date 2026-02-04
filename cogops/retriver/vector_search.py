import os
import yaml
import chromadb
import logging
import asyncio
from collections import defaultdict
from dotenv import load_dotenv
from typing import List, Dict, Optional, Any, Tuple

# --- Custom Module Imports ---
# Adjust these paths based on your actual project structure
from cogops.models.embGemma_embedder import GemmaTritonEmbedder, GemmaTritonEmbedderConfig
from cogops.retriver.db import SQLDatabaseManager
from cogops.utils.db_config import get_postgres_config
# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Load Environment Variables ---
load_dotenv()
POSTGRES_CONFIG = get_postgres_config()

class VectorRetriever:
    """
    Retrieves and ranks documents by first querying multiple vector collections in ChromaDB,
    fusing the results using RRF to get the top passage IDs, and then fetching the
    full passage content from a PostgreSQL database.
    """
    def __init__(self, config_path: str = "configs/v1.yaml"):
        logging.info("Initializing VectorRetriever...")
        full_config = self._load_config(config_path)
        retriever_config = full_config.get("vector_retriever")
        if not retriever_config:
            raise ValueError(f"Config file '{config_path}' is missing 'vector_retriever' section.")

        # --- Load configuration ---
        self.top_k = retriever_config.get("top_k", 10)
        self.collection_names = retriever_config.get("collections", [])
        self.max_passages_to_select = retriever_config.get("max_passages_to_select", 3)
        self.rrf_k = retriever_config.get("rrf_k", 60)
        self.passage_id_key = retriever_config.get("passage_id_meta_key", "passage_id")

        if not self.collection_names:
            raise ValueError("Config missing 'collections' key.")

        # --- Initialize clients and embedder ---
        self.chroma_client = self._connect_to_chroma()
        self.db_manager = SQLDatabaseManager(POSTGRES_CONFIG)
        self.embedder = self._initialize_embedder()

        # Get handles to all required ChromaDB collections
        self.collections = {
            name: self.chroma_client.get_collection(name=name) for name in self.collection_names
        }
        logging.info(f"VectorRetriever initialized. Will select top {self.max_passages_to_select} passages after RRF.")

    def _load_config(self, config_path: str) -> Dict:
        logging.info(f"Loading configuration from: {config_path}")
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logging.error(f"Configuration file not found at: {config_path}")
            raise

    def _connect_to_chroma(self) -> chromadb.HttpClient:
        CHROMA_HOST = os.environ.get("CHROMA_DB_HOST", "localhost")
        CHROMA_PORT = int(os.environ.get("CHROMA_DB_PORT", 8000))
        logging.info(f"Connecting to ChromaDB at {CHROMA_HOST}:{CHROMA_PORT}...")
        try:
            client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
            client.heartbeat()
            logging.info("✅ ChromaDB connection successful!")
            return client
        except Exception as e:
            logging.error(f"Failed to connect to ChromaDB: {e}", exc_info=True)
            raise

    def _initialize_embedder(self) -> GemmaTritonEmbedder:
        TRITON_URL = os.environ.get("TRITON_EMBEDDER_URL", "http://localhost:6000")
        logging.info(f"Initializing GemmaTritonEmbedder with Triton at: {TRITON_URL}")
        embedder_config = GemmaTritonEmbedderConfig(triton_url=TRITON_URL)
        return GemmaTritonEmbedder(config=embedder_config)

    async def _query_collection_async(
        self,
        collection_name: str,
        query_embedding: List[float],
        top_k: int
    ) -> List[Tuple[int, int]]:
        """
        Queries a single collection and returns a list of (passage_id, rank) tuples.
        """
        collection = self.collections[collection_name]
        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["metadatas"]
            )
            
            ranked_results = []
            if results and results['metadatas'] and results['metadatas'][0]:
                for i, meta in enumerate(results['metadatas'][0]):
                    passage_id_val = meta.get(self.passage_id_key)
                    if passage_id_val is not None:
                        try:
                            passage_id = int(passage_id_val)
                            rank = i + 1
                            ranked_results.append((passage_id, rank))
                        except (ValueError, TypeError):
                            logging.warning(f"In collection '{collection_name}', could not convert passage_id '{passage_id_val}' to int. Skipping.")
            return ranked_results
        except Exception as e:
            logging.error(f"Error querying {collection_name}: {e}")
            return []

    async def retrieve_passages(
        self,
        query: str,
        top_k_per_collection: int = None,
    ) -> List[Dict[str, Any]]:
        """
        Performs the end-to-end retrieval process.
        
        1. Embeds the query.
        2. Queries all vector collections concurrently.
        3. Fuses the results using RRF to rank passage IDs.
        4. Selects the top N passage IDs.
        5. Fetches the full passage data for these IDs from PostgreSQL.
        
        Returns:
            A list of dictionaries, each containing passage details, ordered by RRF score.
        """
        if top_k_per_collection is None:
            top_k_per_collection = self.top_k

        logging.info(f"Starting retrieval for query: '{query}'")
        
        # Step 1: Embed the query
        query_embedding = self.embedder.embed_queries([query])[0]

        # Step 2: Query all collections in parallel
        tasks = [
            self._query_collection_async(name, query_embedding, top_k_per_collection)
            for name in self.collection_names
        ]
        list_of_ranked_lists = await asyncio.gather(*tasks)

        # Step 3: Apply Reciprocal Rank Fusion
        fused_scores = defaultdict(float)
        for ranked_list in list_of_ranked_lists:
            for passage_id, rank in ranked_list:
                fused_scores[passage_id] += 1.0 / (self.rrf_k + rank)
        
        if not fused_scores:
            logging.warning("No passages found after querying all vector collections.")
            return []

        # Step 4: Sort by RRF score and select the top passage IDs
        sorted_passage_ids = sorted(
            fused_scores.keys(),
            key=lambda pid: fused_scores[pid],
            reverse=True
        )
        top_passage_ids = sorted_passage_ids[:self.max_passages_to_select]
        logging.info(f"RRF found {len(fused_scores)} unique passages. Selecting top {len(top_passage_ids)} IDs for retrieval.")

        if not top_passage_ids:
            return []

        # Step 5: Fetch full passage data from PostgreSQL
        try:
            logging.info(f"Fetching full data for IDs from PostgreSQL: {top_passage_ids}")
            passages_df = self.db_manager.select_passages_by_ids(top_passage_ids)
            
            if passages_df.empty:
                logging.warning(f"PostgreSQL query returned no data for IDs: {top_passage_ids}")
                return []

            # Convert DataFrame to a dictionary for efficient, ordered lookup
            passage_map = {row['passage_id']: row.to_dict() for index, row in passages_df.iterrows()}

            # Re-order the results from the database to match the RRF ranking
            final_ordered_passages = []
            for pid in top_passage_ids:
                if pid in passage_map:
                    final_ordered_passages.append(passage_map[pid])
            
            return final_ordered_passages

        except Exception as e:
            logging.error(f"Failed to retrieve passages from PostgreSQL. Error: {e}", exc_info=True)
            return []

    def close(self):
        """Cleanly closes any open connections."""
        if self.embedder:
            self.embedder.close()
            logging.info("Embedder connection closed.")

async def main():
    """Main function to test the VectorRetriever."""
    retriever = None
    try:
        retriever = VectorRetriever(config_path="configs/v1.yaml")
        user_query = "আমার এন আই ডি হারায়ে গেছে রাস্তায়"
        
        print(f"\n--- Testing retrieval for query: '{user_query}' ---")
        passages = await retriever.retrieve_passages(user_query)
        
        if passages:
            print(f"\nRetrieved {len(passages)} passages from PostgreSQL, sorted by relevance:")
            for i, passage in enumerate(passages):
                print("-" * 20)
                print(f"Rank {i+1}:")
                print(f"  Passage ID: {passage.get('passage_id')}")
                print(f"  URL: {passage.get('url')}")
                print(f"  Date: {passage.get('date')}")
                print(f"  Text: '{str(passage.get('text'))[:150]}...'")
        else:
            print("\nNo passages were retrieved for the query.")
            
    except Exception as e:
        logging.error(f"An error occurred in the main execution: {e}", exc_info=True)
    finally:
        if retriever:
            retriever.close()
        logging.info("\n--- Script Finished ---")


if __name__ == "__main__":
    asyncio.run(main())