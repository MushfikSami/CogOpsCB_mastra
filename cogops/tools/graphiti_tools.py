import os
import yaml
import json
import asyncio
import logging
from typing import List, Dict, Any, Optional
from ast import literal_eval
from dotenv import load_dotenv
from graphiti_core import Graphiti
from cogops.models.triton_embedder import TritonEmbedder, TritonEmbedderConfig
from cogops.models.qwen3_reranker import QwenRerankerClient
from openai import AsyncOpenAI
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.driver.neo4j_driver import Neo4jDriver
from graphiti_core.search.search_config_recipes import COMBINED_HYBRID_SEARCH_CROSS_ENCODER

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Load Environment & Config ---
load_dotenv()

def load_config(config_path: str = "configs/v2.yaml") -> Dict:
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        return {}

CONFIG = load_config()

# --- Global Singleton Client ---
# We use a global variable to hold the connection so we don't re-connect on every request.
_GRAPHITI_CLIENT: Optional[Graphiti] = None



async def get_graphiti_client() -> Graphiti:
    global _GRAPHITI_CLIENT
    if _GRAPHITI_CLIENT is None:
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

        inner_client = AsyncOpenAI(
            api_key=os.getenv("VLLM_API_KEY", "sk-placeholder"),
            base_url=os.getenv("VLLM_BASE_URL")
        )
        reranker = QwenRerankerClient(client=inner_client, config=llm_config)

        neo4j_driver = Neo4jDriver(
            uri=os.getenv("NEO4J_URI"),
            user=os.getenv("NEO4J_USER"),
            password=os.getenv("NEO4J_PASSWORD")
        )

        _GRAPHITI_CLIENT = Graphiti(
            graph_driver=neo4j_driver,
            llm_client=llm_client,
            embedder=embedder,
            cross_encoder=reranker
)
    return _GRAPHITI_CLIENT
# --- The Tool Function ---

async def graph_search(query: str) -> str:
    """
    Searches the Government Knowledge Graph for relevant facts, regulations, and procedures.
    
    Args:
        query (str): The specific search query (e.g., "passport renewal fee", "birth registration process").
    
    Returns:
        str: A formatted text summary of the findings.
    """
    client = await get_graphiti_client()
    search_config = COMBINED_HYBRID_SEARCH_CROSS_ENCODER.model_copy(deep=True)
    # Load limit from config
    search_config_params = CONFIG.get('graph_search', {})
    limit = search_config_params.get('limit', 5)
    reranker_thresh=search_config_params.get('min_score','0.9')

    
    logger.info(f"🔍 Executing Graph Search: '{query}' (Limit: {limit})")
    
    try:
        # Execute Search
        # Graphiti's search returns a list of Edge/Fact objects
        results = await client._search(
            query=query,config=search_config
        )
        
        # Initialize Markdown sections
        md_content = ""

        # Nodes Section
        md_content += "\n## Nodes\n"
        node_summaries = []
        for node, score in zip(results.nodes, results.node_reranker_scores):
            if score > reranker_thresh:
                node_summaries.append(f"**{node.name}**:{node.summary}")
        if node_summaries:
            md_content += "- " + "\n- ".join(node_summaries[:limit]) + "\n\n"
        else:
            md_content += "No relevant nodes found.\n\n"

        # Edges Section
        md_content += "## Edges\n"
        edge_facts = []
        for edge, score in zip(results.edges, results.edge_reranker_scores):
            if score > reranker_thresh:
                edge_facts.append(edge.fact)
        if edge_facts:
            md_content += "- " + "\n- ".join(edge_facts[:limit]) + "\n\n"
        else:
            md_content += "No relevant edges found.\n\n"

        # Episodes Section
        md_content += "## Passages\n"
        episode_data = []
        urls = []  # Separate list for URLs
        for episode, score in zip(results.episodes, results.episode_reranker_scores):
            if score > reranker_thresh:
                json_episode = literal_eval(episode.content)
                passage= json_episode["text"].split("Category")[0].strip()
                context=json_episode["text"].split("Category")[-1].strip()
                url = json_episode.get("url", "")  # Assuming 'url' key exists
                text=f"# passage_context:\n {context}\n\n # passage_text:\n{passage} \n # Sources:\n{url}"
                episode_data.append(text)
        if episode_data:
            md_content += "- " + "\n- ".join(episode_data[:limit]) + "\n\n"
        else:
            md_content += "No Passages found.\n\n"

        # # Separate URLs Section
        # md_content += "## SOURCES\n"
        # if urls:
        #     md_content += "- " + "\n- ".join(urls) + "\n"
        # else:
        #     md_content += "No SOURCES found.\n"
        return md_content
    except Exception as e:
        logger.error(f"❌ Error during graph search: {e}", exc_info=True)
        return f"System Error: Unable to retrieve data due to {str(e)}"

# --- Tool Schema & Mapping ---

# 1. The Schema (OpenAI Compatible)
graphiti_tools_list = [
    {
        "type": "function",
        "function": {
            "name": "graph_search",
            "description": "Search the official Bangladesh Government Knowledge Graph. Use this tool whenever the user asks about any information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The specific topic to search for (e.g., 'driving license fee', 'NID correction documents')."
                    }
                },
                "required": ["query"]
            }
        }
    }
]

# 2. The Function Map (For the Agent to execute)
available_tools_map = {
    "graph_search": graph_search
}