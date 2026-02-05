import os
import yaml
import json
import logging
from typing import AsyncGenerator, Dict, Any, List, Tuple

from dotenv import load_dotenv

# --- Core Component Imports ---
from cogops.models.qwen3async_llm_CoT import AsyncLLMService
from cogops.prompts.graphiti_prompt import get_graph_prompt
from cogops.tools.graphiti_tools import graphiti_tools_list, available_tools_map
from cogops.utils.token_manager import TokenManager  # Reusing your existing utility

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

class GraphitiAgent:
    """
    The Orchestrator for the GovOps Agent.
    Connects the Qwen Reasoning Engine, Graphiti Tools, and Constitutional Prompt.
    """
    def __init__(self, config_path: str = "configs/v2.yaml"):
        logger.info("🤖 Initializing Graphiti GovAgent...")
        self.config = self._load_config(config_path)
        
        # --- Identity & Config ---
        self.agent_name = self.config.get('agent_name', 'Gov Assistant')
        self.agent_story = self.config.get('agent_story', '')
        self.history_window = self.config['conversation']['history_window']
        self.llm_call_params = self.config.get('llm_call_parameters', {})
        self.response_templates = self.config['response_templates']

        # --- Initialize LLM Service ---
        self.llm_service = self._initialize_llm()

        # --- Initialize Token Manager ---
        # Used to calculate how much history we can fit in the context
        tm_config = self.config['token_management']
        self.token_manager = TokenManager(
            model_name=os.getenv(tm_config['tokenizer_model_env'], "Qwen/Qwen2.5-32B-Instruct"),
            reservation_tokens=tm_config['system_prompt_reservation'],
            history_budget=tm_config['history_budget']
        )

        # --- Tool Setup ---
        self.tools_schema = graphiti_tools_list
        self.tool_map = available_tools_map
        # Pre-format tools description for the system prompt
        self.tools_desc_str = json.dumps(self.tools_schema, indent=2, ensure_ascii=False)

        # --- Memory ---
        self.history: List[Tuple[str, str]] = []
        
        logger.info("✅ Graphiti GovAgent Ready.")

    def _load_config(self, path: str) -> Dict:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Config load failed: {e}")
            raise

    def _initialize_llm(self) -> AsyncLLMService:
        svc_conf = self.config['llm_service']
        return AsyncLLMService(
            api_key=os.getenv(svc_conf['api_key_env']),
            model=os.getenv(svc_conf['model_name_env']),
            base_url=os.getenv(svc_conf['base_url_env']),
            max_context_tokens=svc_conf['max_context_tokens']
        )

    def _format_history_for_prompt(self) -> str:
        """
        Formats history list into a text block, truncating if necessary via TokenManager.
        """
        if not self.history:
            return "No previous conversation."
        
        # We rely on the existing TokenManager logic to truncate based on the budget
        # defined in v2.yaml (e.g., 30% of context).
        return self.token_manager._truncate_history(
            self.history, 
            max_tokens=self.llm_service.max_context_tokens # Pass dynamic limit if needed
        )

    async def process_query(self, user_query: str) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Main Pipeline:
        1. Build Prompt -> 2. Stream LLM (with CoT Hiding) -> 3. Tool Loop -> 4. Final Answer
        """
        logger.info(f"📨 Processing Query: {user_query}")
        
        try:
            # 1. Prepare Context
            history_str = self._format_history_for_prompt()
            
            # 2. Build The Constitutional Prompt
            system_prompt = get_graph_prompt(
                agent_name=self.agent_name,
                agent_story=self.agent_story,
                tools_description=self.tools_desc_str,
                conversation_history=history_str,
                user_query=user_query
            )

            # 3. Construct Messages Payload
            # Note: We put the massive SOP in 'system'. 
            # The 'user' message is the specific query triggering this turn.
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ]

            # 4. Stream Response via the Reasoning Engine
            full_answer_accumulator = []
            
            # This calls the CoT-aware LLM service
            stream_gen = self.llm_service.stream_with_tool_calls(
                messages=messages,
                tools=self.tools_schema,
                available_tools=self.tool_map,
                debug_mode=True, # Set to True to see CoT logs in console (optional)
                **self.llm_call_params
            )

            async for event in stream_gen:
                # Pass through the events (answer chunks, debug logs, errors)
                # The API layer will decide what to forward to the frontend.
                yield event
                
                if event["type"] == "answer_chunk":
                    full_answer_accumulator.append(event["content"])

            # 5. Update Memory
            final_response = "".join(full_answer_accumulator).strip()
            if final_response:
                self.history.append((user_query, final_answer_str := final_response))
                
                # Simple Window Truncation (TokenManager handles the prompt size, 
                # this just keeps the list from growing infinitely in RAM)
                if len(self.history) > self.history_window:
                    self.history.pop(0)

        except Exception as e:
            logger.error(f"🔥 Critical Error in Agent Pipeline: {e}", exc_info=True)
            yield {"type": "error", "content": self.response_templates['error_fallback']}

    def clear_session(self):
        """Resets the memory."""
        self.history = []
        logger.info("🧹 Session cleared.")