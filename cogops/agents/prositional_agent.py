import os
import yaml
import json
import asyncio
import logging
from typing import AsyncGenerator, Dict, Any, List, Tuple

# Environment Variable Loading
from dotenv import load_dotenv
load_dotenv()

# --- Exception Imports for Network Errors ---
from openai import APIConnectionError, APITimeoutError
from requests.exceptions import RequestException

# --- Core Component Imports ---
# The new master prompt that defines the agent's behavior and tool usage
from cogops.prompts.propositional_prompt import get_agent_prompt
# The available tools (functions and their schemas) for the agent
from cogops.tools.propositional_tools import tools_list, available_tools_map
# The asynchronous LLM service with tool-calling capabilities
from cogops.models.qwen3async_llm import AsyncLLMService
# The token manager is still useful for formatting history
from cogops.utils.token_manager import TokenManager

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ChatAgent:
    """
    An end-to-end conversational agent that uses a tool-based pipeline.

    This agent leverages a single, powerful LLM guided by a master system prompt.
    The LLM's primary role is to understand the user's intent, call the
    necessary tools (like a knowledge base retriever), and synthesize a final
    answer based on the tool outputs.
    """
    def __init__(self, config_path: str = "configs/config.yaml"):
        logging.info("Initializing Tool-Based ChatAgent...")
        self.config = self._load_config(config_path)
        
        # --- Agent Identity ---
        self.agent_name = self.config.get('agent_name', 'AI Assistant')
        self.agent_story = self.config.get('agent_story', 'I am a helpful AI assistant.')

        # --- Initialize LLM Service ---
        # In a tool-based model, we primarily use one powerful orchestrator LLM.
        self.llm_service = self._initialize_llm_service()

        # --- Initialize Tokenizer ---
        # Used for formatting conversation history without exceeding limits.
        token_cfg = self.config['token_management']
        tokenizer_model = os.getenv(token_cfg['tokenizer_model_name_env'])
        if not tokenizer_model:
            raise ValueError("Tokenizer model name not found in environment variables.")
        # Note: We don't use the full TokenManager for prompt building anymore,
        # as the system prompt is now static. We only need its history truncation.
        self.token_manager = TokenManager(
            model_name=tokenizer_model,
            reservation_tokens=0, # Not needed in this new flow
            history_budget=1.0 # Not needed in this new flow
        )

        # --- Conversation and Tool Management ---
        self.history: List[Tuple[str, str]] = []
        self.history_window = self.config['conversation']['history_window']
        self.llm_call_params = self.config.get('llm_call_parameters', {})
        self.response_templates = self.config['response_templates']
        
        # Load the tools the agent can use
        self.tools_schema = tools_list
        self.tool_functions = available_tools_map
        # Format a description of tools for the master prompt
        self.tools_description = json.dumps(self.tools_schema, indent=2)

        logging.info("✅ Tool-Based ChatAgent initialized successfully.")

    def _load_config(self, config_path: str) -> Dict:
        logging.info(f"Loading configuration from: {config_path}")
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logging.error(f"[FATAL ERROR] Configuration file not found at: {config_path}")
            raise
    
    def _initialize_llm_service(self) -> AsyncLLMService:
        logging.info("Initializing primary LLM service...")
        cfg = self.config['llm_service']
        api_key = os.getenv(cfg['api_key_env'])
        model = os.getenv(cfg['model_name_env'])
        url = os.getenv(cfg['base_url_env'])
        max_tokens = cfg.get('max_context_tokens', 32000)
        
        if not all([api_key, model, url]):
            raise ValueError(f"Missing environment variables for LLM service '{cfg['name']}'")
            
        service = AsyncLLMService(api_key, model, url, max_tokens)
        logging.info(f"LLM service '{cfg['name']}' is ready.")
        return service

    def _format_conversation_history(self) -> str:
        """Formats the conversation history into a simple string for the prompt."""
        if not self.history:
            return "No conversation history yet."
        # Use the token manager just for its truncation logic if needed,
        # though with large context models this is less of a concern.
        return "\n---\n".join([f"User: {u}\nAI: {a}" for u, a in self.history])

    async def process_query(self, user_query: str) -> AsyncGenerator[Dict[str, Any], None]:
        logging.info(f"\n--- New Query Received: '{user_query}' ---")
        
        try:
            # --- 1. Construct the Master Prompt and Message History ---
            conversation_history_str = self._format_conversation_history()

            master_prompt = get_agent_prompt(
                agent_name=self.agent_name,
                agent_story=self.agent_story,
                tools_description=self.tools_description,
                conversation_history=conversation_history_str,
                user_query=user_query
            )
            
            # The message list for the API must contain the full context.
            # The master prompt serves as the 'system' message.
            messages = [{"role": "system", "content": master_prompt}]
            
            # Note: We're providing the history within the system prompt as per the prompt.py template.
            # We add the final user query here to trigger the response.
            messages.append({"role": "user", "content": user_query})

            # --- 2. Stream the Response from the LLM with Tools ---
            # The `stream_with_tool_calls` method handles the entire flow:
            # - Sending the prompt to the LLM.
            # - If the LLM requests a tool, executing it.
            # - Sending the tool's result back to the LLM.
            # - Streaming the final, synthesized answer.
            
            full_final_answer = []
            stream_generator = self.llm_service.stream_with_tool_calls(
                messages=messages,
                tools=self.tools_schema,
                available_tools=self.tool_functions,
                **self.llm_call_params
            )

            async for chunk in stream_generator:
                full_final_answer.append(chunk)
                yield {"type": "answer_chunk", "content": chunk}

            final_answer_str = "".join(full_final_answer).strip()

            # --- 3. Update Conversation History ---
            if final_answer_str:
                self.history.append((user_query, final_answer_str))
            
            # Truncate history to maintain the configured window size
            if len(self.history) > self.history_window:
                self.history.pop(0)

        except (APIConnectionError, APITimeoutError, RequestException) as e:
            logging.error(f"A network service is unavailable. Underlying error: {e}")
            yield {"type": "error", "content": self.response_templates['error_fallback']}
            return
        
        except Exception as e:
            logging.error(f"An unexpected error occurred during query processing: {e}", exc_info=True)
            yield {"type": "error", "content": self.response_templates['error_fallback']}
            return


async def main():
    """Main function to demonstrate a conversational flow with the ChatAgent."""
    try:
        agent = ChatAgent(config_path="configs/config.yaml")
        
        queries = [
            # "হ্যালো, আপনি কে?", # Identity inquiry
            # "আমার এনআইডি কার্ড হারিয়ে গেছে, এখন কি করব?", # In-domain service query
            # "এর জন্য কি কোনো ফি দিতে হবে?", # Follow-up query
            # "এখন সময় কত?", # Tool use (get_current_time)
            # "আপত্তিকর মন্তব্য", # Abusive language test
            # "জন্ম নিবন্ধন করার প্রক্রিয়া কি?",
            # "নিবন্ধন নম্বর হারিয়ে ফেলেছি কি করতে পারি ?",
            # "আমার মামা অসুস্থ । আমি যশোরের হাসপাতালের নাম্বার চাই ",
            # "ই-পাসপোর্টের নির্দেশনাবলি",
            # "২০০৮ সালে আমার পাসপোর্ট হায়ায়ে গেছে । আমার করনীয় কি ? আমি কি স্থনীয় পাস্পোর্ট অফিসে যাবও?",
            # "রাজশাহী কৃষি অফিসের ঠিকানা কি?",
            #"মুক্তিযোদ্ধাদের আয়কর এর তথ্য সম্পর্কে জানতে চাই",
            "বিধবা ভাতা আর মুক্তিযোদ্ধা ভাতা কত ? কোনটা পাইতে কি করা লাগে ?"
            "ধন্যবাদ"
            
        ]

        for query in queries:
            print("\n" + "="*50 + f"\nUser Query: {query}\n" + "="*50)
            print("AI Response: ", end="", flush=True)
            
            async for event in agent.process_query(query):
                if event["type"] == "answer_chunk":
                    print(event["content"], end="", flush=True)
                elif event["type"] == "error":
                    print(f"\n[AGENT ERROR] {event['content']}")
            
            print("\n")
            
    except Exception as e:
        logging.error(f"A fatal error occurred during agent initialization or execution: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())