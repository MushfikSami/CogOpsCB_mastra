import os
import json
import asyncio
import logging
from dotenv import load_dotenv
from openai import AsyncOpenAI, BadRequestError, APIConnectionError, APITimeoutError
from typing import Any, AsyncGenerator, List, Dict
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Load environment variables
load_dotenv()

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ContextLengthExceededError(Exception):
    """Raised when the conversation history exceeds the model's limit."""
    pass

# Exceptions that trigger a retry
RETRYABLE_EXCEPTIONS = (APIConnectionError, APITimeoutError)

# Safety limit to prevent infinite tool-calling loops
MAX_TURNS = 5  

def log_retry_attempt(retry_state):
    """Helper to log warnings when retrying API calls."""
    logger.warning(
        f"LLM API call failed with {retry_state.outcome.exception()}, "
        f"retrying in {retry_state.next_action.sleep} seconds... "
        f"(Attempt {retry_state.attempt_number})"
    )

class AsyncLLMService:
    """
    Asynchronous LLM Service specialized for Qwen/vLLM with Chain-of-Thought (CoT) hiding
    and Tool Calling capabilities.
    """
    def __init__(self, api_key: str, model: str, base_url: str, max_context_tokens: int):
        if not api_key:
            raise ValueError("API key cannot be empty.")
        
        self.model = model
        self.max_context_tokens = max_context_tokens
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        logger.info(f"✅ AsyncLLMService initialized for model '{self.model}' @ {base_url}")

    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        before_sleep=log_retry_attempt
    )
    async def stream_with_tool_calls(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        available_tools: Dict[str, callable],
        debug_mode: bool = False,
        **kwargs: Any
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Orchestrates the conversation:
        1. Sends Prompt -> Yields Text (hiding CoT)
        2. Captures Tool Calls -> Executes Tools
        3. Sends Results back to Model -> Yields Final Answer
        """
        
        # Extract vLLM specific parameters from kwargs if present
        extra_body = kwargs.pop('extra_body', {})
        vllm_params = ['repetition_penalty', 'top_k', 'top_p']
        for param in vllm_params:
            if param in kwargs:
                extra_body[param] = kwargs.pop(param)

        turn_count = 0

        # --- MULTI-TURN LOOP (Reasoning -> Tool -> Reasoning -> Answer) ---
        while turn_count < MAX_TURNS:
            turn_count += 1
            logger.info(f"🔄 Turn {turn_count}/{MAX_TURNS} started.")

            try:
                # Call vLLM / OpenAI API
                stream = await self.client.chat.completions.create(
                    model=self.model, 
                    messages=messages, 
                    tools=tools if tools else None, 
                    tool_choice="auto" if tools else None, 
                    stream=True, 
                    extra_body=extra_body, 
                    **kwargs
                )

                # --- State Management for this Turn ---
                full_content_accumulator = ""
                tool_call_index_map = {}
                
                # CoT Filtering State
                cot_active = False     # Are we currently inside a <CoT> block?
                buffer = ""            # Buffer to hold text while checking for tags
                
                async for chunk in stream:
                    if not chunk.choices: continue
                    delta = chunk.choices[0].delta
                    
                    # ---------------------------------------------------------
                    # 1. Handle Text Content (CoT Parsing logic)
                    # ---------------------------------------------------------
                    if delta.content:
                        content_chunk = delta.content
                        full_content_accumulator += content_chunk # Keep everything for history
                        buffer += content_chunk

                        # Check for Start Tag <CoT>
                        if not cot_active:
                            if "<CoT>" in buffer:
                                cot_active = True
                                # If there was text before <CoT>, yield it
                                pre_cot, post_cot = buffer.split("<CoT>", 1)
                                if pre_cot:
                                    yield {"type": "answer_chunk", "content": pre_cot}
                                buffer = post_cot # Keep checking inside buffer for potential immediate close
                            
                            # Heuristic: If buffer gets too long without a tag, safe to yield
                            # This prevents holding "Hello" in buffer forever waiting for a tag that never comes
                            elif len(buffer) > 20 and "<" not in buffer:
                                yield {"type": "answer_chunk", "content": buffer}
                                buffer = ""
                        
                        # Check for End Tag </CoT>
                        if cot_active:
                            if "</CoT>" in buffer:
                                # CoT finished.
                                _, remainder = buffer.split("</CoT>", 1)
                                
                                # OPTIONAL: Yield debug info if enabled
                                if debug_mode:
                                    cot_content = buffer.split("</CoT>")[0]
                                    yield {"type": "debug_log", "title": "🧠 Thinking", "data": cot_content}
                                
                                cot_active = False
                                buffer = remainder # Keep remainder to process normally
                            else:
                                # We are inside CoT, suppress output (do not yield)
                                pass

                        # If we are NOT in CoT and buffer is safe (no partial tags)
                        if not cot_active and buffer:
                            # Avoid yielding partial tags like "<" or "<Co"
                            if "<" in buffer:
                                # We have a potential tag start, keep in buffer
                                pass 
                            else:
                                yield {"type": "answer_chunk", "content": buffer}
                                buffer = ""

                    # ---------------------------------------------------------
                    # 2. Handle Tool Call Accumulation (Standard OpenAI Format)
                    # ---------------------------------------------------------
                    if delta.tool_calls:
                        for tc_delta in delta.tool_calls:
                            index = tc_delta.index
                            
                            # Initialize map for this index if new
                            if index not in tool_call_index_map:
                                tool_call_index_map[index] = {
                                    "id": "", 
                                    "type": "function", 
                                    "function": {"name": "", "arguments": ""}
                                }
                            
                            # Append parts
                            if tc_delta.id: 
                                tool_call_index_map[index]["id"] += tc_delta.id
                            if tc_delta.function and tc_delta.function.name: 
                                tool_call_index_map[index]["function"]["name"] += tc_delta.function.name
                            if tc_delta.function and tc_delta.function.arguments: 
                                tool_call_index_map[index]["function"]["arguments"] += tc_delta.function.arguments
                
                # --- End of Stream Processing ---
                
                # Flush any remaining buffer (e.g., if </CoT> was missing or text ended)
                if buffer and not cot_active:
                    # Cleanup stray tags if model hallucinated malformed XML
                    clean_buffer = buffer.replace("</CoT>", "").replace(">", "")
                    if clean_buffer:
                        yield {"type": "answer_chunk", "content": clean_buffer}

                # Update conversation history with what the model actually generated (including CoT)
                response_message = {
                    "role": "assistant", 
                    "content": full_content_accumulator if full_content_accumulator else None
                }
                
                # Prepare tool calls list
                tool_calls_list = list(tool_call_index_map.values())
                if tool_calls_list:
                    response_message["tool_calls"] = tool_calls_list
                
                # Append Assistant message to history
                messages.append(response_message)

                # --- DECISION POINT ---
                # If no tools were called, the model is done. Break the loop.
                if not tool_calls_list:
                    logger.info("🏁 No tools called. Ending turn loop.")
                    break
                
                # ---------------------------------------------------------
                # 3. Tool Execution Phase
                # ---------------------------------------------------------
                logger.info(f"🛠️ Executing {len(tool_calls_list)} tool(s)...")
                
                for tool_call in tool_calls_list:
                    function_name = tool_call["function"]["name"]
                    call_id = tool_call["id"]
                    
                    
                    # Look up the function
                    function_to_call = available_tools.get(function_name)
                    tool_result_content = ""

                    if function_to_call:
                        try:
                            # Parse JSON arguments
                            raw_args = tool_call["function"]["arguments"] or "{}"
                            try:
                                function_args = json.loads(raw_args)
                            except json.JSONDecodeError:
                                tool_result_content = "Error: Invalid JSON arguments generated by model."
                                function_args = {}

                            if not tool_result_content:
                                # Execute (Support both Async and Sync functions)
                                if asyncio.iscoroutinefunction(function_to_call):
                                    response_data = await function_to_call(**function_args)
                                else:
                                    response_data = await asyncio.to_thread(function_to_call, **function_args)
                                
                                tool_result_content = str(response_data)
                                if debug_mode:
                                    yield {"type": "debug_log", "title": "🛠️ Tool Call", "data": tool_result_content}
                    

                        except Exception as e:
                            logger.error(f"Error executing {function_name}: {e}", exc_info=True)
                            tool_result_content = f"System Error executing tool: {str(e)}"
                    else:
                        tool_result_content = f"Error: Tool '{function_name}' not defined in available_tools_map."

                    # Add Tool Result to History
                    messages.append({
                        "tool_call_id": call_id,
                        "role": "tool",
                        "name": function_name,
                        "content": tool_result_content
                    })

                # Loop continues to next turn to let Model synthesize the answer...

            except BadRequestError as e:
                if "context length" in str(e).lower():
                    logger.error(f"FATAL: Prompt exceeded context window.")
                    yield {"type": "error", "content": "Context limit reached. Please clear session."}
                    raise ContextLengthExceededError() from e
                else:
                    logger.error(f"API Bad Request: {e}")
                    raise
            except Exception as e:
                logger.error(f"Unexpected error in LLM loop: {e}", exc_info=True)
                yield {"type": "error", "content": "An internal error occurred."}
                raise