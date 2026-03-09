import json
import logging
from datetime import datetime
from openai import OpenAI, APIError, BadRequestError
from typing import Generator, Any, Type, TypeVar, List, Dict
from pydantic import BaseModel, Field

# --- Static Configuration ---
VLLM_BASE_URL = "http://127.0.0.1:5000/v1"
VLLM_API_KEY = "YOUR_VLLM_API"
VLLM_MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
MAX_CONTEXT_TOKENS = 32768

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PydanticModel = TypeVar("PydanticModel", bound=BaseModel)

class ContextLengthExceededError(Exception):
    """Custom exception for when a prompt exceeds the model's context window."""
    pass

class LLMService:
    """A synchronous client for OpenAI-compatible APIs using the 'openai' library."""
    def __init__(self, api_key: str, model: str, base_url: str, max_context_tokens: int):
        if not api_key:
            raise ValueError("API key cannot be empty.")
        
        self.model = model
        self.max_context_tokens = max_context_tokens
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        
        print(f"✅ LLMService initialized for model '{self.model}' with max_tokens={self.max_context_tokens}.")

    def invoke(self, prompt: str, **kwargs: Any) -> str:
        messages = [{"role": "user", "content": prompt}]
        try:
            response = self.client.chat.completions.create(model=self.model, messages=messages, **kwargs)
            return response.choices[0].message.content or ""
        except Exception as e:
            logging.error(f"An error occurred during invoke: {e}", exc_info=True)
            raise

    def stream(self, prompt: str, **kwargs: Any) -> Generator[str, None, None]:
        messages = [{"role": "user", "content": prompt}]
        try:
            stream = self.client.chat.completions.create(model=self.model, messages=messages, stream=True, **kwargs)
            for chunk in stream:
                content_chunk = chunk.choices[0].delta.content
                if content_chunk:
                    yield content_chunk
        except Exception as e:
            logging.error(f"An error occurred during stream: {e}", exc_info=True)
            raise

    def invoke_structured(self, prompt: str, response_model: Type[PydanticModel], **kwargs: Any) -> PydanticModel:
        structured_prompt = f"""
Please extract information from the following text and format it strictly as a JSON object that conforms to the provided Pydantic schema.
Do not include any explanatory text, markdown, or any other content outside of the JSON object.

Text to process:
"{prompt}"

Pydantic Schema:
{json.dumps(response_model.model_json_schema(), indent=2)}
"""
        messages = [{"role": "user", "content": structured_prompt}]

        try:
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, response_format={"type": "json_object"}, **kwargs
            )
            json_response_str = response.choices[0].message.content
            if not json_response_str:
                raise ValueError("The model returned an empty response.")
            return response_model.model_validate_json(json_response_str)
        except Exception as e:
            logging.error(f"An error occurred during structured invoke: {e}", exc_info=True)
            raise

    def invoke_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        available_tools: Dict[str, callable]
    ) -> str:
        """Handles a multi-step conversation with tool-calling capabilities."""
        try:
            print("\n   [Step 1: Asking model if a tool is needed...]")
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, tools=tools, tool_choice="auto",
            )
            response_message = response.choices[0].message
            tool_calls = response_message.tool_calls
            
            if not tool_calls:
                print("   [Model responded directly without using a tool.]")
                return response_message.content or ""

            print("   [Step 2: Model requested a tool call. Executing it...]")
            messages.append(response_message)
            
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_to_call = available_tools.get(function_name)
                
                if function_to_call:
                    function_args = json.loads(tool_call.function.arguments or "{}")
                    function_response = function_to_call(**function_args)
                    messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": str(function_response),
                    })
                else:
                    logging.warning(f"Model tried to call an unknown tool: {function_name}")
            
            print("   [Step 3: Sending tool result back to model for final answer...]")
            second_response = self.client.chat.completions.create(
                model=self.model, messages=messages,
            )
            return second_response.choices[0].message.content or "Model did not provide a final response."
        except Exception as e:
            logging.error(f"An error occurred during tool invocation: {e}", exc_info=True)
            raise

    def stream_with_tool_calls(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        available_tools: Dict[str, callable]
    ) -> Generator[str, None, None]:
        """Handles a multi-step conversation with tool-calling capabilities, streaming the results."""
        try:
            print("\n   [Step 1: Streaming model response to check for tool calls...]")

            stream = self.client.chat.completions.create(
                model=self.model, messages=messages, tools=tools, tool_choice="auto", stream=True
            )

            response_message = {"role": "assistant", "content": "", "tool_calls": []}
            tool_call_index_map = {}

            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta

                if delta.content:
                    response_message["content"] += delta.content
                    yield delta.content

                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        index = tc_delta.index if tc_delta.index is not None else 0
                        if index not in tool_call_index_map:
                            tool_call_index_map[index] = {
                                "id": "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""}
                            }

                        if tc_delta.id:
                            tool_call_index_map[index]["id"] += tc_delta.id
                        if tc_delta.type:
                            tool_call_index_map[index]["type"] = tc_delta.type
                        if tc_delta.function and tc_delta.function.name:
                            tool_call_index_map[index]["function"]["name"] += tc_delta.function.name
                        if tc_delta.function and tc_delta.function.arguments:
                            tool_call_index_map[index]["function"]["arguments"] += tc_delta.function.arguments

            response_message["tool_calls"] = list(tool_call_index_map.values())
            tool_calls = response_message["tool_calls"]

            if not tool_calls:
                print("   [Model responded directly without using a tool.]")
                return

            print("   [Step 2: Model requested tool calls. Executing them...]")

            messages.append(response_message)

            for tool_call in tool_calls:
                function_name = tool_call["function"]["name"]
                function_to_call = available_tools.get(function_name)

                if function_to_call:
                    function_args = json.loads(tool_call["function"]["arguments"] or "{}")
                    function_response = function_to_call(**function_args)
                    messages.append({
                        "tool_call_id": tool_call["id"],
                        "role": "tool",
                        "name": function_name,
                        "content": str(function_response),
                    })
                else:
                    logging.warning(f"Model tried to call an unknown tool: {function_name}")

            print("   [Step 3: Streaming final answer from model...]")

            final_stream = self.client.chat.completions.create(
                model=self.model, messages=messages, stream=True
            )

            for chunk in final_stream:
                if not chunk.choices:
                    continue
                content_chunk = chunk.choices[0].delta.content
                if content_chunk:
                    yield content_chunk

        except Exception as e:
            logging.error(f"An error occurred during streaming tool invocation: {e}", exc_info=True)
            raise

if __name__ == '__main__':
    class NIDInfo(BaseModel):
        name: str = Field(description="The person's full name in Bengali.")
        father_name: str = Field(description="The person's father's name in Bengali.")
        occupation: str = Field(description="The person's occupation in Bengali.")

    print("--- Running LLMService Tests ---")
    
    try:
        llm_service = LLMService(
            api_key=VLLM_API_KEY, model=VLLM_MODEL_NAME,
            base_url=VLLM_BASE_URL, max_context_tokens=MAX_CONTEXT_TOKENS
        )
    except Exception as e:
        print(f"\nFATAL: Could not initialize LLMService. Is the vLLM server running? Error: {e}")
        exit(1)

    print("\n--- Test 1: Invoke ---")
    try:
        prompt = "জন্ম নিবন্ধন সনদের গুরুত্ব কী?"
        print(f"Prompt: {prompt}")
        response = llm_service.invoke(prompt, temperature=0.1, max_tokens=256)
        print(f"Response:\n{response}")
    except Exception as e:
        print(f"An error occurred: {e}")

    print("\n--- Test 2: Stream ---")
    try:
        prompt = "পাসপোর্ট অফিসের একজন কর্মকর্তার একটি সংক্ষিপ্ত বর্ণনা দিন।"
        print(f"Prompt: {prompt}\nStreaming Response:")
        for chunk in llm_service.stream(prompt, temperature=0.2, max_tokens=256):
            print(chunk, end="", flush=True)
        print()
    except Exception as e:
        print(f"An error occurred: {e}")

    print("\n--- Test 3: Structured Invoke ---")
    try:
        prompt = "আমার নাম 'করিম চৌধুরী', পিতার নাম 'রহিম চৌধুরী', আমি একজন ছাত্র। এই তথ্য দিয়ে একটি এনআইডি কার্ডের তথ্য তৈরি করুন।"
        print(f"Prompt: {prompt}")
        nid_data = llm_service.invoke_structured(prompt, NIDInfo, temperature=0.0)
        # --- FIX FOR PYDANTIC V2 ---
        # model_dump_json doesn't accept ensure_ascii. We dump to a dict first, then use the standard json library.
        parsed_json_string = json.dumps(nid_data.model_dump(), indent=2, ensure_ascii=False)
        print(f"Parsed Response:\n{parsed_json_string}")
    except Exception as e:
        print(f"An error occurred: {e}")

    print("\n--- Test 4: Invoke with Tools ---")
    def get_current_time():
        """Returns the current server date and time as a formatted string."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    available_tools_map = {"get_current_time": get_current_time}
    tools_list = [{"type": "function","function": {"name": "get_current_time","description": "Get the current date and time.","parameters": {"type": "object","properties": {},"required": []}}}]

    try:
        user_prompt = "এখন সময় কত?"
        print(f"Prompt: {user_prompt}")
        messages = [{"role": "user", "content": user_prompt}]
        final_response = llm_service.invoke_with_tools(
            messages=messages, tools=tools_list, available_tools=available_tools_map
        )
        print(f"\nFinal Model Response:\n{final_response}")
    except Exception as e:
        print(f"An error occurred: {e}")

    print("\n--- Test 5: Stream with Tool Calls ---")
    try:
        user_prompt = "এখন সময় কত?"
        print(f"Prompt: {user_prompt}\nStreaming Response:")
        messages = [{"role": "user", "content": user_prompt}]
        for chunk in llm_service.stream_with_tool_calls(
            messages=messages, tools=tools_list, available_tools=available_tools_map
        ):
            print(chunk, end="", flush=True)
        print()
    except Exception as e:
        print(f"An error occurred: {e}")
            
    print("\n--- All Tests Concluded ---")