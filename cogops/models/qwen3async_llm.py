import os
import json
import asyncio
import logging
from datetime import datetime
from dotenv import load_dotenv
from openai import AsyncOpenAI, APIError, BadRequestError
from typing import Any, Type, TypeVar, AsyncGenerator, List, Dict
from pydantic import BaseModel, Field
from cogops.utils.prompt import build_structured_prompt
# Load environment variables and set up logging
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PydanticModel = TypeVar("PydanticModel", bound=BaseModel)

class ContextLengthExceededError(Exception):
    """Custom exception for when a prompt exceeds the model's context window."""
    pass

class AsyncLLMService:
    """
    An ASYNCHRONOUS client for OpenAI-compatible APIs.
    """
    def __init__(self, api_key: str, model: str, base_url: str, max_context_tokens: int):
        if not api_key:
            raise ValueError("API key cannot be empty.")
        
        self.model = model
        self.max_context_tokens = max_context_tokens
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        
        print(f"✅ AsyncLLMService initialized for model '{self.model}' with max_tokens={self.max_context_tokens}.")

    async def invoke(self, prompt: str, **kwargs: Any) -> str:
        messages = [{"role": "user", "content": prompt}]
        try:
            response = await self.client.chat.completions.create(model=self.model, messages=messages, **kwargs)
            return response.choices[0].message.content or ""
        except BadRequestError as e:
            if "context length" in str(e).lower() or "too large" in str(e).lower():
                logging.error(f"Prompt exceeded context window for model {self.model}.")
                raise ContextLengthExceededError(f"Prompt is too long for the model's {self.max_context_tokens} token limit.") from e
            else:
                logging.error(f"Unhandled BadRequestError during invoke: {e}")
                raise
        except Exception as e:
            logging.error(f"An error occurred during invoke: {e}", exc_info=True)
            raise

    async def stream(self, prompt: str, **kwargs: Any) -> AsyncGenerator[str, None]:
        messages = [{"role": "user", "content": prompt}]
        try:
            stream = await self.client.chat.completions.create(model=self.model, messages=messages, stream=True, **kwargs)
            async for chunk in stream:
                content_chunk = chunk.choices[0].delta.content if chunk.choices else None
                if content_chunk:
                    yield content_chunk
        except BadRequestError as e:
            if "context length" in str(e).lower() or "too large" in str(e).lower():
                logging.error(f"Prompt exceeded context window for model {self.model}.")
                raise ContextLengthExceededError(f"Prompt is too long for the model's {self.max_context_tokens} token limit.") from e
            else:
                logging.error(f"Unhandled BadRequestError during stream: {e}")
                raise
        except Exception as e:
            logging.error(f"An error occurred during stream: {e}", exc_info=True)
            raise

    async def invoke_structured(
        self, prompt: str, response_model: Type[PydanticModel], **kwargs: Any
    ) -> PydanticModel:
        # --- MODIFIED: Use the shared utility function ---
        structured_prompt = build_structured_prompt(prompt, response_model)
        messages = [{"role": "user", "content": structured_prompt}]
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model, messages=messages, response_format={"type": "json_object"}, **kwargs
            )
            json_response_str = response.choices[0].message.content
            if not json_response_str:
                raise ValueError("The model returned an empty response.")
            return response_model.model_validate_json(json_response_str)
        except BadRequestError as e:
            if "context length" in str(e).lower() or "too large" in str(e).lower():
                logging.error(f"Prompt exceeded context window for model {self.model}.")
                raise ContextLengthExceededError(f"Prompt is too long for the model's {self.max_context_tokens} token limit.") from e
            else:
                logging.error(f"Unhandled BadRequestError during structured invoke: {e}")
                raise
        except Exception as e:
            logging.error(f"An error occurred during structured invoke: {e}", exc_info=True)
            raise

    async def invoke_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        available_tools: Dict[str, callable],
        **kwargs: Any  # <-- FIX: Accept kwargs
    ) -> str:
        """Handles a multi-step conversation with tool-calling capabilities asynchronously."""
        try:
            print("\n   [Step 1: Asking model if a tool is needed...]")
            response = await self.client.chat.completions.create(
                model=self.model, messages=messages, tools=tools, tool_choice="auto", **kwargs # <-- FIX: Pass kwargs
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
                    if asyncio.iscoroutinefunction(function_to_call):
                        function_response = await function_to_call(**function_args)
                    else:
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
            second_response = await self.client.chat.completions.create(
                model=self.model, messages=messages, **kwargs # <-- FIX: Pass kwargs
            )
            return second_response.choices[0].message.content or "Model did not provide a final response."
        except Exception as e:
            logging.error(f"An error occurred during tool invocation: {e}", exc_info=True)
            raise

    async def stream_with_tool_calls(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        available_tools: Dict[str, callable],
        **kwargs: Any  # <-- FIX: Accept kwargs
    ) -> AsyncGenerator[str, None]:
        """Handles a multi-step conversation with tool-calling capabilities, streaming the results asynchronously."""
        try:
            print("\n   [Step 1: Streaming model response to check for tool calls...]")

            stream = await self.client.chat.completions.create(
                model=self.model, messages=messages, tools=tools, tool_choice="auto", stream=True, **kwargs # <-- FIX: Pass kwargs
            )

            response_message = {"role": "assistant", "content": "", "tool_calls": []}
            tool_call_index_map = {}

            async for chunk in stream:
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
                    if asyncio.iscoroutinefunction(function_to_call):
                        function_response = await function_to_call(**function_args)
                    else:
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

            final_stream = await self.client.chat.completions.create(
                model=self.model, messages=messages, stream=True, **kwargs # <-- FIX: Pass kwargs
            )

            async for chunk in final_stream:
                if not chunk.choices:
                    continue
                content_chunk = chunk.choices[0].delta.content
                if content_chunk:
                    yield content_chunk

        except Exception as e:
            logging.error(f"An error occurred during streaming tool invocation: {e}", exc_info=True)
            raise

async def main():
    # --- Pydantic Models for Testing ---
    class NIDInfo(BaseModel):
        name: str = Field(description="The person's full name in Bengali.")
        father_name: str = Field(description="The person's father's name in Bengali.")
        occupation: str = Field(description="The person's occupation in Bengali.")

    class PassportInfo(BaseModel):
        application_type: str = Field(description="The type of passport application, e.g., 'নতুন' (New) or 'নবায়ন' (Renewal).")
        delivery_type: str = Field(description="The delivery speed, e.g., 'জরুরি' (Urgent) or 'সাধারণ' (Regular).")
        validity_years: int = Field(description="The validity period of the passport in years.")
        
    # --- Setup and Initialization ---
    print("--- Running Asynchronous LLMService Tests ---")
    
    # Load Qwen LLM Config
    api_key = os.getenv("VLLM_API_KEY")
    model = os.getenv("VLLM_MODEL_NAME")
    base_url = os.getenv("VLLM_BASE_URL")
    llm_service = None
    if all([api_key, model, base_url]):
        llm_service = AsyncLLMService(api_key, model, base_url, max_context_tokens=32000)
    else:
        print("\nWARNING: Qwen LLM environment variables not set. Skipping tests.")
        return

    # --- Test Cases ---

    # Test 1: Invoke
    print("\n--- Test 1: Invoke (Async) ---")
    try:
        prompt = "জন্ম নিবন্ধন সনদের গুরুত্ব কী?"
        print(f"Prompt: {prompt}")
        response = await llm_service.invoke(prompt, temperature=0.1, max_tokens=256)
        print(f"Response:\n{response}")
    except Exception as e:
        print(f"An error occurred: {e}")

    # Test 2: Stream
    print("\n--- Test 2: Stream (Async) ---")
    try:
        prompt = "পাসপোর্ট অফিসের একজন কর্মকর্তার একটি সংক্ষিপ্ত বর্ণনা দিন।"
        print(f"Prompt: {prompt}\nStreaming Response:")
        async for chunk in llm_service.stream(prompt, temperature=0.2, max_tokens=256):
            print(chunk, end="", flush=True)
        print()
    except Exception as e:
        print(f"An error occurred: {e}")

    # Test 3: Structured Invoke
    print("\n--- Test 3: Structured Invoke (Async) ---")
    try:
        prompt = "আমার নাম 'করিম চৌধুরী', পিতার নাম 'রহিম চৌধুরী', আমি একজন ছাত্র। এই তথ্য দিয়ে একটি এনআইডি কার্ডের তথ্য তৈরি করুন।"
        print(f"Prompt: {prompt}")
        nid_data = await llm_service.invoke_structured(prompt, NIDInfo, temperature=0.0)
        print(f"Parsed Response:\n{nid_data.model_dump_json(indent=2)}")
    except Exception as e:
        print(f"An error occurred: {e}")
            
    # Test 4: Invoke with Tools (Time Tool Example)
    print("\n--- Test 4: Invoke with Tools - Time Tool Example (Async) ---")
    try:
        user_prompt = "এখন সময় কত?"
        print(f"Prompt: {user_prompt}")
        messages = [{"role": "user", "content": user_prompt}]
        final_response = await llm_service.invoke_with_tools(
            messages=messages, tools=tools_list, available_tools=available_tools_map, temperature=0.0
        )
        print(f"\nFinal Model Response:\n{final_response}")
    except Exception as e:
        print(f"An error occurred: {e}")

    # Test 5: Stream with Tool Calls (Time Tool Example)
    print("\n--- Test 5: Stream with Tool Calls - Time Tool Example (Async) ---")
    try:
        user_prompt = "এখন সময় কত?"
        print(f"Prompt: {user_prompt}\nStreaming Response:")
        messages = [{"role": "user", "content": user_prompt}]
        async for chunk in llm_service.stream_with_tool_calls(
            messages=messages, tools=tools_list, available_tools=available_tools_map, temperature=0.0
        ):
            print(chunk, end="", flush=True)
        print()
    except Exception as e:
        print(f"An error occurred: {e}")

    # Test 6: Invoke with Tools (Retriever Tool Example)
    print("\n--- Test 6: Invoke with Tools - Retriever Tool Example (Async) ---")
    try:
        from cogops.tools.propositional_tools import tools_list, available_tools_map
        user_prompt = "আমার এন আই ডি হারায়ে গেছে রাস্তায়, কি করব?"
        print(f"Prompt: {user_prompt}")
        messages = [{"role": "user", "content": user_prompt}]
        final_response = await llm_service.invoke_with_tools(
            messages=messages, tools=tools_list, available_tools=available_tools_map
        )
        print(f"\nFinal Model Response:\n{final_response}")
    except Exception as e:
        print(f"An error occurred: {e}")

    # Test 7: Stream with Tool Calls (Retriever Tool Example)
    print("\n--- Test 7: Stream with Tool Calls - Retriever Tool Example (Async) ---")
    try:
        user_prompt = "THIS IS A GOVT SERVICE RELATED QUERY. MAKE SURE YOU ANSWER FROM KNOWLEDGEBASE. আমার এন আই ডি হারায়ে গেছে রাস্তায়, কি করব? "
        print(f"Prompt: {user_prompt}\nStreaming Response:")
        messages = [{"role": "user", "content": user_prompt}]
        async for chunk in llm_service.stream_with_tool_calls(
            messages=messages, tools=tools_list, available_tools=available_tools_map
        ):
            print(chunk, end="", flush=True)
        print()
    except Exception as e:
        print(f"An error occurred: {e}")
            
    print("\n--- All Asynchronous Tests Concluded ---")

if __name__ == '__main__':
    asyncio.run(main())