"""
cogops/llm/reasoning_loop.py

The tool-calling while loop, extracted from AsyncLLMService.stream_with_tool_calls.
Tags every yielded event with a channel ("user" or "debug").
"""

import os
import json
import asyncio
import logging
from typing import Any, AsyncGenerator, List, Dict, Optional, Callable
from openai import AsyncOpenAI, BadRequestError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

RETRYABLE_EXCEPTIONS = (ConnectionError, TimeoutError, RuntimeError)

_MISSING = object()  # sentinel for getattr on pydantic models

MAX_TURNS = 10

logger = logging.getLogger(__name__)


def log_retry_attempt(retry_state):
    """Helper to log warnings when retrying API calls."""
    logger = logging.getLogger(__name__)
    logger.warning(
        f"LLM API call failed with {retry_state.outcome.exception()}, "
        f"retrying in {retry_state.next_action.sleep} seconds... "
        f"(Attempt {retry_state.attempt_number})"
    )


class ContextLengthExceededError(Exception):
    """Raised when the conversation history exceeds the model's limit."""
    pass


def _make_event(event_type: str, data: dict, channel: str) -> Dict[str, Any]:
    """Create a tagged event dict."""
    evt = {"type": event_type, "channel": channel}
    evt.update(data)
    return evt


@retry(
    wait=wait_exponential(multiplier=1, min=1, max=10),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
    before_sleep=log_retry_attempt,
)
async def stream_with_tool_calls(
    client_llm: AsyncOpenAI,
    model: str,
    messages: List[Dict[str, Any]],
    tools_schema: List[Dict[str, Any]],
    available_tools: Dict[str, Callable],
    debug_mode: bool = False,
    max_turns: int = MAX_TURNS,
    extra_body: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Orchestrates the conversation using the primary LLM endpoint:

    1. Sends prompt -> yields text
    2. Captures tool calls -> executes tools
    3. Sends results back -> yields final answer

    All events are tagged with a channel:
      - "debug": reasoning chunks, tool calls, tool results, turn markers
      - "user":  only the final turn's text content (answer_chunk)
                 plus any error messages
    """
    extra_body = extra_body or {}
    vllm_params = ['repetition_penalty', 'top_k', 'top_p']
    for param in vllm_params:
        if param in kwargs:
            extra_body[param] = kwargs.pop(param)

    # Include usage info in the last streaming chunk for token accounting
    extra_body.setdefault("stream_options", {"include_usage": True})

    turn_count = 0
    reasoning_accumulator = ""

    # Track whether the current turn is the last (no tools)
    # We don't know this until the stream ends.
    is_last_turn = False

    while turn_count < max_turns:
        turn_count += 1
        logger.info(f"Turn {turn_count}/{max_turns} started.")

        is_last_turn = False  # will be set to True if no tool_calls

        try:
            yield _make_event("turn_start", {"turn_number": turn_count}, "debug")

            # Force graph_search on the first turn to prevent the model from
            # answering without searching. After tool results are fed back,
            # allow the model to choose freely (auto).
            tool_choice_val = "auto"
            if tools_schema and turn_count <= 1:
                tool_choice_val = {"type": "function", "function": {"name": "graph_search"}}

            stream = await client_llm.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools_schema if tools_schema else None,
                tool_choice=tool_choice_val if tools_schema else None,
                stream=True,
                extra_body=extra_body,
                **kwargs
            )

            full_content_accumulator = ""
            tool_call_index_map = {}

            # Track usage from vLLM's stream_options.include_usage chunk
            stream_usage: Dict[str, Any] = {}

            async for chunk in stream:
                # Capture usage from the special usage chunk (empty choices, has usage)
                if not chunk.choices and hasattr(chunk, "usage") and chunk.usage:
                    u = chunk.usage
                    stream_usage = {
                        "prompt_tokens": getattr(u, "prompt_tokens", 0),
                        "completion_tokens": getattr(u, "completion_tokens", 0),
                        "total_tokens": getattr(u, "total_tokens", 0),
                    }
                    continue

                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta

                # --- Handle Native Thinking Content (always debug) ---
                reasoning = getattr(delta, "reasoning", _MISSING)
                if reasoning is not _MISSING and reasoning:
                    reasoning_accumulator += reasoning
                    yield _make_event("reasoning_chunk", {"data": reasoning}, "debug")
                    continue

                # --- Handle Text Content ---
                if delta.content:
                    content_chunk = delta.content
                    full_content_accumulator += content_chunk
                    # Only yield as user channel on the final turn
                    if is_last_turn:
                        yield _make_event("answer_chunk", {"content": content_chunk}, "both")
                    else:
                        yield _make_event("reasoning_chunk", {"data": content_chunk}, "debug")

                # --- Handle Tool Call Accumulation ---
                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        index = tc_delta.index
                        if index not in tool_call_index_map:
                            tool_call_index_map[index] = {
                                "id": "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""}
                            }
                        if tc_delta.id:
                            tool_call_index_map[index]["id"] += tc_delta.id
                        if tc_delta.function and tc_delta.function.name:
                            tool_call_index_map[index]["function"]["name"] += tc_delta.function.name
                        if tc_delta.function and tc_delta.function.arguments:
                            tool_call_index_map[index]["function"]["arguments"] += tc_delta.function.arguments

            # Flush remaining reasoning
            if reasoning_accumulator:
                yield _make_event("reasoning_chunk", {"data": reasoning_accumulator}, "debug")
            reasoning_accumulator = ""

            # Emit captured usage from vLLM stream_options
            if stream_usage:
                yield _make_event("usage", {"tokens": stream_usage}, "debug")

            # Build response message
            response_message = {
                "role": "assistant",
                "content": full_content_accumulator if full_content_accumulator else None
            }

            tool_calls_list = list(tool_call_index_map.values())
            if tool_calls_list:
                response_message["tool_calls"] = tool_calls_list

            messages.append(response_message)

            # Safety net: if no tools called despite tool_choice forcing graph_search,
            # inject a reminder into the system prompt and continue. Only allow this
            # once (on turn 1), then bail out.
            if not tool_calls_list:
                if turn_count <= 1:
                    logger.warning("No tool call emitted — injecting mandatory tool reminder.")
                    if messages and messages[0].get("role") == "system":
                        messages[0]["content"] += (
                            "\n\n[SYSTEM REMINDER: You MUST call a tool before producing any answer. "
                            "Do NOT answer from your own knowledge. Call a tool with Bangla keywords now.]"
                        )
                    continue
                else:
                    logger.info("No tools called. Ending turn loop.")
                    is_last_turn = True
                    if full_content_accumulator:
                        yield _make_event("answer_chunk", {"content": full_content_accumulator}, "both")
                    break

            # Emit tool_call events (debug only)
            yield _make_event("tool_call", {
                "tool_calls": tool_calls_list,
                "turn": turn_count
            }, "debug")

            # --- Tool Execution Phase ---
            logger.info(f"Executing {len(tool_calls_list)} tool(s)...")
            import time
            start_time = time.time()

            for tool_call in tool_calls_list:
                function_name = tool_call["function"]["name"]
                call_id = tool_call["id"]

                function_to_call = available_tools.get(function_name)
                tool_result_content = ""

                if function_to_call:
                    try:
                        raw_args = tool_call["function"]["arguments"] or "{}"
                        try:
                            function_args = json.loads(raw_args)
                        except json.JSONDecodeError:
                            tool_result_content = "Error: Invalid JSON arguments generated by model."
                            function_args = {}

                        if not tool_result_content:
                            if asyncio.iscoroutinefunction(function_to_call):
                                response_data = await function_to_call(**function_args)
                            else:
                                response_data = await asyncio.to_thread(function_to_call, **function_args)
                            tool_result_content = str(response_data)

                    except Exception as e:
                        from cogops.tools.ask_user import ClarificationRequested
                        if isinstance(e, ClarificationRequested):
                            raise  # Re-raise to outer handler
                        logger.error(f"Error executing {function_name}: {e}", exc_info=True)
                        tool_result_content = f"System Error executing tool: {str(e)}"
                else:
                    tool_result_content = f"Error: Tool '{function_name}' not defined in available_tools_map."

                elapsed = time.time() - start_time
                yield _make_event("tool_result", {
                    "call_id": call_id,
                    "content": tool_result_content,
                    "duration_ms": round(elapsed * 1000),
                    "status": "error" if tool_result_content.startswith("Error") else "ok"
                }, "debug")

                messages.append({
                    "tool_call_id": call_id,
                    "role": "tool",
                    "name": function_name,
                    "content": tool_result_content
                })

        except BadRequestError as e:
            if "context length" in str(e).lower():
                logger.error("FATAL: Prompt exceeded context window.")
                yield _make_event("error", {"content": "Context limit reached. Please clear session."}, "user")
                raise ContextLengthExceededError() from e
            else:
                logger.error(f"API Bad Request: {e}")
                raise
        except Exception as e:
            from cogops.tools.ask_user import ClarificationRequested
            if isinstance(e, ClarificationRequested):
                logger.info("ClarificationRequested raised by tool.")
                yield _make_event("clarification_needed", {
                    "question": e.question,
                    "options": e.options,
                    "reason": e.reason,
                    "turn_id": e.turn_id,
                }, "user")
                return  # End stream cleanly on clarification

            logger.error(f"Unexpected error in LLM loop: {e}", exc_info=True)
            yield _make_event("error", {"content": "An internal error occurred.", "detail": str(e)}, "both")
            raise

        yield _make_event("turn_end", {"turn_number": turn_count}, "debug")

    # If we exhausted max_turns without producing a user-visible answer,
    # emit any remaining accumulated content as final answer.
    if not is_last_turn:
        logger.info(f"Reached max turns ({max_turns}) without a final answer. Emitting accumulated content.")
        yield _make_event("answer_chunk", {"content": full_content_accumulator}, "both")


async def classify(
    client_reranker: Optional[AsyncOpenAI],
    reranker_model: str,
    query: str,
    passages: List[str],
) -> List[tuple]:
    """Use reranker endpoint for binary classification of passages."""
    if not client_reranker:
        return [(p, 0.0) for p in passages]

    from cogops.llm.reranker import QwenRerankerClient
    from graphiti_core.llm_client.config import LLMConfig as RerankerLLMConfig

    reranker_llm_config = RerankerLLMConfig(
        api_key=client_reranker.api_key or "",
        base_url=client_reranker.base_url or "",
        model=reranker_model or "reranker",
        max_tokens=1
    )
    reranker = QwenRerankerClient(client=client_reranker, config=reranker_llm_config)
    return await reranker.rank(query, passages)
