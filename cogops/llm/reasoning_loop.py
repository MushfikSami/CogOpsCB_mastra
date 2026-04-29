"""
cogops/llm/reasoning_loop.py

The ReAct tool-calling while loop. Every yielded event is tagged with a channel
("user", "debug", or "both"); the API layer filters events by channel based
on whether the caller is authenticated for debug.

Behaviour guarantees:
- Content deltas are streamed chunk-by-chunk on channel "both" as they
  arrive. No end-of-turn buffering.
"""

import json
import asyncio
import logging
import time
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional

from openai import AsyncOpenAI, BadRequestError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

_DEFAULT_CHARS_PER_CHUNK = 12
_DEFAULT_DELAY_SECONDS = 0.015

RETRYABLE_EXCEPTIONS = (ConnectionError, TimeoutError, RuntimeError)

_DEFAULT_MAX_TURNS = 10

logger = logging.getLogger(__name__)


def log_retry_attempt(retry_state):
    logger.warning(
        f"LLM API call failed with {retry_state.outcome.exception()}, "
        f"retrying in {retry_state.next_action.sleep} seconds... "
        f"(Attempt {retry_state.attempt_number})"
    )


class ContextLengthExceededError(Exception):
    """Raised when the conversation history exceeds the model's limit."""
    pass


def _make_event(event_type: str, data: dict, channel: str) -> Dict[str, Any]:
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
    max_turns: int = _DEFAULT_MAX_TURNS,
    extra_body: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Orchestrates the ReAct conversation using the primary LLM endpoint.

    Stream-friendly: text deltas are yielded as `answer_chunk` events on
    channel "both" as they arrive. Reasoning, tool calls, tool results,
    and token usage are yielded on channel "debug".

    The model operates in a Thought-Action-Observation loop:
    - THOUGHT: classifies intent, decides which tool to call
    - ACTION: the model calls exactly one tool
    - OBSERVATION: the tool result is fed back; model decides if answer is complete
    """
    extra_body = dict(extra_body or {})

    turn_count = 0
    is_last_turn = False

    while turn_count < max_turns:
        turn_count += 1
        logger.info(f"Turn {turn_count}/{max_turns} started.")

        is_last_turn = False

        try:
            yield _make_event("turn_start", {"turn_number": turn_count}, "debug")

            tool_choice_val = "auto"

            stream = await client_llm.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools_schema if tools_schema else None,
                tool_choice=tool_choice_val if tools_schema else None,
                stream=True,
                extra_body=extra_body,
                **kwargs,
            )

            full_content_accumulator = ""
            tool_call_index_map: Dict[int, Dict[str, Any]] = {}

            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta

                if delta.content:
                    content_chunk = delta.content
                    full_content_accumulator += content_chunk
                    yield _make_event(
                        "answer_chunk", {"content": content_chunk}, "both"
                    )

                # Accumulate tool call fragments.
                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        index = tc_delta.index
                        if index not in tool_call_index_map:
                            tool_call_index_map[index] = {
                                "id": "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            }
                        if tc_delta.id:
                            tool_call_index_map[index]["id"] += tc_delta.id
                        if tc_delta.function and tc_delta.function.name:
                            tool_call_index_map[index]["function"]["name"] += tc_delta.function.name
                        if tc_delta.function and tc_delta.function.arguments:
                            tool_call_index_map[index]["function"]["arguments"] += tc_delta.function.arguments

            # Build the assistant message to feed back in.
            response_message: Dict[str, Any] = {
                "role": "assistant",
                "content": full_content_accumulator if full_content_accumulator else None,
            }

            tool_calls_list = list(tool_call_index_map.values())
            if tool_calls_list:
                response_message["tool_calls"] = tool_calls_list

            messages.append(response_message)

            if not tool_calls_list:
                # Model produced a final answer without a tool call.
                is_last_turn = True
                break

            yield _make_event(
                "tool_call",
                {"tool_calls": tool_calls_list, "turn": turn_count},
                "debug",
            )

            # --- Execute the tools ---
            logger.info(f"Executing {len(tool_calls_list)} tool(s)...")

            for tool_call in tool_calls_list:
                function_name = tool_call["function"]["name"]
                call_id = tool_call["id"]

                function_to_call = available_tools.get(function_name)
                tool_result_content = ""
                start_time = time.time()

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
                                inner = getattr(function_to_call, "func", function_to_call)
                                if asyncio.iscoroutinefunction(inner):
                                    response_data = await function_to_call(**function_args)
                                else:
                                    response_data = await asyncio.to_thread(function_to_call, **function_args)
                            tool_result_content = str(response_data) if response_data is not None else ""

                    except Exception as e:
                        logger.error(f"Error executing {function_name}: {e}", exc_info=True)
                        tool_result_content = f"System Error executing tool: {str(e)}"
                else:
                    tool_result_content = (
                        f"Error: Tool '{function_name}' not defined in available_tools_map."
                    )

                elapsed_ms = round((time.time() - start_time) * 1000)
                yield _make_event(
                    "tool_result",
                    {
                        "call_id": call_id,
                        "content": tool_result_content,
                        "duration_ms": elapsed_ms,
                        "status": "error" if tool_result_content.startswith("Error") else "ok",
                    },
                    "debug",
                )

                messages.append({
                    "tool_call_id": call_id,
                    "role": "tool",
                    "name": function_name,
                    "content": tool_result_content,
                })

        except BadRequestError as e:
            if "context length" in str(e).lower():
                logger.error("FATAL: Prompt exceeded context window.")
                yield _make_event(
                    "error",
                    {"content": "Context limit reached. Please clear session."},
                    "user",
                )
                raise ContextLengthExceededError() from e
            logger.error(f"API Bad Request: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in LLM loop: {e}", exc_info=True)
            yield _make_event(
                "error",
                {"content": "An internal error occurred.", "detail": str(e)},
                "both",
            )
            raise

        yield _make_event("turn_end", {"turn_number": turn_count}, "debug")

    if not is_last_turn:
        logger.info(
            f"Reached max turns ({max_turns}) without a final answer."
        )
