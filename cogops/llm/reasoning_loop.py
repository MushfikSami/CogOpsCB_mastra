"""
cogops/llm/reasoning_loop.py

The tool-calling while loop. Every yielded event is tagged with a channel
("user", "debug", or "both"); the API layer filters events by channel based
on whether the caller is authenticated for debug.

Behaviour guarantees:
- The primary LLM MUST call some tool before any user-visible answer. On
  turn 1 `tool_choice="required"` makes the model pick one (a real info
  tool, or the `answer_directly` meta-tool for chit-chat / identity /
  safety). After turn 1 `tool_choice="auto"` lets it stop when done.
- `answer_directly` short-circuits: its result string carries a sentinel
  that tells the loop to stream the user-facing text and end immediately,
  without feeding the tool result back for another primary-LLM pass.
- Content deltas are streamed chunk-by-chunk on channel "user" as they
  arrive. No end-of-turn buffering.
- Large tool results (over a configurable token threshold) are condensed
  through the secondary LLM before being fed back to the primary. Debug
  stream sees both raw and refined content via `tool_result_refined`.
"""

import os
import json
import asyncio
import logging
import time
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional

from openai import AsyncOpenAI, BadRequestError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from cogops.tools.answer_directly import ANSWER_DIRECTLY_SENTINEL

# Streaming cadence for answer_directly text. The underlying model has
# already produced the full string synchronously, so we reveal it to the
# user in small pieces to match the chunk-by-chunk feel of normal answers.
_DIRECT_STREAM_CHARS_PER_CHUNK = 12
_DIRECT_STREAM_DELAY_SECONDS = 0.015

RETRYABLE_EXCEPTIONS = (ConnectionError, TimeoutError, RuntimeError)

_MISSING = object()

MAX_TURNS = 10

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


def _parse_direct_answer(tool_result: str):
    """Return (category, text) if this is an answer_directly result, else None."""
    if not isinstance(tool_result, str) or not tool_result.startswith(ANSWER_DIRECTLY_SENTINEL + "::"):
        return None
    try:
        _, category, text = tool_result.split("::", 2)
    except ValueError:
        return None
    return category, text


async def _refine_tool_output(
    secondary_client,
    secondary_model: str,
    user_query: str,
    tool_name: str,
    raw: str,
    max_tokens: int = 1024,
) -> str:
    """Ask the secondary LLM to extract query-relevant bits from a bulky tool result."""
    from cogops.llm.secondary import call_secondary

    prompt = (
        "Extract only the parts of the tool output below that are relevant to the "
        "user's query. Keep the same language as the raw output.\n\n"
        f"User query: {user_query}\n"
        f"Tool: {tool_name}\n"
        f"Raw output:\n{raw}\n\n"
        "Return a tight, structured excerpt. If nothing is relevant, return exactly: "
        "NO_RELEVANT_DATA."
    )
    return await call_secondary(
        secondary_client,
        secondary_model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.2,
    )


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
    max_turns: int = MAX_TURNS,
    extra_body: Optional[Dict[str, Any]] = None,
    user_query: str = "",
    secondary_client=None,
    secondary_model: str = "",
    tokenizer=None,
    refine_threshold_tokens: int = 0,
    **kwargs: Any,
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Orchestrates the conversation using the primary LLM endpoint.

    Stream-friendly: text deltas are yielded as `answer_chunk` events on
    channel "user" as they arrive. Reasoning, tool calls, tool results,
    and token usage are yielded on channel "debug".

    Args:
        user_query: the original user turn (used by the post-tool refiner
                    to decide relevance).
        secondary_client / secondary_model / tokenizer / refine_threshold_tokens:
                    control the post-tool secondary-LLM refine step. If any
                    is missing or `refine_threshold_tokens<=0`, refine is off.
    """
    extra_body = dict(extra_body or {})
    vllm_params = ("repetition_penalty", "top_k", "top_p")
    for param in vllm_params:
        if param in kwargs:
            extra_body[param] = kwargs.pop(param)

    # Capture usage in the final streaming chunk.
    extra_body.setdefault("stream_options", {"include_usage": True})

    post_refine_enabled = bool(
        secondary_client and secondary_model and tokenizer and refine_threshold_tokens > 0
    )

    turn_count = 0
    reasoning_accumulator = ""
    is_last_turn = False

    while turn_count < max_turns:
        turn_count += 1
        logger.info(f"Turn {turn_count}/{max_turns} started.")

        is_last_turn = False

        try:
            yield _make_event("turn_start", {"turn_number": turn_count}, "debug")

            # Turn 1: force the model to pick SOME tool. It can freely choose
            # any real info tool or the answer_directly meta-tool for
            # chit-chat/identity/safety. This preserves "never answer from
            # parametric knowledge on factual queries" without pinning the
            # model to graph_search.
            # Turn 2+: the model may stop when it has enough context.
            if tools_schema and turn_count <= 1:
                tool_choice_val = "required"
            else:
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
            stream_usage: Dict[str, Any] = {}
            # Content deltas are streamed live; if this turn also emits
            # tool_calls (rare with vLLM+Qwen), the streamed content becomes
            # part of the assistant message but is not the final answer. We
            # flag that so the caller can ignore it.
            streamed_content_this_turn = False

            async for chunk in stream:
                # Special vLLM usage chunk.
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

                # Native thinking channel (Qwen3 vLLM parser).
                reasoning = getattr(delta, "reasoning", _MISSING)
                if reasoning is not _MISSING and reasoning:
                    reasoning_accumulator += reasoning
                    yield _make_event("reasoning_chunk", {"data": reasoning}, "debug")
                    continue

                # Stream user-facing content as it arrives.
                if delta.content:
                    content_chunk = delta.content
                    full_content_accumulator += content_chunk
                    streamed_content_this_turn = True
                    yield _make_event(
                        "answer_chunk", {"content": content_chunk}, "user"
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

            if reasoning_accumulator:
                # No final-flush event — each reasoning chunk was already
                # streamed above; clear the accumulator for the next turn.
                reasoning_accumulator = ""

            if stream_usage:
                yield _make_event("usage", {"tokens": stream_usage}, "debug")

            # Build the assistant message to feed back in.
            response_message: Dict[str, Any] = {
                "role": "assistant",
                "content": full_content_accumulator if full_content_accumulator else None,
            }

            tool_calls_list = list(tool_call_index_map.values())
            if tool_calls_list:
                response_message["tool_calls"] = tool_calls_list
                if streamed_content_this_turn:
                    # Content should not have been user-facing on a tool-calling
                    # turn. Mark it in debug; user already saw the chunks, which
                    # is tolerable in practice (vLLM+Qwen don't interleave).
                    logger.debug(
                        "Content streamed on a tool-calling turn — should be rare."
                    )

            messages.append(response_message)

            if not tool_calls_list:
                # Model produced a final answer without a tool call.
                # Because we force tool_choice="required" on turn 1, this only
                # happens on turn 2+ when the model has enough tool context
                # and is wrapping up.
                if turn_count <= 1 and tools_schema:
                    # Shouldn't happen (tool_choice=required). Defensive reminder.
                    logger.warning(
                        "No tool call emitted on turn 1 despite tool_choice='required'."
                    )
                    if messages and messages[0].get("role") == "system":
                        messages[0]["content"] += (
                            "\n\n[SYSTEM REMINDER: You MUST call a tool. For "
                            "chit-chat / identity / safety use `answer_directly`. "
                            "For any factual question use an information tool.]"
                        )
                    continue
                is_last_turn = True
                break

            yield _make_event(
                "tool_call",
                {"tool_calls": tool_calls_list, "turn": turn_count},
                "debug",
            )

            # --- Execute the tools ---
            logger.info(f"Executing {len(tool_calls_list)} tool(s)...")

            direct_answer_payload = None

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
                                # functools.partial around a coroutine still reports False for iscoroutinefunction;
                                # detect the underlying function.
                                inner = getattr(function_to_call, "func", function_to_call)
                                if asyncio.iscoroutinefunction(inner):
                                    response_data = await function_to_call(**function_args)
                                else:
                                    response_data = await asyncio.to_thread(function_to_call, **function_args)
                            tool_result_content = str(response_data) if response_data is not None else ""

                    except Exception as e:
                        from cogops.tools.ask_user import ClarificationRequested
                        if isinstance(e, ClarificationRequested):
                            raise
                        logger.error(f"Error executing {function_name}: {e}", exc_info=True)
                        tool_result_content = f"System Error executing tool: {str(e)}"
                else:
                    tool_result_content = (
                        f"Error: Tool '{function_name}' not defined in available_tools_map."
                    )

                # Detect answer_directly short-circuit.
                direct = _parse_direct_answer(tool_result_content)
                if direct is not None:
                    direct_answer_payload = {
                        "call_id": call_id,
                        "function_name": function_name,
                        "category": direct[0],
                        "text": direct[1],
                    }
                    # Don't refine, don't feed back.
                    yield _make_event(
                        "tool_result",
                        {
                            "call_id": call_id,
                            "content": f"[answer_directly:{direct[0]}] "
                                       f"(streamed to user directly)",
                            "duration_ms": round((time.time() - start_time) * 1000),
                            "status": "ok",
                        },
                        "debug",
                    )
                    # Still append a synthetic tool message so the transcript is
                    # well-formed (not strictly needed since we break next).
                    messages.append({
                        "tool_call_id": call_id,
                        "role": "tool",
                        "name": function_name,
                        "content": tool_result_content,
                    })
                    break  # out of the tool loop

                # Optional secondary-LLM refine for large results.
                if (
                    post_refine_enabled
                    and not tool_result_content.startswith("Error")
                    and not tool_result_content.startswith("System Error")
                    and tokenizer is not None
                    and tokenizer.count(tool_result_content) > refine_threshold_tokens
                ):
                    raw = tool_result_content
                    try:
                        refined = await _refine_tool_output(
                            secondary_client=secondary_client,
                            secondary_model=secondary_model,
                            user_query=user_query,
                            tool_name=function_name,
                            raw=raw,
                        )
                    except Exception as e:
                        logger.warning(f"Refine failed for {function_name}: {e}")
                        refined = ""

                    if refined and refined.strip() and refined.strip() != "NO_RELEVANT_DATA":
                        tool_result_content = refined
                        yield _make_event(
                            "tool_result_refined",
                            {
                                "call_id": call_id,
                                "raw": raw,
                                "refined": tool_result_content,
                            },
                            "debug",
                        )
                    else:
                        yield _make_event(
                            "tool_result_refined",
                            {
                                "call_id": call_id,
                                "raw": raw,
                                "refined": raw,
                                "note": "refine returned no relevant data; keeping raw",
                            },
                            "debug",
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

            # Short-circuit after streaming the direct answer.
            if direct_answer_payload is not None:
                yield _make_event(
                    "direct_answer",
                    {
                        "category": direct_answer_payload["category"],
                        "call_id": direct_answer_payload["call_id"],
                    },
                    "debug",
                )
                # Emit the text as small chunks so the UI renders it as a
                # live stream rather than a single blob.
                text = direct_answer_payload["text"]
                step = _DIRECT_STREAM_CHARS_PER_CHUNK
                for i in range(0, len(text), step):
                    yield _make_event(
                        "answer_chunk",
                        {"content": text[i:i + step]},
                        "user",
                    )
                    if _DIRECT_STREAM_DELAY_SECONDS > 0:
                        await asyncio.sleep(_DIRECT_STREAM_DELAY_SECONDS)
                is_last_turn = True
                yield _make_event("turn_end", {"turn_number": turn_count}, "debug")
                return

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
            from cogops.tools.ask_user import ClarificationRequested
            if isinstance(e, ClarificationRequested):
                logger.info("ClarificationRequested raised by tool.")
                yield _make_event(
                    "clarification_needed",
                    {
                        "question": e.question,
                        "options": e.options,
                        "reason": e.reason,
                        "turn_id": e.turn_id,
                    },
                    "both",
                )
                return

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
        max_tokens=1,
    )
    reranker = QwenRerankerClient(client=client_reranker, config=reranker_llm_config)
    return await reranker.rank(query, passages)
