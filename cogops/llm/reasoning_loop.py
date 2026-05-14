"""
cogops/llm/reasoning_loop.py

The ReAct tool-calling while loop. Two-phase approach:
- Phase 1: Non-streaming call with tool_choice="required" to force tool calls
- Phase 2: Streaming call for final answer (when no tool calls)
- Tool results appended as messages between turns
"""

import asyncio
import json
import logging
import time
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional

from openai import AsyncOpenAI, BadRequestError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from cogops.prompts.messages import SERVER_LOAD_FALLBACK_BN
from cogops.utils.thinking_parser import ThinkingParser

RETRYABLE = (ConnectionError, TimeoutError, RuntimeError)
_DEFAULT_MAX_TURNS = 10

logger = logging.getLogger(__name__)


def _make_event(event_type: str, data: dict, channel: str) -> Dict[str, Any]:
    evt = {"type": event_type, "channel": channel}
    evt.update(data)
    return evt


def _unpack_tool_response(response_data: Any) -> tuple:
    """Normalize tool return into (content_for_model, sources_for_debug)."""
    if response_data is None:
        return "", []

    if isinstance(response_data, tuple) and len(response_data) == 2:
        context_part, sources_part = response_data
        content = "\n\n".join(str(p) for p in context_part) if isinstance(context_part, list) else str(context_part)
        sources = list(sources_part) if isinstance(sources_part, list) else []
        return content, sources

    return str(response_data), []


@retry(
    wait=wait_exponential(multiplier=1, min=1, max=10),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type(RETRYABLE),
)
async def _call_llm_nonstream(
    client: AsyncOpenAI,
    model: str,
    messages: List[Dict],
    tools_schema: List[Dict],
    extra_body: Dict,
) -> Any:
    """Non-streaming call with required tool choice."""
    return await client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools_schema if tools_schema else None,
        tool_choice="required" if tools_schema else None,
        stream=False,
        extra_body=extra_body,
    )


@retry(
    wait=wait_exponential(multiplier=1, min=1, max=10),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type(RETRYABLE),
)
async def _call_llm_stream(
    client: AsyncOpenAI,
    model: str,
    messages: List[Dict],
    extra_body: Dict,
) -> Any:
    """Streaming call for final answer (no tools)."""
    return await client.chat.completions.create(
        model=model,
        messages=messages,
        tools=None,
        tool_choice=None,
        stream=True,
        extra_body=extra_body,
    )


@retry(
    wait=wait_exponential(multiplier=1, min=1, max=10),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type(RETRYABLE),
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
    Orchestrates the ReAct conversation.

    Strategy: Use non-streaming calls with tool_choice="required" for tool-calling turns.
    This avoids the Qwen36 bug where streaming mode produces inline text instead of tool calls.
    When the model produces no tool calls (final answer), stream the response.
    """
    extra_body = dict(extra_body or {})
    turn_count = 0
    answer_accumulator = ""
    last_function_name: Optional[str] = None

    while turn_count < max_turns:
        turn_count += 1

        try:
            yield _make_event("turn_start", {"turn_number": turn_count}, "debug")

            # --- Non-streaming call with required tool choice (only when tools exist) ---
            try:
                if tools_schema:
                    response = await _call_llm_nonstream(
                        client_llm, model, messages, tools_schema, extra_body,
                    )
                else:
                    # No tools: stream directly for real-time token-by-token output
                    response = await _call_llm_stream(
                        client_llm, model, messages, extra_body,
                    )

            except Exception:
                yield _make_event("answer_chunk", {"content": SERVER_LOAD_FALLBACK_BN}, "both")
                raise

            if tools_schema:
                # Non-streaming path: get message directly
                msg = response.choices[0].message
                tool_calls = msg.tool_calls
                content = msg.content or ""
            else:
                # Streaming path: accumulate text from streaming chunks
                msg_content_parts = []
                thinking_content_parts = []
                tool_calls = []
                content = ""
                parser = ThinkingParser()
                async for chunk in response:
                    delta = chunk.choices[0].delta
                    text = delta.content or ""
                    if not text:
                        continue
                    for channel, piece in parser.feed(text):
                        if channel == "answer":
                            msg_content_parts.append(piece)
                            answer_accumulator += piece
                            yield _make_event("answer_chunk", {"content": piece}, "both")
                        else:
                            thinking_content_parts.append(piece)
                            yield _make_event("reasoning_chunk", {"content": piece}, "debug")
                for channel, piece in parser.flush():
                    if channel == "answer":
                        msg_content_parts.append(piece)
                        answer_accumulator += piece
                        yield _make_event("answer_chunk", {"content": piece}, "both")
                    else:
                        thinking_content_parts.append(piece)
                content = "".join(msg_content_parts)

            # --- No tool calls = final answer ---
            if not tool_calls:
                messages.append({"role": "assistant", "content": content})
                break

            # --- Tool calls detected ---
            tool_calls_dicts = []
            for tc in tool_calls:
                tool_calls_dicts.append({
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments or "",
                    },
                })

            # Store assistant message
            response_msg: Dict[str, Any] = {
                "role": "assistant",
                "content": content if content else None,
                "tool_calls": tool_calls_dicts,
            }
            messages.append(response_msg)

            # Emit reasoning/thinking content (if any)
            if content:
                parser = ThinkingParser()
                for channel, piece in parser.feed(content):
                    if channel == "answer":
                        answer_accumulator += piece
                        yield _make_event("answer_chunk", {"content": piece}, "both")
                    else:
                        yield _make_event("reasoning_chunk", {"content": piece}, "debug")

            # Emit tool_call event
            tool_call_summaries = []
            for tc in tool_calls:
                raw_args = tc.function.arguments or ""
                try:
                    parsed = json.loads(raw_args) if raw_args else {}
                except (json.JSONDecodeError, TypeError):
                    parsed = {"_raw": raw_args}
                tool_call_summaries.append({
                    "call_id": tc.id,
                    "name": tc.function.name,
                    "arguments": parsed,
                })

            yield _make_event("tool_call", {
                "tool_call_summaries": tool_call_summaries,
                "turn": turn_count,
            }, "debug")

            # --- Execute tools in parallel ---
            logger.info("Executing %d tool(s)...", len(tool_calls))

            async def execute_tool(tc: Any) -> Dict[str, Any]:
                func_name = tc.function.name
                call_id = tc.id
                start = time.time()

                content = ""
                sources: List[str] = []

                fn = available_tools.get(func_name)
                if fn:
                    try:
                        raw_args = tc.function.arguments or "{}"
                        try:
                            args = json.loads(raw_args)
                        except json.JSONDecodeError:
                            content = "Error: Invalid JSON arguments from model."
                            args = {}

                        if not content:
                            if asyncio.iscoroutinefunction(fn):
                                result = await fn(**args)
                            else:
                                result = await asyncio.to_thread(fn, **args)
                            content, sources = _unpack_tool_response(result)

                    except Exception as e:
                        logger.error("Error executing %s: %s", func_name, e, exc_info=True)
                        content = f"Error executing {func_name}: {e}"
                else:
                    content = f"Tool '{func_name}' not found."

                elapsed = round((time.time() - start) * 1000)

                return {
                    "name": func_name,
                    "call_id": call_id,
                    "content": content,
                    "sources": sources,
                    "elapsed_ms": elapsed,
                }

            results = await asyncio.gather(*[execute_tool(tc) for tc in tool_calls])

            for result in results:
                last_function_name = result["name"]

                yield _make_event("tool_result", {
                    "call_id": result["call_id"],
                    "name": result["name"],
                    "sources": result["sources"],
                    "preview": result["content"][:300],
                    "duration_ms": result["elapsed_ms"],
                    "status": "error" if result["content"].startswith("Error") else "ok",
                }, "debug")

                messages.append({
                    "tool_call_id": result["call_id"],
                    "role": "tool",
                    "name": result["name"],
                    "content": result["content"],
                })

        except BadRequestError as e:
            if "context length" in str(e).lower():
                logger.critical("Context length exceeded on turn %d. Last tool=%s.", turn_count, last_function_name)
                yield _make_event("answer_chunk", {"content": SERVER_LOAD_FALLBACK_BN}, "both")
                raise
            logger.error("Bad Request: %s", e)
            raise RuntimeError(f"Bad Request: {e}") from e
        except Exception as e:
            logger.error("Unexpected error in LLM loop: %s", e, exc_info=True)
            yield _make_event("error", {"content": "Internal error.", "detail": str(e)}, "debug")
            yield _make_event("answer_chunk", {"content": SERVER_LOAD_FALLBACK_BN}, "both")
            raise

        yield _make_event("turn_end", {"turn_number": turn_count}, "debug")

    # If we exhausted turns without a final answer
    if turn_count >= max_turns:
        logger.critical("Exhausted max_turns=%d without final answer.", max_turns)
        yield _make_event("answer_chunk", {"content": SERVER_LOAD_FALLBACK_BN}, "both")
