"""
cogops/llm/reasoning_loop.py

The ReAct tool-calling while loop. Every yielded event is tagged with a channel
("user", "debug", or "both"); the API layer filters events by channel based
on whether the caller is authenticated for debug.

Behaviour guarantees:
- Final-answer text is streamed chunk-by-chunk on channel "both" as it arrives,
  with `<thinking>...</thinking>` segments stripped.
- Reasoning is emitted on channel "debug" as `reasoning_chunk` events. The
  loop picks up reasoning from either `delta.reasoning_content` (vLLM/Qwen
  field) or inline `<thinking>` tags inside `delta.content` — whichever the
  provider emits.
- `tool_result` debug events carry the `sources` list and a short `preview`
  of the retrieved context, never the full retrieved blob (which can be
  large). The model still sees the full content via the `messages` array.
- When `max_turns` is exhausted without a final answer, the user sees a
  graceful Bengali "server load" message. The actual loop-exhaustion is
  logged at CRITICAL level for ops.
"""

import json
import asyncio
import logging
import time
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Tuple

from openai import AsyncOpenAI, BadRequestError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from cogops.prompts.messages import SERVER_LOAD_FALLBACK_BN

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


class ThinkingStripper:
    """
    Streaming separator for `<thinking>...</thinking>` tags.

    Feed it text deltas via `feed()`; iterate the yielded `(channel, text)`
    pairs where channel is `"answer"` (outside tags) or `"thinking"` (inside).
    Tags themselves are consumed and never re-emitted.

    The stripper holds back up to HOLDBACK chars at the tail of its buffer
    so a tag spanning multiple deltas is detected correctly. Call `flush()`
    at end of stream to drain any buffered content.
    """

    OPEN_TAG = "<thinking>"
    CLOSE_TAG = "</thinking>"
    HOLDBACK = len(CLOSE_TAG)  # widest tag we might be in the middle of receiving

    def __init__(self):
        self.buffer = ""
        self.in_thinking = False

    def _channel(self) -> str:
        return "thinking" if self.in_thinking else "answer"

    def feed(self, text: str):
        if not text:
            return
        self.buffer += text

        while True:
            target = self.CLOSE_TAG if self.in_thinking else self.OPEN_TAG
            idx = self.buffer.find(target)

            if idx >= 0:
                before = self.buffer[:idx]
                if before:
                    yield (self._channel(), before)
                self.buffer = self.buffer[idx + len(target):]
                self.in_thinking = not self.in_thinking
                continue

            # No complete tag found. Emit the prefix that cannot possibly be
            # part of a tag, hold back the tail.
            if len(self.buffer) > self.HOLDBACK:
                safe = self.buffer[:-self.HOLDBACK]
                yield (self._channel(), safe)
                self.buffer = self.buffer[-self.HOLDBACK:]
            return

    def flush(self):
        """Drain any buffered content at end of stream."""
        if not self.buffer:
            return
        if self.in_thinking:
            logger.warning(
                "Stream ended with unclosed <thinking> tag; flushing %d "
                "buffered chars to debug channel.",
                len(self.buffer),
            )
        yield (self._channel(), self.buffer)
        self.buffer = ""


def _unpack_tool_response(
    response_data: Any,
) -> Tuple[str, List[str]]:
    """
    Normalize a tool's return value into (content_for_model, sources_for_debug).

    Tools may return:
      - A `(context, sources)` tuple where context is either a list of
        formatted strings (success path) or an error string. This is the
        contract used by `search_knowledge` and `search_wiki`.
      - A plain string (e.g. `history_query`).
      - None (treated as empty content).
    """
    if response_data is None:
        return "", []

    if isinstance(response_data, tuple) and len(response_data) == 2:
        context_part, sources_part = response_data
        if isinstance(context_part, list):
            content = "\n\n".join(str(p) for p in context_part)
        else:
            content = str(context_part)
        sources = list(sources_part) if isinstance(sources_part, list) else []
        return content, sources

    return str(response_data), []


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

    Stream-friendly: answer text is yielded as `answer_chunk` events on
    channel "both" as it arrives. Reasoning is yielded as `reasoning_chunk`
    on channel "debug". Tool calls and tool results are also "debug".
    """
    extra_body = dict(extra_body or {})

    turn_count = 0
    is_last_turn = False
    last_function_name: Optional[str] = None

    while turn_count < max_turns:
        turn_count += 1
        logger.info(f"Turn {turn_count}/{max_turns} started.")

        is_last_turn = False

        try:
            yield _make_event("turn_start", {"turn_number": turn_count}, "debug")

            stream = await client_llm.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools_schema if tools_schema else None,
                tool_choice="auto" if tools_schema else None,
                stream=True,
                extra_body=extra_body,
                **kwargs,
            )

            answer_accumulator = ""
            stripper = ThinkingStripper()
            tool_call_index_map: Dict[int, Dict[str, Any]] = {}
            in_reasoning = False  # Track native-thinking mode (Path A)

            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta

                # Path A: provider exposes reasoning as a separate field.
                reasoning_delta = getattr(delta, "reasoning_content", None)
                if reasoning_delta:
                    in_reasoning = True
                    yield _make_event(
                        "reasoning_chunk", {"content": reasoning_delta}, "debug"
                    )
                    # DO NOT add reasoning_content to answer_accumulator — it
                    # must never leak into the user-facing answer.
                    continue

                # If we are currently in native-thinking mode, hold back any
                # delta.content until the thinking ends.
                if in_reasoning:
                    # The model switches from thinking to answer when
                    # reasoning_content stops arriving. We detect the end
                    # of thinking when content arrives but reasoning_content
                    # is empty (or we hit a tool call / stop).
                    # Qwen36 signals end-of-thinking with a special token or
                    # by simply stopping reasoning_content. For safety, we
                    # treat all content as answer once reasoning_content
                    # stops for a chunk that has content.
                    if delta.content:
                        in_reasoning = False
                        # Fall through to Path B to process this content.
                    else:
                        continue

                # Path B: reasoning is inline in content via <thinking> tags.
                if delta.content:
                    for channel, piece in stripper.feed(delta.content):
                        if channel == "answer":
                            answer_accumulator += piece
                            yield _make_event(
                                "answer_chunk", {"content": piece}, "both"
                            )
                        else:
                            yield _make_event(
                                "reasoning_chunk", {"content": piece}, "debug"
                            )

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

            # Drain any tail content the stripper held back.
            for channel, piece in stripper.flush():
                if channel == "answer":
                    answer_accumulator += piece
                    yield _make_event("answer_chunk", {"content": piece}, "both")
                else:
                    yield _make_event("reasoning_chunk", {"content": piece}, "debug")

            # Build the assistant message to feed back in. The content stored
            # in history is the answer-only portion (thinking is stripped).
            response_message: Dict[str, Any] = {
                "role": "assistant",
                "content": answer_accumulator if answer_accumulator else None,
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

            # --- Execute the tools in parallel ---
            logger.info(f"Executing {len(tool_calls_list)} tool(s)...")

            async def execute_tool(tool_call: Dict[str, Any]) -> Dict[str, Any]:
                function_name = tool_call["function"]["name"]
                call_id = tool_call["id"]
                start_time = time.time()

                tool_result_content = ""
                sources_for_debug: List[str] = []

                function_to_call = available_tools.get(function_name)
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
                                response_data = await asyncio.to_thread(
                                    function_to_call, **function_args
                                )
                            tool_result_content, sources_for_debug = _unpack_tool_response(
                                response_data
                            )

                    except Exception as e:
                        logger.error(f"Error executing {function_name}: {e}", exc_info=True)
                        tool_result_content = f"System Error executing tool: {str(e)}"
                else:
                    tool_result_content = (
                        f"Error: Tool '{function_name}' not defined in available_tools_map."
                    )

                elapsed_ms = round((time.time() - start_time) * 1000)
                return {
                    "name": function_name,
                    "call_id": call_id,
                    "content": tool_result_content,
                    "sources": sources_for_debug,
                    "elapsed_ms": elapsed_ms,
                }

            results = await asyncio.gather(*[execute_tool(tc) for tc in tool_calls_list])

            for result in results:
                last_function_name = result["name"]
                yield _make_event(
                    "tool_result",
                    {
                        "call_id": result["call_id"],
                        "name": result["name"],
                        "sources": result["sources"],
                        "preview": result["content"][:200],
                        "duration_ms": result["elapsed_ms"],
                        "status": "error" if result["content"].startswith("Error") else "ok",
                    },
                    "debug",
                )

                messages.append({
                    "tool_call_id": result["call_id"],
                    "role": "tool",
                    "name": result["name"],
                    "content": result["content"],
                })

        except BadRequestError as e:
            if "context length" in str(e).lower():
                logger.critical(
                    "Prompt exceeded model context window on turn %d. "
                    "Last tool=%s. Returning server-load fallback to user.",
                    turn_count,
                    last_function_name,
                )
                yield _make_event(
                    "answer_chunk", {"content": SERVER_LOAD_FALLBACK_BN}, "both"
                )
                raise ContextLengthExceededError() from e
            logger.error(f"API Bad Request: {e}")
            # Re-raise so orchestrator handles it (yields error + fallback)
            raise RuntimeError(f"API Bad Request: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error in LLM loop: {e}", exc_info=True)
            yield _make_event(
                "error",
                {"content": "An internal error occurred.", "detail": str(e)},
                "debug",
            )
            yield _make_event(
                "answer_chunk", {"content": SERVER_LOAD_FALLBACK_BN}, "both"
            )
            raise

        yield _make_event("turn_end", {"turn_number": turn_count}, "debug")

    if not is_last_turn:
        logger.critical(
            "ReAct loop exhausted max_turns=%d without a final answer. "
            "Last tool=%s, partial answer length=%d chars.",
            max_turns,
            last_function_name,
            len(answer_accumulator) if 'answer_accumulator' in locals() else 0,
        )
        yield _make_event(
            "answer_chunk", {"content": SERVER_LOAD_FALLBACK_BN}, "both"
        )
