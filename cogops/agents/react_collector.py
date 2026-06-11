"""
cogops/agents/react_collector.py

ReAct/TAO Information Collector — Stage 3 refinement agent.

Operates in REFINEMENT mode: starts with the results of a parallel fan-out
search (from RetrievalAgent) and searches for additional passages from new
angles until coverage is sufficient or max_turns is reached.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, AsyncGenerator, Dict, List, Optional

from openai import AsyncOpenAI

from cogops.llm.reasoning_loop import _call_llm_nonstream
from cogops.prompts.react_collector import REFINEMENT_SYSTEM_PROMPT
from cogops.prompts.time_reminder import build_time_reminder
from cogops.tools.registry import ToolContext, bind_tools, build_tool_registry

logger = logging.getLogger(__name__)

_ENABLED_TOOLS = ["jiggasha"]
_PROGRESS_MSG = "অতিরিক্ত তথ্য সংগ্রহ করা হচ্ছে…"


class ReActCollector:
    def __init__(
        self,
        primary_client: AsyncOpenAI,
        primary_model: str,
        max_turns: int = 2,
        tool_top_k: int = 25,
        tool_min_score: float = 0.35,
        timeout_seconds: float = 15.0,
        min_passages_to_stop_early: int = 3,
        stream_progress: bool = True,
    ):
        self.primary_client = primary_client
        self.primary_model = primary_model
        self.max_turns = max(max_turns, 1)
        self.tool_top_k = tool_top_k
        self.tool_min_score = tool_min_score
        self.timeout_seconds = timeout_seconds
        self.min_passages_to_stop_early = max(min_passages_to_stop_early, 1)
        self.stream_progress = stream_progress

    async def refine(
        self,
        query: str,
        initial_source_map: Dict[str, Dict[str, Any]],
        formalized_queries: List[str],
        history: Optional[List[Dict[str, str]]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Run refinement: search for additional passages beyond initial coverage.

        Yields debug events. The caller merges the final source_map back into
        the initial_source_map.
        """
        t0 = time.time()
        history = history or []
        initial_count = len(initial_source_map)

        # Fast-path: already enough passages
        if initial_count >= self.min_passages_to_stop_early:
            logger.info(
                "ReActCollector fast-path: %d passages already found (≥%d). Skipping refinement.",
                initial_count, self.min_passages_to_stop_early,
            )
            yield {
                "type": "collector_done",
                "channel": "debug",
                "passages_returned": initial_count,
                "source_map": initial_source_map,
                "elapsed_ms": 0,
                "turns_used": 0,
                "action": "fast_path_skipped",
            }
            return

        # Build tool registry
        tools_schema, raw_tools = build_tool_registry(
            enabled=_ENABLED_TOOLS,
            tool_configs={
                "jiggasha": {
                    "top_k": self.tool_top_k,
                    "min_score": self.tool_min_score,
                }
            },
        )
        ctx = ToolContext()
        available_tools = bind_tools(raw_tools, ctx)

        # Seed ToolContext with already-known passage_ids to avoid duplicates
        for tag, meta in initial_source_map.items():
            pid = meta.get("passage_id")
            if pid:
                ctx.mark_passage(pid)
            # Also copy the source_map entry so citations are valid
            ctx.source_map[tag] = dict(meta)

        # Build messages
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": REFINEMENT_SYSTEM_PROMPT},
        ]
        for msg in history[-4:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "assistant", "content": build_time_reminder()})

        # User context: question + what was already searched + what was found
        context_block = self._build_context_block(query, formalized_queries, initial_source_map)
        messages.append({"role": "user", "content": context_block})

        turn_count = 0
        progress_emitted = False

        while turn_count < self.max_turns:
            turn_count += 1
            yield {
                "type": "collector_turn_start",
                "channel": "debug",
                "turn_number": turn_count,
            }

            if self.stream_progress and not progress_emitted:
                progress_emitted = True
                yield {
                    "type": "answer_chunk",
                    "channel": "both",
                    "content": _PROGRESS_MSG,
                }

            try:
                response = await asyncio.wait_for(
                    _call_llm_nonstream(
                        client=self.primary_client,
                        model=self.primary_model,
                        messages=messages,
                        tools_schema=tools_schema,
                        extra_body={},
                        tool_choice="auto",
                    ),
                    timeout=self.timeout_seconds,
                )
            except asyncio.TimeoutError:
                logger.warning("ReActCollector turn %d timed out after %.1fs", turn_count, self.timeout_seconds)
                yield {
                    "type": "collector_error",
                    "channel": "debug",
                    "error": f"turn_timeout after {self.timeout_seconds}s",
                    "turn_number": turn_count,
                }
                break
            except Exception as e:
                logger.error("ReActCollector LLM call failed on turn %d: %s", turn_count, e)
                yield {
                    "type": "collector_error",
                    "channel": "debug",
                    "error": str(e),
                    "turn_number": turn_count,
                }
                break

            msg = response.choices[0].message
            tool_calls = msg.tool_calls
            content = msg.content or ""

            if not tool_calls:
                logger.info("ReActCollector finished after %d turn(s). source_map has %d entries.",
                            turn_count, len(ctx.source_map))
                break

            # Store assistant message with tool_calls
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
            messages.append({
                "role": "assistant",
                "content": content if content else None,
                "tool_calls": tool_calls_dicts,
            })

            # Emit tool_call debug event
            summaries = []
            for tc in tool_calls:
                raw_args = tc.function.arguments or ""
                try:
                    parsed = json.loads(raw_args) if raw_args else {}
                except (json.JSONDecodeError, TypeError):
                    parsed = {"_raw": raw_args}
                summaries.append({
                    "call_id": tc.id,
                    "name": tc.function.name,
                    "arguments": parsed,
                })
            yield {
                "type": "collector_tool_call",
                "channel": "debug",
                "tool_call_summaries": summaries,
                "turn": turn_count,
            }

            # Execute tools in parallel
            async def _execute_one(tc: Any) -> Dict[str, Any]:
                func_name = tc.function.name
                call_id = tc.id
                start = time.time()

                fn = available_tools.get(func_name)
                result_content = ""
                if fn:
                    try:
                        raw_args = tc.function.arguments or "{}"
                        try:
                            args = json.loads(raw_args)
                        except json.JSONDecodeError:
                            result_content = "Error: Invalid JSON arguments."
                            args = {}

                        if not result_content:
                            if asyncio.iscoroutinefunction(fn):
                                result = await fn(**args)
                            else:
                                result = await asyncio.to_thread(fn, **args)
                            if isinstance(result, tuple) and len(result) == 2:
                                result_content = str(result[0])
                            else:
                                result_content = str(result)
                    except Exception as e:
                        logger.error("Tool %s error: %s", func_name, e)
                        result_content = f"Error executing {func_name}: {e}"
                else:
                    result_content = f"Tool '{func_name}' not found."

                return {
                    "name": func_name,
                    "call_id": call_id,
                    "content": result_content,
                    "elapsed_ms": round((time.time() - start) * 1000),
                }

            results = await asyncio.gather(*[_execute_one(tc) for tc in tool_calls])

            for result in results:
                yield {
                    "type": "collector_tool_result",
                    "channel": "debug",
                    "call_id": result["call_id"],
                    "name": result["name"],
                    "preview": result["content"][:300],
                    "duration_ms": result["elapsed_ms"],
                }
                messages.append({
                    "tool_call_id": result["call_id"],
                    "role": "tool",
                    "name": result["name"],
                    "content": result["content"],
                })

            current_passage_count = len(ctx.source_map)
            yield {
                "type": "collector_turn_end",
                "channel": "debug",
                "turn_number": turn_count,
                "source_map_entries": current_passage_count,
            }

            # Stop early if we now have enough passages
            if current_passage_count >= self.min_passages_to_stop_early + initial_count:
                logger.info(
                    "ReActCollector early-stop: %d passages after turn %d (+%d new).",
                    current_passage_count, turn_count, current_passage_count - initial_count,
                )
                break

        elapsed_ms = int((time.time() - t0) * 1000)
        yield {
            "type": "collector_done",
            "channel": "debug",
            "passages_returned": len(ctx.source_map),
            "source_map": ctx.source_map,
            "elapsed_ms": elapsed_ms,
            "turns_used": turn_count,
        }

    def _build_context_block(
        self,
        query: str,
        formalized_queries: List[str],
        initial_source_map: Dict[str, Dict[str, Any]],
    ) -> str:
        lines = [f"User question: {query}", ""]

        if formalized_queries:
            lines.append("Sub-queries already searched:")
            for i, q in enumerate(formalized_queries, 1):
                lines.append(f"  {i}. {q}")
            lines.append("")

        if initial_source_map:
            lines.append(f"Passages already found ({len(initial_source_map)} total):")
            for tag, meta in sorted(initial_source_map.items()):
                service = meta.get("service", "")
                category = meta.get("category", "")
                text = (meta.get("text", "") or "")[:120]
                lines.append(f"  [{tag}] {category} > {service}: {text}...")
            lines.append("")
        else:
            lines.append("No passages found yet.")
            lines.append("")

        lines.append("Search for ADDITIONAL passages from new angles. Do NOT repeat the sub-queries above.")
        return "\n".join(lines)
