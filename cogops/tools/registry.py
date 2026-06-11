"""
cogops/tools/registry.py

Plug-in tool registry. Tools are discovered at runtime from a config-driven
enabled list. Each tool module under cogops/tools/ exports:

    NAME: str                                       # function name the LLM calls
    DESCRIPTION: str                                # 'when to use' guidance (LLM-facing)
    SCHEMA: dict                                    # OpenAI-style tool schema
    async def handler(**kwargs) -> tuple[str, list] # returns (content_for_model, sources)

Adding a new tool: drop a module in cogops/tools/, list its slug in
configs/config.yml under tools.enabled.

Server-side params (user_id, store, ctx) are injected via bind_tools() so they
never appear in the LLM-visible parameter schema. Tool handlers may declare any
subset of these as keyword params; bind_tools wires them in via functools.partial.
"""

import importlib
import inspect
import logging
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Parameters injected server-side, never surfaced to the model schema.
_INJECTABLE_PARAMS = ("user_id", "store", "ctx")


@dataclass
class ToolContext:
    """Per-request state shared across all tool calls in a single turn."""
    user_id: Optional[str] = None
    store: Optional[Any] = None                                   # RedisSessionStore / InMemoryStore
    source_map: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # {"S1": {...}, "S2": {...}}
    seen_passage_ids: set = field(default_factory=set)            # dedup across turns
    tool_map: Optional[Any] = None                                # reserved
    tools_schema: Optional[Any] = None                            # reserved

    def allocate_source_tag(self, payload: Dict[str, Any]) -> str:
        """Allocate the next S# tag and store the source payload under it.

        Monotonic across the whole turn so the LLM sees consistent numbering
        regardless of how many tool calls happen. Returns the tag string (e.g. "S3").
        """
        next_num = len(self.source_map) + 1
        tag = f"S{next_num}"
        self.source_map[tag] = payload
        return tag

    def has_passage(self, passage_id: Any) -> bool:
        """Check if a passage_id has already been seen in this turn."""
        return passage_id in self.seen_passage_ids

    def mark_passage(self, passage_id: Any) -> None:
        """Mark a passage_id as seen."""
        if passage_id is not None:
            self.seen_passage_ids.add(passage_id)


def build_tool_registry(
    enabled: Optional[List[str]] = None,
    tool_configs: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Callable]]:
    """Discover and load tools listed in `enabled`.

    Args:
        enabled: list of tool slugs to load. Each slug must match a module name
                 importable as `cogops.tools.<slug>`. If None or empty, returns
                 empty lists (no tools — agent falls back to no-tool streaming path).
        tool_configs: optional per-tool config dict, surfaced to modules that
                      expose a `configure(cfg)` hook. Keyed by slug.

    Returns:
        (tools_schema, raw_tool_map) — schemas to send to the LLM, and a name→handler
        dict (handlers UNBOUND — callers must run through bind_tools()).
    """
    enabled = enabled or []
    tool_configs = tool_configs or {}

    all_schema: List[Dict[str, Any]] = []
    all_map: Dict[str, Callable] = {}

    for slug in enabled:
        try:
            module = importlib.import_module(f"cogops.tools.{slug}")
        except ImportError as e:
            logger.error("Tool plugin '%s' could not be imported: %s", slug, e)
            raise

        try:
            name = getattr(module, "NAME")
            schema = getattr(module, "SCHEMA")
            handler = getattr(module, "handler")
        except AttributeError as e:
            raise RuntimeError(
                f"Tool plugin '{slug}' is missing required exports (NAME, SCHEMA, handler): {e}"
            ) from e

        configure_fn = getattr(module, "configure", None)
        if callable(configure_fn) and slug in tool_configs:
            configure_fn(tool_configs[slug])

        params = schema.get("function", {}).get("parameters", {})
        params.setdefault("additionalProperties", False)

        all_schema.append(schema)
        all_map[name] = handler

    logger.info("Tool registry built: %d tools loaded (%s).", len(all_schema), enabled)
    return all_schema, all_map


def bind_tools(
    raw_tool_map: Dict[str, Callable],
    ctx: ToolContext,
) -> Dict[str, Callable]:
    """Wrap each handler with functools.partial so server-side inputs are injected.

    Only parameters whose names appear in _INJECTABLE_PARAMS are injected, and only
    if the handler's signature declares them. Model-visible parameters remain unbound.
    """
    ctx_dict: Dict[str, Any] = {
        "user_id": ctx.user_id,
        "store": ctx.store,
        "ctx": ctx,
    }

    bound: Dict[str, Callable] = {}
    for name, fn in raw_tool_map.items():
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            bound[name] = fn
            continue

        inject = {
            k: ctx_dict[k]
            for k in _INJECTABLE_PARAMS
            if k in sig.parameters and ctx_dict.get(k) is not None
        }
        bound[name] = partial(fn, **inject) if inject else fn

    return bound
