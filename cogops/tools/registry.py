"""
cogops/tools/registry.py

Build the tool registry: JSON schemas + name-to-callback map.
Supports dynamic/modular tool registration. The system prompt does NOT
embed tool descriptions — tools are discoverable at runtime from the
schema list returned here. This keeps the prompt stable while tools
evolve independently.

Current tools:
  - (none — tools are registered here when the agent needs external data)
"""

import inspect
import logging
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Parameters injected server-side, never surfaced to the model.
_INJECTABLE_PARAMS = ("user_id", "store")


@dataclass
class ToolContext:
    """Per-request state that context-dependent tools need."""
    user_id: Optional[str] = None
    store: Optional[Any] = None            # RedisSessionStore
    tool_map: Optional[Any] = None         # Reserved for future tools
    tools_schema: Optional[Any] = None     # Reserved for future nested tool calls


def build_tool_registry() -> Tuple[List[Dict[str, Any]], Dict[str, Callable]]:
    """
    Build the raw tool registry. Returned handlers are UNBOUND.
    Callers must pass through bind_tools(raw_map, ctx).

    The system prompt does not embed tool descriptions; tools are discovered
    at runtime from the returned schema list. This keeps the prompt stable
    while tools evolve.
    """
    all_schema: List[Dict[str, Any]] = []
    all_map: Dict[str, Callable] = {}

    # NOTE: Memory tools are intentionally disabled here.
    # When only memory_read/memory_write are available, the model
    # gets stuck in a tool-loop (calling memory for every query,
    # even simple greetings). Register tools here when the agent
    # genuinely needs external data to answer.

    # Enforce additionalProperties: false on every schema
    for spec in all_schema:
        params = spec.get("function", {}).get("parameters", {})
        params.setdefault("additionalProperties", False)

    logger.info("Tool registry built: %d tools, %d entries.", len(all_schema), len(all_map))
    return all_schema, all_map


def bind_tools(
    raw_tool_map: Dict[str, Callable],
    ctx: ToolContext,
) -> Dict[str, Callable]:
    """
    Wrap each handler with functools.partial so context-dependent tools get
    their server-side inputs. Model-visible parameters remain unbound.
    """
    ctx_dict = {
        "user_id": ctx.user_id,
        "store": ctx.store,
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


