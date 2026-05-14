"""
cogops/tools/memory.py

Memory tools for the GovOps Agent. Read/write session-scoped memory via Redis.
These tools are discovered at runtime from the tool schema list — no need to
describe them in the system prompt.

tools: memory_read, memory_write
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _client(store: Any) -> Any:
    """Return the underlying client from a store, or the store itself."""
    return getattr(store, "_client", store)


def memory_read(
    key: Optional[str] = None,
    user_id: Optional[str] = None,
    store: Optional[Any] = None,
    **_injectable,
) -> str:
    """
    Read session memory from Redis.

    Args:
        key: Optional string — specific key to read. If empty or None, returns all memory keys.
        user_id: Session user_id (injected server-side).
        store: RedisSessionStore instance (injected server-side).
        **_injectable: Unused.

    Returns:
        Formatted memory entries, or 'No memory found.' if empty.
    """
    if store is None or not store.available:
        return "Memory store not available."

    if not user_id:
        return "Missing user_id."

    client = _client(store)
    pattern = f"session:{user_id}:memory:*"
    keys = client.keys(pattern)

    if not keys:
        return "No memory found."

    if key:
        # Read specific key
        raw = client.get(f"session:{user_id}:memory:{key}")
        if raw is None:
            return f"No memory found for key '{key}'."
        return f"Memory [{key}]:\n{raw}"

    # Read all keys
    entries: List[str] = []
    for k in sorted(keys):
        actual_key = k.replace(f"session:{user_id}:memory:", "")
        raw = client.get(k)
        if raw:
            entries.append(f"[{actual_key}]:\n{raw}")

    if not entries:
        return "No memory found."

    return "Session memory:\n\n" + "\n\n".join(entries)


def memory_write(
    key: str,
    value: str,
    user_id: Optional[str] = None,
    store: Optional[Any] = None,
    **_injectable,
) -> str:
    """
    Write session memory to Redis.

    Args:
        key: String key for the memory entry.
        value: String value to store.
        user_id: Session user_id (injected server-side).
        store: RedisSessionStore instance (injected server-side).
        **_injectable: Unused.

    Returns:
        Confirmation message.
    """
    if store is None or not store.available:
        return "Memory store not available."

    if not user_id:
        return "Missing user_id."

    if not key or not key.strip():
        return "Memory key cannot be empty."

    client = _client(store)
    full_key = f"session:{user_id}:memory:{key}"
    client.set(full_key, value)
    # Apply TTL from the store's config
    if hasattr(store, "ttl") and store.ttl:
        client.expire(full_key, store.ttl)

    return f"Memory saved under key '{key}'."


# --- Tool Schemas ---

memory_read_tools_list = [
    {
        "type": "function",
        "function": {
            "name": "memory_read",
            "description": (
                "Read session memory from Redis. Call this at the start of every turn "
                "to check if there is relevant memory from prior conversation turns. "
                "Use when you need context about what was discussed, confirmed entities, "
                "or unresolved questions. Returns all memory if no key is specified, "
                "or a specific key's value if provided."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": (
                            "Optional memory key to read. If omitted or empty, reads all memory entries."
                        ),
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    }
]

memory_write_tools_list = [
    {
        "type": "function",
        "function": {
            "name": "memory_write",
            "description": (
                "Write session memory to Redis. Write when any of these conditions are met: "
                "a named entity was confirmed (service name, form number, office name, law citation, deadline); "
                "the user corrected or confirmed a specific fact; a query reached a fully cited resolution; "
                "the conversation moved to a new topic and the previous topic should be remembered. "
                "Keep entries brief — extract core entities, confirmed outcome, and any unresolved questions. "
                "Never write uncited facts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Memory key (e.g., 'passport_query', 'nid_correction'). Use short descriptive names.",
                    },
                    "value": {
                        "type": "string",
                        "description": (
                            "Memory content: core entities, confirmed outcome, unresolved questions. "
                            "Keep it brief and factual."
                        ),
                    },
                },
                "required": ["key", "value"],
                "additionalProperties": False,
            },
        },
    }
]

memory_read_tools_map = {
    "memory_read": memory_read,
}

memory_write_tools_map = {
    "memory_write": memory_write,
}
