"""
cogops/events/channels.py

Channel filtering: filter_for_user(), filter_for_debug()
"""

from typing import Iterable, Dict, Any


def filter_for_user(events: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    """Yield only events visible to the user (channel=user or channel=both)."""
    for evt in events:
        ch = evt.get("channel", "user")
        if ch in ("user", "both"):
            yield evt


def filter_for_debug(events: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    """Yield only debug-visible events (channel=debug or channel=both)."""
    for evt in events:
        ch = evt.get("channel", "user")
        if ch in ("debug", "both"):
            yield evt


def strip_channel(events: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    """Return events with the channel field removed (for external consumers)."""
    for evt in events:
        return_val = dict(evt)
        return_val.pop("channel", None)
        yield return_val
