# Events (`cogops/events/`)

The event system defines the contract between the pipeline and the UI / debug consumers.

---

## `types.py`

**Role:** Event type constants and factory.

**Why it exists:**
- Defines the `Channel` enum (`user`, `debug`, `both`).
- Provides an `event()` factory that creates consistently-shaped event dicts.
- Every agent in the pipeline yields events through this type system so the UI knows what to render and the debugger knows what to log.

---

## `channels.py`

**Role:** Event channel filtering helpers.

**Why it exists:**
- `filter_for_user(events)` — strips debug-only events before sending to the frontend.
- `filter_for_debug(events)` — keeps everything for the audit log.
- Centralizing filter logic prevents the UI from accidentally leaking debug prompts or thinking blocks to end users.
