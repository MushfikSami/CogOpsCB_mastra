# Config (`cogops/config/`)

Configuration loading and endpoint credential management.

---

## `loader.py`

**Role:** YAML config loader and `EndpointConfig` dataclass.

**Why it exists:**
- Reads the main `configs/config.yml` file.
- Provides `EndpointConfig` dataclass that reads LLM endpoint credentials from environment variables (not hardcoded).
- Supports nested override so deployment-specific values (thresholds, timeouts, policy) can be changed without touching code.

**Key design choice:** Environment-variable-driven credentials mean the same Docker image runs in dev, staging, and prod with only env vars changing.
