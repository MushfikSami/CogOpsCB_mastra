"""
api.py — FastAPI proxy for GovOps Chatbot.

The agent orchestration now lives in the Mastra sidecar (Node/TypeScript, see
./mastra). This FastAPI layer is a thin proxy that:
  - serves the single-page UI (unchanged),
  - forwards POST /chat/stream to the Mastra service and relays its NDJSON stream,
  - applies the same X-Debug-Key gate + channel filtering as before,
  - keeps the query log.

Endpoints:
  POST /chat/stream  — Main streaming chat (NDJSON), proxied to Mastra
  GET  /health       — Service health (proxy + Mastra reachability)
  POST /session/clear — No-op acknowledgement (memory is owned by Mastra/LibSQL)
  GET  /query-log    — Last N query entries
"""

import json
import logging
import os
import uuid
from typing import Optional

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from cogops.events.channels import filter_for_debug, filter_for_user
from cogops.session.logger import QuerySessionLogger

load_dotenv()

# --- Config ---
API_PORT = int(os.getenv("API_PORT", "9000"))
API_HOST = "0.0.0.0"
MASTRA_URL = os.getenv("MASTRA_URL", "http://localhost:9100").rstrip("/")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("api")

# --- App ---
app = FastAPI(title="GovOps API", description="Government Services AI Agent", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Singleton logger ---
_session_logger = QuerySessionLogger()

# --- UI ---
_UI_INDEX_PATH = os.path.join(os.path.dirname(__file__), "cogops", "ui", "index.html")
try:
    with open(_UI_INDEX_PATH, "r", encoding="utf-8") as _f:
        _UI_INDEX_HTML = _f.read()
except FileNotFoundError:
    _UI_INDEX_HTML = "<h1>UI bundle missing</h1>"
    logger.warning("UI bundle not found at %s", _UI_INDEX_PATH)


# --- Request models ---
class ChatRequest(BaseModel):
    user_id: str
    query: str


class SessionRequest(BaseModel):
    user_id: str


# --- Endpoints ---
@app.get("/ui", response_class=HTMLResponse)
async def ui_root():
    """Single-page chat UI."""
    return HTMLResponse(content=_UI_INDEX_HTML)


@app.get("/health")
async def health_check():
    """Service health: proxy + Mastra sidecar reachability."""
    status = {"status": "online", "mastra_url": MASTRA_URL}
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{MASTRA_URL}/health")
            status["mastra"] = "ok" if resp.status_code == 200 else f"http_{resp.status_code}"
    except Exception:
        status["mastra"] = "error"
    return status


@app.post("/chat/stream")
async def stream_chat(
    request: ChatRequest,
    x_debug_key: Optional[str] = Header(None, alias="X-Debug-Key"),
):
    """Main chat endpoint. Proxies to the Mastra sidecar and relays NDJSON.

    Without a valid `X-Debug-Key` header, the stream is filtered to
    user-visible events only. With the correct key, all debug events pass through.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    _session_logger.append_query(request.query)
    server_secret = os.getenv("ADMIN_DEBUG_SECRET")
    debug_mode = False
    if x_debug_key:
        if server_secret == "":
            debug_mode = True
        else:
            debug_mode = x_debug_key == server_secret

    _session_logger.start_session(request.user_id, request.query)
    request_id = str(uuid.uuid4())[:8]
    logger.info("[req:%s] Chat request from %s → %s", request_id, request.user_id, MASTRA_URL)

    async def event_generator():
        payload = {"user_id": request.user_id, "query": request.query}
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, read=None)) as client:
                async with client.stream(
                    "POST", f"{MASTRA_URL}/chat/stream", json=payload
                ) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            event = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        _session_logger.ingest_event(event)
                        filtered = filter_for_debug([event]) if debug_mode else filter_for_user([event])
                        for evt in filtered:
                            yield json.dumps(evt, ensure_ascii=True) + "\n"
        except Exception as e:
            logger.error("[req:%s] Proxy stream error: %s", request_id, e, exc_info=True)
            error_evt = {"type": "error", "content": "Streaming error.", "channel": "user"}
            _session_logger.ingest_event(error_evt)
            yield json.dumps(error_evt, ensure_ascii=True) + "\n"

        _session_logger.finalize_session(request.user_id)

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")


@app.get("/query-log")
async def query_log_endpoint(limit: int = 500):
    """Return stored queries (last 10 days)."""
    return _session_logger.get_queries(limit=limit)


@app.post("/session/clear")
async def clear_session(request: SessionRequest):
    """Clear conversation for a user.

    Conversational memory is owned by the Mastra sidecar (LibSQL threads keyed by
    user_id). This endpoint is retained for UI compatibility; to hard-clear a
    thread, delete it via Mastra's memory API. Returns success for the UI flow.
    """
    return {"status": "success", "message": "Session reset acknowledged."}


if __name__ == "__main__":
    uvicorn.run(app, host=API_HOST, port=API_PORT)
