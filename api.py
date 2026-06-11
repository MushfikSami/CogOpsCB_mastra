"""
api.py — FastAPI server for GovOps Chatbot.

Endpoints:
  POST /chat/stream  — Main streaming chat (NDJSON)
  GET  /health       — Service health (LLM, Redis)
  POST /session/clear — Clear conversation for a user
  GET  /query-log    — Last N query entries
"""

import asyncio
import json
import logging
import os
import uuid
from typing import Dict, Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from cogops.agents.orchestrator import Orchestrator
from cogops.events.channels import filter_for_debug, filter_for_user
from cogops.session.logger import QuerySessionLogger

load_dotenv()

# --- Config ---
API_PORT = int(os.getenv("API_PORT", "9000"))
API_HOST = "0.0.0.0"
AGENT_CONFIG_PATH = "configs/config.yml"

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

# --- Session state ---
active_sessions: Dict[str, Orchestrator] = {}
session_lock = asyncio.Lock()

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


# --- Helpers ---
async def get_agent_session(user_id: str) -> Orchestrator:
    """Thread-safe retrieval or creation of an agent session."""
    async with session_lock:
        if user_id not in active_sessions:
            try:
                agent = Orchestrator(config_path=AGENT_CONFIG_PATH)
                active_sessions[user_id] = agent
                logger.info("New session created for %s", user_id)
            except Exception as e:
                logger.error("Agent init failed for %s: %s", user_id, e)
                raise HTTPException(status_code=500, detail="Agent initialization failed.")
        return active_sessions[user_id]


# --- Endpoints ---
@app.get("/ui", response_class=HTMLResponse)
async def ui_root():
    """Single-page chat UI."""
    return HTMLResponse(content=_UI_INDEX_HTML)


@app.get("/health")
async def health_check():
    """Service health: LLM, Redis."""
    status: Dict[str, str] = {"status": "online"}

    # LLM
    try:
        agent = await get_agent_session("health_probe")
        llm_status = await agent.llm_service.health_check()
        status["llm"] = llm_status
        del active_sessions["health_probe"]
    except Exception:
        status["llm"] = "error"

    # Redis
    try:
        agent = await get_agent_session("health_redis")
        redis_status = "ok" if agent.redis_store.available else "unavailable"
        if agent.redis_store.available:
            agent.redis_store._client.ping()
        status["redis"] = redis_status
        del active_sessions["health_redis"]
    except Exception:
        status["redis"] = "error"

    status["active_sessions"] = str(len(active_sessions))
    return status


@app.post("/chat/stream")
async def stream_chat(
    request: ChatRequest,
    x_debug_key: Optional[str] = Header(None, alias="X-Debug-Key"),
):
    """Main chat endpoint. Streams NDJSON events.

    Without a valid `X-Debug-Key` header, the stream is filtered to
    user-visible events only. With the correct key, all debug events pass through.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    agent = await get_agent_session(request.user_id)

    _session_logger.append_query(request.query)
    server_secret = os.getenv("ADMIN_DEBUG_SECRET")
    debug_mode = False
    if x_debug_key:
        if server_secret == "":
            debug_mode = True
        else:
            debug_mode = x_debug_key == server_secret

    session_id = _session_logger.start_session(request.user_id, request.query)
    request_id = str(uuid.uuid4())[:8]
    logger.info("[req:%s] Chat request from %s", request_id, request.user_id)

    async def event_generator():
        try:
            async for event in agent.process_query(request.query, user_id=request.user_id):
                _session_logger.ingest_event(event)
                if debug_mode:
                    filtered = filter_for_debug([event])
                else:
                    filtered = filter_for_user([event])
                for evt in filtered:
                    yield json.dumps(evt, ensure_ascii=True) + "\n"
        except Exception as e:
            logger.error("[req:%s] Stream error: %s", request_id, e, exc_info=True)
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
    """Clear conversation for a user."""
    async with session_lock:
        if request.user_id in active_sessions:
            del active_sessions[request.user_id]
            logger.info("Session cleared for %s", request.user_id)
            return {"status": "success", "message": "Session cleared."}
        return {"status": "ignored", "message": "No active session found."}


if __name__ == "__main__":
    uvicorn.run(app, host=API_HOST, port=API_PORT)
