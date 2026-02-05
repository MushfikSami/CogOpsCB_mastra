import asyncio
import os
import json
import logging
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict,Optional
from fastapi import Header
from dotenv import load_dotenv
from cogops.agents.graphiti_agent import GraphitiAgent

load_dotenv()

# --- Configuration ---
API_PORT = 9000
API_HOST = "0.0.0.0"
AGENT_CONFIG_PATH = "configs/v2.yaml"

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("api_v2")

# --- API Application ---
app = FastAPI(
    title="GovOps Graphiti API (v2)",
    description="Government Services AI Agent backed by Neo4j Knowledge Graph.",
    version="2.0.0"
)

# --- CORS (Allow all for development) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Session Store ---
# In-memory storage: { user_id: GraphitiAgent_Instance }
# For production scaling, you would use Redis/Database persistence.
active_sessions: Dict[str, GraphitiAgent] = {}
session_lock = asyncio.Lock()

# --- Request Models ---
class ChatRequest(BaseModel):
    user_id: str
    query: str

class SessionRequest(BaseModel):
    user_id: str

# --- Helper Functions ---

async def get_agent_session(user_id: str) -> GraphitiAgent:
    """
    Thread-safe retrieval or creation of an agent session.
    """
    async with session_lock:
        if user_id not in active_sessions:
            logger.info(f"🆕 Creating new session for User: {user_id}")
            try:
                # Initialize a fresh agent for this user
                agent = GraphitiAgent(config_path=AGENT_CONFIG_PATH)
                active_sessions[user_id] = agent
            except Exception as e:
                logger.error(f"Failed to create agent session: {e}")
                raise HTTPException(status_code=500, detail="Agent initialization failed.")
        else:
            logger.info(f"🔄 Resuming session for User: {user_id}")
            
        return active_sessions[user_id]

# --- Endpoints ---

@app.get("/health", tags=["System"])
async def health_check():
    """System status and active session count."""
    return {
        "status": "online", 
        "service": "GovOps Graphiti v2",
        "active_sessions": len(active_sessions)
    }

@app.post("/chat/stream", tags=["Chat"])
async def stream_chat(request: ChatRequest,x_debug_key: Optional[str] = Header(None, alias="X-Debug-Key") ):
    """
    Main Chat Endpoint.
    Streams the agent's response (and tool logs) as Newline Delimited JSON (NDJSON).
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    agent = await get_agent_session(request.user_id)
    server_debug_secret = os.getenv("ADMIN_DEBUG_SECRET")
    debug_mode = (server_debug_secret is not None) and (x_debug_key == server_debug_secret)

    async def event_generator():
        try:
            # Process the query through the agent pipeline
            async for event in agent.process_query(request.query,debug_mode=debug_mode):
                # Serialize event to JSON string
                json_str = json.dumps(event, ensure_ascii=True) 
                yield f"{json_str}\n"
                # Brief sleep to allow asyncio loop to breathe under load
                await asyncio.sleep(0.001)
                
        except Exception as e:
            logger.error(f"Stream error for user {request.user_id}: {e}")
            err_msg = json.dumps({
                "type": "error", 
                "content": "Server-side streaming error."
            })
            yield f"{err_msg}\n"

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")

@app.post("/session/clear", tags=["Session"])
async def clear_session(request: SessionRequest):
    """
    Wipes the conversation history for a specific user.
    """
    async with session_lock:
        if request.user_id in active_sessions:
            # Call agent's internal clear method if exists
            active_sessions[request.user_id].clear_session()
            # Remove from memory
            del active_sessions[request.user_id]
            logger.info(f"🧹 Cleared session for User: {request.user_id}")
            return {"status": "success", "message": "Session cleared."}
        else:
            return {"status": "ignored", "message": "No active session found."}

if __name__ == "__main__":
    print(f"🚀 Starting GovOps API v2 on http://{API_HOST}:{API_PORT}")
    uvicorn.run(app, host=API_HOST, port=API_PORT)