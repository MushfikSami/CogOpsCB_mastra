import asyncio
import json
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Dict, Any

# --- MODIFIED: Import your ChatAgent class ---
from cogops.agents.prositional_agent import ChatAgent
from fastapi.middleware.cors import CORSMiddleware

# --- Global Configuration ---
# Define the path to your config file once
AGENT_CONFIG_PATH = "configs/v1.yaml"

# --- API Setup ---
app = FastAPI(
    title="Configurable Chat Agent API",
    description="A production-grade API for the Chat Agent service with multi-user session management.",
    version="2.0.0",
    docs_url=None,    # Disable default access
    redoc_url=None,   # Disable default access
    openapi_url=None  # Disable default access
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Session Management ---
# In-memory session store. For production, consider Redis or another distributed cache.
# MODIFIED: The dictionary now stores ChatAgent instances.
chat_sessions: Dict[str, ChatAgent] = {}
sessions_lock = asyncio.Lock()


# --- Pydantic Models for Request Bodies ---
class ChatRequest(BaseModel):
    user_id: str
    query: str

class ClearSessionRequest(BaseModel):
    user_id: str


# --- Core Session Logic ---
async def get_or_create_session(user_id: str) -> ChatAgent:
    """
    Retrieves an existing chat session or creates a new one. This function is thread-safe.
    """
    async with sessions_lock:
        if user_id not in chat_sessions:
            print(f"-> Creating new chat session for user_id: {user_id}")
            # MODIFIED: Instantiate the ChatAgent with its required config path.
            chat_sessions[user_id] = ChatAgent(config_path=AGENT_CONFIG_PATH)
        else:
            print(f"-> Found existing session for user_id: {user_id}")
        return chat_sessions[user_id]


# --- API Endpoints ---
@app.get("/health", tags=["Monitoring"])
async def health_check():
    """Confirms the service is running and reports the number of active sessions."""
    return {"status": "ok", "active_sessions": len(chat_sessions)}


@app.post("/chat/clear_session", tags=["Session Management"])
async def clear_session(request: ClearSessionRequest):
    """
    Clears the conversation history for a specific user by deleting their session instance.
    """
    user_id = request.user_id
    message = ""
    
    async with sessions_lock:
        if user_id in chat_sessions:
            del chat_sessions[user_id]
            message = f"Session for user_id '{user_id}' has been cleared."
            print(f"-> Cleared session for user_id: {user_id}")
        else:
            message = f"No active session found for user_id '{user_id}'. Nothing to clear."
            print(f"-> Attempted to clear non-existent session for user_id: {user_id}")
            
    return {"status": "success", "message": message}


@app.post("/chat/stream", tags=["Chat"])
async def stream_chat(chat_request: ChatRequest):
    """
    Main chat endpoint. Manages the user's session and streams the response
    back as newline-delimited JSON (NDJSON).
    """
    session = await get_or_create_session(chat_request.user_id)

    async def response_generator():
        """
        Async generator that yields events from the ChatAgent, formatted for streaming.
        """
        try:
            # MODIFIED: Use `async for` to iterate over the async generator `process_query`.
            async for event in session.process_query(chat_request.query):
                json_event = json.dumps(event, ensure_ascii=False)
                yield f"{json_event}\n"
                # A small sleep is good practice to prevent tight-looping if the
                # downstream process is very fast, allowing other tasks to run.
                await asyncio.sleep(0.001)
        except Exception as e:
            print(f"An error occurred during generation for user {chat_request.user_id}: {e}")
            error_event = {
                "type": "error",
                "content": "An internal error occurred. Please try again later."
            }
            yield f"{json.dumps(error_event)}\n"

    return StreamingResponse(response_generator(), media_type="application/x-ndjson")


if __name__ == "__main__":
    # To run this service:
    # 1. Place this file in the same root directory as your `chat_agent.py`, `prompts`, etc.
    # 2. Run the command in your terminal:
    #    uvicorn api_service:app --host 0.0.0.0 --port 9000 --reload
    print("Starting Uvicorn server on http://0.0.0.0:9000")
    # Note: The uvicorn.run call is mainly for IDE-based execution.
    # The terminal command is the standard way to run it.
    uvicorn.run("api_service:app", host="0.0.0.0", port=9000, reload=True)