import streamlit as st
import requests
import json
import uuid

# --- Configuration ---
API_BASE_URL = "http://localhost:9000"
CHAT_ENDPOINT = f"{API_BASE_URL}/chat/stream"
CLEAR_ENDPOINT = f"{API_BASE_URL}/session/clear"

# --- Page Setup ---
st.set_page_config(
    page_title="সেবা বন্ধু - সরকারী তথ্য সহকারী",
    page_icon="🇧🇩",
    layout="wide"
)

# --- Custom CSS for Government Theme ---
st.markdown("""
<style>
    .stApp {
        background-color: #f5f7f6;
    }
    .stExpander {
        border: 1px solid #d1d5db;
        border-radius: 8px;
        background-color: #ffffff;
        margin-bottom: 10px;
    }
    .stChatInput {
        border-color: #006a4e !important;
    }
    h1 {
        color: #006a4e;
    }
</style>
""", unsafe_allow_html=True)

# --- Initialize Session State ---
if "user_id" not in st.session_state:
    # Generate a random User ID for this browser session
    st.session_state.user_id = str(uuid.uuid4())[:8]

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "আসসালামু আলাইকুম! আমি **সেবা বন্ধু**।\n\nবাংলাদেশ সরকারের বিভিন্ন সেবা, ফি, এবং নিয়মাবলি সম্পর্কে সঠিক তথ্য দিয়ে আপনাকে সাহায্য করাই আমার কাজ।\n\nআজ আপনাকে কীভাবে সাহায্য করতে পারি?"
        }
    ]

# --- Helper Functions ---

def clear_session():
    """Calls API to clear memory and resets UI state."""
    try:
        requests.post(CLEAR_ENDPOINT, json={"user_id": st.session_state.user_id})
    except Exception:
        pass # Ignore connection errors on clear
    
    st.session_state.messages = [] # Clear UI
    st.rerun()

# --- UI Rendering ---

# 1. Sidebar (Debug & Controls)
with st.sidebar:
    st.title("কন্ট্রোল প্যানেল")
    
    st.markdown(f"**User ID:** `{st.session_state.user_id}`")
    
    st.markdown("---")
    st.subheader("🔧 Debugging")
    debug_key = st.text_input("Admin Debug Secret", type="password", help="Enter the secret key to see CoT and Tool usage.")
    
    st.markdown("---")
    if st.button("নতুন করে শুরু করুন (Clear)", type="primary"):
        clear_session()

# 2. Main Header
st.title("🇧🇩 সেবা বন্ধু (Seba Bondhu)")
st.caption("আপনার ব্যক্তিগত সরকারী সেবা সহকারী | Powered by Graphiti & GovOps AI")

# 3. Chat History Render Loop
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        # A. User Message
        if msg["role"] == "user":
            st.markdown(msg["content"])
        
        # B. Assistant Message (The 3-Box Layout)
        elif msg["role"] == "assistant":
            
            # Box 1: Chain of Thought (Hidden unless debug key was used)
            if msg.get("cot_content"):
                with st.expander("🧠 চিন্তাভাবনা (Reasoning)", expanded=False):
                    st.markdown(msg["cot_content"])
            
            # Box 2: Tool Activity (Hidden unless debug key was used)
            if msg.get("tool_content"):
                with st.expander("🛠️ প্রযুক্তিগত কার্যক্রম (Tool Logs)", expanded=False):
                    st.markdown(msg["tool_content"])
            
            # Box 3: The Actual Answer
            st.markdown(msg["content"])

# 4. Input & Streaming Logic
if prompt := st.chat_input("আপনার প্রশ্ন লিখুন (যেমন: পাসপোর্ট ফি কত?)..."):
    
    # 4a. Add User Message to State
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 4b. Stream Assistant Response
    with st.chat_message("assistant"):
        
        # Prepare Containers
        # Only show expanders if we provided a debug key (heuristic)
        if debug_key:
            cot_expander = st.expander("🧠 চিন্তাভাবনা (Reasoning)", expanded=True)
            cot_placeholder = cot_expander.empty()
            
            tool_expander = st.expander("🛠️ প্রযুক্তিগত কার্যক্রম (Tool Logs)", expanded=True)
            tool_placeholder = tool_expander.empty()
        else:
            cot_placeholder = None
            tool_placeholder = None

        answer_placeholder = st.empty()

        # Buffers
        full_cot = ""
        full_tool_log = ""
        full_response = ""

        # Prepare Request
        payload = {
            "user_id": st.session_state.user_id,
            "query": prompt
        }
        headers = {"X-Debug-Key": debug_key} if debug_key else {}

        try:
            with requests.post(CHAT_ENDPOINT, json=payload, headers=headers, stream=True) as r:
                r.raise_for_status()
                
                for line in r.iter_lines():
                    if line:
                        try:
                            event = json.loads(line.decode('utf-8'))
                            evt_type = event.get("type")

                            # --- EVENT: Chain of Thought ---
                            if evt_type == "debug_log" and "Thinking" in event.get("title", ""):
                                if cot_placeholder:
                                    chunk = event.get("data", "")
                                    full_cot += f"{chunk}\n\n"
                                    cot_placeholder.markdown(full_cot)

                            # --- EVENT: Tool Calls & Results ---
                            elif evt_type == "debug_log" and ("Tool" in event.get("title", "") or "Result" in event.get("title", "")):
                                if tool_placeholder:
                                    title = event.get("title", "Log")
                                    data = event.get("data", "")
                                    
                                    # Create a nice log entry
                                    if "Result" in title:
                                        # Truncate massive JSON for display
                                        display_data = str(data)[:500] + ("..." if len(str(data)) > 500 else "")
                                        full_tool_log += f"**✅ {title}:**\n```\n{display_data}\n```\n"
                                    else:
                                        full_tool_log += f"**⚡ {title}:** `{data}`\n\n"
                                        
                                    tool_placeholder.markdown(full_tool_log)

                            # --- EVENT: Final Answer Text ---
                            elif evt_type == "answer_chunk":
                                full_response += event.get("content", "")
                                answer_placeholder.markdown(full_response + "▌")
                            
                            # --- EVENT: Error ---
                            elif evt_type == "error":
                                full_response += f"\n\n🚨 *System Error: {event.get('content')}*"
                                answer_placeholder.markdown(full_response)

                        except json.JSONDecodeError:
                            pass

            # Final render to remove cursor
            answer_placeholder.markdown(full_response)

            # 4c. Save to History
            # We save the debug logs too, so if the user scrolls up, they persist
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "cot_content": full_cot if full_cot else None,
                "tool_content": full_tool_log if full_tool_log else None
            })

        except Exception as e:
            st.error(f"সার্ভারের সাথে সংযোগ স্থাপন করা যাচ্ছে না। (Error: {e})")