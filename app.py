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
    page_title="RAMDOM UI - Government Services Chatbot",
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
    st.session_state.user_id = str(uuid.uuid4())[:8]

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "আসসালামু আলাইকুম! বাংলাদেশ সরকারের বিভিন্ন সেবা, ফি, এবং নিয়মাবলি সম্পর্কে সঠিক তথ্য দিয়ে আপনাকে সাহায্য করাই আমার কাজ।\n\nআজ আপনাকে কীভাবে সাহায্য করতে পারি?"
        }
    ]

# --- Helper Functions ---

def clear_session():
    """Calls API to clear memory and resets UI state."""
    try:
        requests.post(CLEAR_ENDPOINT, json={"user_id": st.session_state.user_id})
    except Exception:
        pass
    st.session_state.messages = []
    st.rerun()

# --- UI Rendering ---

# 1. Sidebar (Debug & Controls)
with st.sidebar:
    st.title("কন্ট্রোল প্যানেল")
    st.markdown(f"**User ID:** `{st.session_state.user_id}`")
    st.markdown("---")
    st.subheader("Debugging")
    debug_key = st.text_input("Admin Debug Secret", type="password", help="Enter the secret key to see Reasoning and Tool usage.")
    st.markdown("---")
    if st.button("নতুন করে শুরু করুন (Clear)", type="primary"):
        clear_session()

# 2. Main Header
st.title("DUMMY UI - Government Services Chatbot")
st.caption("আপনার ব্যক্তিগত সরকারী সেবা সহকারী | Powered by Graphiti & GovOps AI")

# 3. Chat History Render Loop
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        elif msg["role"] == "assistant":
            if msg.get("clarification_question"):
                with st.container():
                    st.warning(msg["clarification_question"])
                    if msg.get("clarification_options"):
                        for opt in msg["clarification_options"]:
                            if st.button(f"→ {opt}", key=f"clarify_{msg.get('turn_id', '')}_{opt}"):
                                # This button would normally send the user's reply via the API
                                st.session_state.messages.append({"role": "user", "content": opt})
                                st.rerun()
            if msg.get("cot_content"):
                with st.expander("Reasoning", expanded=False):
                    st.markdown(msg["cot_content"])
            if msg.get("tool_content"):
                with st.expander("Tool Logs", expanded=False):
                    st.markdown(msg["tool_content"])
            st.markdown(msg["content"])

# 4. Input & Streaming Logic
if prompt := st.chat_input("আপনার প্রশ্ন লিখুন (যেমন: পাসপোর্ট ফি কত?)..."):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if debug_key:
            cot_expander = st.expander("Reasoning", expanded=True)
            cot_placeholder = cot_expander.empty()
            tool_expander = st.expander("Tool Logs", expanded=True)
            tool_placeholder = tool_expander.empty()
        else:
            cot_placeholder = None
            tool_placeholder = None

        answer_placeholder = st.empty()
        full_cot = ""
        full_tool_log = ""
        full_response = ""
        clarification_data = None

        payload = {"user_id": st.session_state.user_id, "query": prompt}
        headers = {"X-Debug-Key": debug_key} if debug_key else {}

        try:
            with requests.post(CHAT_ENDPOINT, json=payload, headers=headers, stream=True) as r:
                r.raise_for_status()

                for line in r.iter_lines():
                    if line:
                        try:
                            event = json.loads(line.decode('utf-8'))
                            evt_type = event.get("type")

                            # --- New event types ---
                            if evt_type == "reasoning_chunk":
                                if cot_placeholder:
                                    chunk = event.get("data", "")
                                    full_cot += chunk
                                    cot_placeholder.markdown(full_cot + "▌")

                            elif evt_type == "tool_call":
                                if tool_placeholder:
                                    tc = event.get("tool_calls", [])
                                    names = ", ".join(f"`{t['function']['name']}`" for t in tc)
                                    full_tool_log += f"**🔧 Tool Call (turn {event.get('turn', '?')}):** {names}\n"
                                    tool_placeholder.markdown(full_tool_log)

                            elif evt_type == "tool_result":
                                if tool_placeholder:
                                    content = event.get("content", "")
                                    full_tool_log += f"**✅ Tool Result:**\n```\n{str(content)[:400]}{'...' if len(str(content)) > 400 else ''}\n```\n\n"
                                    tool_placeholder.markdown(full_tool_log)

                            elif evt_type == "clarification_needed":
                                clarification_data = {
                                    "question": event.get("question", ""),
                                    "options": event.get("options", []),
                                    "reason": event.get("reason", ""),
                                    "turn_id": event.get("turn_id", ""),
                                }
                                st.warning(clarification_data["question"])
                                if clarification_data["options"]:
                                    for opt in clarification_data["options"]:
                                        if st.button(f"→ {opt}", key=f"clarify_{clarification_data['turn_id']}_{opt}"):
                                            st.session_state.messages.append({
                                                "role": "user",
                                                "content": opt,
                                            })
                                            st.rerun()
                                break  # Stream ends on clarification

                            # --- Ignored debug-only structural events (silently skipped) ---
                            elif evt_type in ("turn_start", "turn_end", "usage"):
                                pass

                            # --- Existing event types ---
                            elif evt_type == "answer_chunk":
                                full_response += event.get("content", "")
                                answer_placeholder.markdown(full_response + "▌")

                            elif evt_type == "error":
                                full_response += f"\n\nSystem Error: {event.get('content')}"
                                answer_placeholder.markdown(full_response)

                        except json.JSONDecodeError:
                            pass

            answer_placeholder.markdown(full_response)

            msg_entry = {
                "role": "assistant",
                "content": full_response,
                "cot_content": full_cot if full_cot else None,
                "tool_content": full_tool_log if full_tool_log else None,
            }
            if clarification_data:
                msg_entry["clarification_question"] = clarification_data["question"]
                msg_entry["clarification_options"] = clarification_data["options"]
                msg_entry["turn_id"] = clarification_data.get("turn_id", "")
            st.session_state.messages.append(msg_entry)

        except Exception as e:
            st.error(f"সার্ভারের সাথে সংযোগ স্থাপন করা যাচ্ছে না। (Error: {e})")
