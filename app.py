import streamlit as st
import requests
import json
import uuid
import os

# --- Configuration ---
API_BASE_URL = os.getenv("GOVOPS_API_URL", "http://localhost:9000")
CHAT_ENDPOINT = f"{API_BASE_URL}/chat/stream"
CLEAR_ENDPOINT = f"{API_BASE_URL}/session/clear"
HEALTH_ENDPOINT = f"{API_BASE_URL}/health"

# Debug secret — matches ADMIN_DEBUG_SECRET in .env
DEBUG_SECRET = os.getenv("GOVOPS_DEBUG_SECRET", "SuperDebugCoTCB")
REQUEST_TIMEOUT = int(os.getenv("GOVOPS_REQUEST_TIMEOUT", "300"))

# Search backend endpoints
WIKI_SEARCH_URL = os.getenv("WIKIPEDIA_DATA__QUERY_END_POINT", "http://172.22.11.241:9220/search")
JIGGASHA_SEARCH_URL = os.getenv("JIGGASHA_DATA__QUERY_END_POINT", "http://172.22.11.241:9210/search")

# --- Page Setup ---
st.set_page_config(
    page_title="Government Services Chatbot",
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
    .stMarkdown pre {
        white-space: pre-wrap;
        font-size: 0.85rem;
    }
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 500;
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

if "server_online" not in st.session_state:
    st.session_state.server_online = True


def clear_session():
    """Calls API to clear memory and resets UI state."""
    try:
        requests.post(CLEAR_ENDPOINT, json={"user_id": st.session_state.user_id}, timeout=10)
    except Exception:
        pass
    st.session_state.messages = []
    st.rerun()


def check_health() -> bool:
    """Check if the API server is running."""
    try:
        resp = requests.get(HEALTH_ENDPOINT, timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


# --- Check server health ---
st.session_state.server_online = check_health()

# --- Sidebar ---
with st.sidebar:
    st.title("কন্ট্রোল প্যানেল")
    st.markdown(f"**User ID:** `{st.session_state.user_id}`")

    # Server status indicator
    status = "🟢 Online" if st.session_state.server_online else "🔴 Offline"
    st.markdown(f"**Server:** {status}")

    st.markdown("---")
    if st.button("নতুন করে শুরু করুন (Clear)", type="primary"):
        clear_session()

# --- Main Header ---
st.title("Government Services Chatbot")
st.caption("আপনার ব্যক্তিগত সরকারী সেবা সহকারী | Powered by GovOps AI")

if not st.session_state.server_online:
    st.warning("⚠️ Government Services API server is not responding. Please check if the govtchat.service is running.")
    st.stop()

# --- Tabs ---
tab_chat, tab_wiki, tab_jiggasha = st.tabs([
    "💬 Chat",
    "📚 Wiki Search Explorer",
    "🏛️ Jiggasha Explorer",
])

# ====================================================================
# Tab 1: Chat
# ====================================================================
with tab_chat:
    # Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(msg["content"])
            elif msg["role"] == "assistant":
                if msg.get("cot_content"):
                    with st.expander("Reasoning", expanded=False):
                        st.markdown(msg["cot_content"])
                if msg.get("tool_content"):
                    with st.expander("Tool Logs", expanded=False):
                        st.markdown(msg["tool_content"])
                st.markdown(msg["content"])

    if prompt := st.chat_input("আপনার প্রশ্ন লিখুন (যেমন: পাসপোর্ট ফি কত?)..."):

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            cot_expander = st.expander("Reasoning", expanded=False)
            cot_placeholder = cot_expander.empty()
            tool_expander = st.expander("Tool Logs", expanded=False)
            tool_placeholder = tool_expander.empty()

            answer_placeholder = st.empty()
            full_cot = ""
            full_tool_log = ""
            full_response = ""

            payload = {"user_id": st.session_state.user_id, "query": prompt}
            headers = {"X-Debug-Key": DEBUG_SECRET}

            try:
                with requests.post(CHAT_ENDPOINT, json=payload, stream=True, timeout=REQUEST_TIMEOUT, headers=headers) as r:
                    r.raise_for_status()

                    for line in r.iter_lines():
                        if line:
                            try:
                                event = json.loads(line.decode('utf-8'))
                                evt_type = event.get("type")

                                if evt_type == "reasoning_chunk":
                                    if cot_placeholder:
                                        chunk = event.get("content", "")
                                        full_cot += chunk
                                        cot_placeholder.markdown(full_cot + "▌")

                                elif evt_type == "tool_call":
                                    if tool_placeholder:
                                        summaries = event.get("tool_call_summaries", [])
                                        if summaries:
                                            for s in summaries:
                                                args = s.get("arguments", {})
                                                args_str = json.dumps(args, ensure_ascii=False, indent=2) if args else "(no arguments)"
                                                full_tool_log += f"**🔧 Tool Call (turn {event.get('turn', '?')}):** `{s['name']}`\n"
                                                full_tool_log += f"```\n{args_str}\n```\n\n"
                                        else:
                                            tc = event.get("tool_calls", [])
                                            names = ", ".join(f"`{t['function']['name']}`" for t in tc)
                                            full_tool_log += f"**🔧 Tool Call (turn {event.get('turn', '?')}):** {names}\n"
                                        tool_placeholder.markdown(full_tool_log)

                                elif evt_type == "tool_result":
                                    if tool_placeholder:
                                        name = event.get("name", "")
                                        sources = event.get("sources", [])
                                        preview = event.get("preview", "")
                                        status = event.get("status", "ok")
                                        icon = "✅" if status == "ok" else "❌"
                                        full_tool_log += f"**{icon} Tool Result (`{name}`):**\n"
                                        if sources:
                                            full_tool_log += "Sources:\n"
                                            for s in sources:
                                                full_tool_log += f"- {s}\n"
                                        if preview:
                                            full_tool_log += f"Preview:\n```\n{preview}\n```\n\n"
                                        tool_placeholder.markdown(full_tool_log)

                                elif evt_type in ("turn_start", "turn_end", "usage"):
                                    pass

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
                st.session_state.messages.append(msg_entry)

            except requests.exceptions.Timeout:
                st.error("⏱️ সার্ভার প্রতিক্রিয়া দিতে সময় নিয়েছে। আবার চেষ্টা করুন।")
            except requests.exceptions.ConnectionError:
                st.error("❌ সার্ভারের সাথে সংযোগ স্থাপন করা যাচ্ছে না।")
            except Exception as e:
                st.error(f"সার্ভারের সাথে সংযোগ স্থাপন করা যাচ্ছে না। (Error: {e})")

# ====================================================================
# Tab 2: Wiki Search Explorer
# ====================================================================
with tab_wiki:
    st.subheader("📚 Wikipedia Search Explorer")
    st.caption(f"Endpoint: `{WIKI_SEARCH_URL}`")

    default_query = '{"formal_query": "বর্তমানে বাংলাদেশ সরকারে রাষ্ট্রপতি হিসেবে যিনি দায়িত্ব পালন করছেন তার নাম কি ?", "keyword_string": "বর্তমান রাষ্ট্রপতি বাংলাদেশ সরকার দায়িত্ব"}'

    json_input = st.text_area(
        "Paste request JSON (formal_query + keyword_string):",
        value=default_query,
        height=120,
        key="wiki_input",
    )

    col_btn1, col_btn2 = st.columns([1, 5])
    with col_btn1:
        search_clicked = st.button("🔍 Search", type="primary", use_container_width=True, key="wiki_search_btn")

    if search_clicked:
        search_url = WIKI_SEARCH_URL
        st.info(f"POST → `{search_url}`")

        try:
            payload = json.loads(json_input)
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")
            st.stop()

        try:
            resp = requests.post(search_url, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            st.markdown("---")
            st.markdown("## Response")

            # --- Request section ---
            st.markdown("### Request")
            st.json(payload, expanded=True)

            # --- Status ---
            st.success(f"Status: {resp.status_code} · Duration: {resp.elapsed.total_seconds()*1000:.0f}ms")

            # --- Combined context ---
            combined = data.get("combined_context", "")
            if combined:
                st.markdown("### Combined Context")
                st.markdown(combined)

            # --- Results ---
            results = data.get("results", [])
            if results:
                st.markdown(f"### Results ({len(results)} matches)")
                for i, r in enumerate(results, 1):
                    title = r.get("title", "Untitled")
                    url = r.get("url", "")
                    context = r.get("context", "")
                    published = r.get("published_at", "")

                    st.markdown(f"---")
                    st.markdown(f"**{i}. {title}**")
                    if url:
                        st.markdown(f"🔗 [{url}]({url})")
                    if published:
                        st.caption(f"Updated: {published}")
                    if context:
                        st.markdown(f"**Context:**\n{context}")

            # --- Raw JSON ---
            st.markdown("---")
            st.markdown("### Raw JSON Response")
            raw_str = json.dumps(data, ensure_ascii=False, indent=2)
            st.code(raw_str, language="json")

        except requests.exceptions.Timeout:
            st.error("⏱️ Request timed out.")
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to search service.")
        except requests.exceptions.HTTPError as e:
            st.error(f"❌ HTTP Error: {e}")
        except Exception as e:
            st.error(f"Error: {e}")

# ====================================================================
# Tab 3: Jiggasha Explorer
# ====================================================================
with tab_jiggasha:
    st.subheader("🏛️ Jiggasha Knowledge Search Explorer")
    st.caption(f"Endpoint: `{JIGGASHA_SEARCH_URL}`")

    default_query = '{"formal_query": "জাতীয় পরিচয়পত্র (NID) কার্ড করতে প্রক্রিয়া এবং আবশ্যক কাগজপত্র কি কি ?", "keyword_string": "NID জাতীয় পরিচয়পত্র নিবন্ধন প্রক্রিয়া কাগজপত্র", "top_k": 10}'

    json_input = st.text_area(
        "Paste request JSON (formal_query + keyword_string, optional top_k):",
        value=default_query,
        height=120,
        key="jiggasha_input",
    )

    col_btn1, col_btn2 = st.columns([1, 5])
    with col_btn1:
        search_clicked = st.button("🔍 Search", type="primary", use_container_width=True, key="search_btn")

    if search_clicked:
        search_url = JIGGASHA_SEARCH_URL
        st.info(f"POST → `{search_url}`")

        try:
            payload = json.loads(json_input)
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")
            st.stop()

        try:
            resp = requests.post(search_url, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            st.markdown("---")
            st.markdown("## Response")

            # --- Request section ---
            st.markdown("### Request")
            st.json(payload, expanded=True)

            # --- Status ---
            st.success(f"Status: {resp.status_code} · Duration: {resp.elapsed.total_seconds()*1000:.0f}ms")

            # --- Metadata fields ---
            formal_q = data.get("formal_query", "")
            keyword = data.get("keyword_string", "")
            expanded_kw = data.get("expanded_keyword_string", "")
            latency = data.get("latency_ms", None)

            meta_cols = st.columns(3)
            with meta_cols[0]:
                st.metric("Formal Query", formal_q or "—")
            with meta_cols[1]:
                st.metric("Keywords", keyword or "—")
            with meta_cols[2]:
                st.metric("Latency", f"{latency:.0f}ms" if latency else "—")
            if expanded_kw:
                st.caption(f"Expanded keywords: {expanded_kw}")

            # --- Combined context ---
            combined = data.get("combined_context", "")
            if combined:
                st.markdown("### Combined Context")
                st.markdown(combined)

            # --- Results ---
            results = data.get("results", [])
            if results:
                st.markdown(f"### Results ({len(results)} matches)")
                for i, r in enumerate(results, 1):
                    node = r.get("node", "")
                    text = r.get("text", "")
                    score = r.get("score", 0)
                    reason = r.get("reason", "")
                    category = r.get("Category", "")

                    st.markdown(f"---")
                    col_head, col_score = st.columns([5, 1])
                    with col_head:
                        st.markdown(f"**{i}. {node}**")
                    with col_score:
                        st.metric("Score", f"{score:.4f}")

                    if category:
                        st.caption(f"Category: {category}")
                    if text:
                        st.markdown(text)
                    if reason:
                        st.caption(f"Reason: {reason}")

            # --- Raw JSON ---
            st.markdown("---")
            st.markdown("### Raw JSON Response")
            raw_str = json.dumps(data, ensure_ascii=False, indent=2)
            st.code(raw_str, language="json")

        except requests.exceptions.Timeout:
            st.error("⏱️ Request timed out.")
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to search service.")
        except requests.exceptions.HTTPError as e:
            st.error(f"❌ HTTP Error: {e}")
        except Exception as e:
            st.error(f"Error: {e}")
