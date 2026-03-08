# GovOps Graphiti API (v2) Technical Manual

## 1. API Architecture Overview
The API acts as a bridge between the user and a **Reasoning LLM** integrated with a **Knowledge Graph**. 
- **Standard Mode:** Provides a clean, filtered response for citizens.
- **Debug Mode:** Provides a transparent "Glass Box" view of the AI's logic, including internal thoughts (Chain-of-Thought) and raw data retrieved from the database.

### Endpoint Details
*   **URL:** `http://<host>:9000/chat/stream`
*   **Method:** `POST`
*   **Protocol:** NDJSON (Newline Delimited JSON)

---

## 2. Case 1: Standard User Mode (No Debug Key)
**Scenario:** A citizen asking a general question.

### Request
```bash
curl -X POST http://localhost:9000/chat/stream \
-H "Content-Type: application/json" \
-d '{"user_id": "citizen_01", "query": "পাসপোর্ট করতে কত টাকা লাগে?"}'
```

### API Behavior
1.  **Suppression:** The API identifies text wrapped in `<CoT>...</CoT>` tags and **removes it** from the stream.
2.  **Tool Silence:** Tool execution happens in the background; the user sees nothing until the final answer starts.
3.  **Output:** Only `answer_chunk` and `error` types are sent.

### Example Stream Output
```json
{"type": "answer_chunk", "content": "বাংলাদেশে"}
{"type": "answer_chunk", "content": " ই-পাসপোর্টের"}
{"type": "answer_chunk", "content": " সাধারণ ফি ৫,৭৫০ টাকা।"}
```

### UI Implementation
*   **Display:** Directly append `content` to the chat bubble.
*   **User Experience:** The user sees a typing effect for the final answer.

---

## 3. Case 2: Debug/Admin Mode (With Debug Key)
**Scenario:** A developer or admin investigating how the AI reached an answer.

### Request
```bash
curl -X POST http://localhost:9000/chat/stream \
-H "Content-Type: application/json" \
-H "X-Debug-Key: ADMIN_SECRET_123" \
-d '{"user_id": "admin_01", "query": "পাসপোর্ট করতে কত টাকা লাগে?"}'
```

### API Behavior
1.  **Thinking Capture:** Text inside `<CoT>` is extracted and sent as a `debug_log` with the title `🧠 Thinking`.
2.  **Tool Transparency:** When the agent calls `graph_search`, the raw Markdown results from the Graph database are sent as a `debug_log` with the title `🛠️ Tool Call`.
3.  **Output:** Includes `debug_log` AND `answer_chunk`.

### Example Stream Output
*(Formatted for readability, normally received line-by-line)*

**Step 1: Reasoning**
```json
{"type": "debug_log", "title": "🧠 Thinking", "data": "User intent: Passport fees. Need to verify latest rates using graph_search(query='e-passport fees')."}
```

**Step 2: Database Results**
```json
{"type": "debug_log", "title": "🛠️ Tool Call", "data": "## Nodes\n**E-Passport**: 48 Pages, 5 Years: 4025 BDT\n**E-Passport**: 48 Pages, 10 Years: 5750 BDT\n"}
```

**Step 3: Final Synthesis**
```json
{"type": "answer_chunk", "content": "বাংলাদেশে ই-পাসপোর্টের জন্য ফি আপনার পাহারার মেয়াদের ওপর নির্ভর করে। "}
{"type": "answer_chunk", "content": "১০ বছর মেয়াদী ৪৮ পাতার পাসপোর্টের ফি ৫,৭৫০ টাকা।"}
```

---

## 4. JSON Schema Specification

### Event: `answer_chunk`
| Field | Type | Description |
| :--- | :--- | :--- |
| `type` | `string` | Always `"answer_chunk"` |
| `content` | `string` | A snippet of the final Bengali response. |

### Event: `debug_log`
| Field | Type | Description |
| :--- | :--- | :--- |
| `type` | `string` | Always `"debug_log"` |
| `title` | `string` | `"🧠 Thinking"` (Logic) or `"🛠️ Tool Call"` (Data Retrieval). |
| `data` | `string` | The actual log content (Plain text or Markdown). |

### Event: `error`
| Field | Type | Description |
| :--- | :--- | :--- |
| `type` | `string` | Always `"error"` |
| `content` | `string` | Human-readable error message. |

---

## 5. UI Implementation Guide for "Independent UIs"

To build a professional dashboard or chat interface using this API, follow these UI component guidelines:

### Component A: The Reasoning Box
*   **Trigger:** When a `debug_log` with title `"🧠 Thinking"` is received.
*   **Visual:** A collapsible "Accordion" or a "Pulse" icon labeled "Thinking...".
*   **Behavior:** Replace or append the `data` inside this box so the developer can see the AI's step-by-step logic.

### Component B: The Logs Terminal
*   **Trigger:** When a `debug_log` with title `"🛠️ Tool Call"` is received.
*   **Visual:** A dark-themed code block or "Terminal" style window.
*   **Behavior:** Render the Markdown `data`. This shows exactly what facts were pulled from the Government Knowledge Graph.

### Component C: The Citizen Bubble
*   **Trigger:** When `answer_chunk` is received.
*   **Visual:** Standard chat bubble.
*   **Behavior:** Append text in real-time. This is the only component shown to regular users.

---

## 6. Summary Table

| Feature | Standard Mode (`X-Debug-Key` missing) | Debug Mode (`X-Debug-Key` valid) |
| :--- | :--- | :--- |
| **Visibility** | Public Citizens | Developers / Admins |
| **Internal Thoughts** | **Hidden** (Filtered out) | **Visible** (as `debug_log`) |
| **Graph Search Logs** | **Hidden** | **Visible** (as `debug_log`) |
| **Final Answer** | Visible | Visible |
| **Latency** | Feels faster (less data) | Higher data overhead |

**Note:** If an invalid `X-Debug-Key` is provided, the API defaults to **Standard User Mode** without throwing an error, but it will not provide logs.