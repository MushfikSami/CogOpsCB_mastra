# Jiggasha Search Service — API Reference

Bengali government service search API. An agent calls this tool to look up information from the Bangladesh government service database (Jiggasha).

**Base URL:** `http://172.22.11.241:9210`

---

## Tool Description (for agent system prompt)

> You have access to a search tool that queries the Bangladesh government service database (Jiggasha). The database contains information about government services, procedures, fees, and requirements across various boards and departments.
>
> When you need to answer a question about Bangladesh government services, use this tool. The tool expects two inputs:
> - `formal_query`: The question you are trying to answer, written in formal Bengali (বাংলা). Write it as a proper question using formal Bengali vocabulary and tone — as you would write on an official government form. For example, "কীভাবে" instead of "কিভাবে", "পরিবর্তন" instead of "চেঞ্জ", "নিয়ম" instead of "কি নিয়ম". The question should express the exact information you are seeking.
> - `keyword_string`: A list of 3–8 key Bengali words/phrases related to the topic that should appear in the database text. Think of the most important terms someone would use to find this information on a government website.
>
> The tool returns relevant passages from the government service database with the exact text answering the query.

---

## Endpoints

### `GET /health`

Health check endpoint.

**Response:**
```json
{ "status": "ok" }
```

---

### `POST /search`

Full search pipeline. Accepts a formal Bengali query and related keywords, then returns relevant government service passages.

**Request Body (JSON):**

| Field              | Type   | Required | Constraints | Description                                                  |
|--------------------|--------|----------|-------------|--------------------------------------------------------------|
| `formal_query`     | string | yes      | 1–500 chars | Formal Bengali question representing the exact information sought. Write in proper formal Bengali (বাংলা) as it would appear on a government document. Example: "এসএসসি সার্টিফিকেটে নিজের নাম পরিবর্তন করতে ঢাকা শিক্ষা বোর্ডে কত টাকা জমা দিতে হয় ?" |
| `keyword_string`   | string | yes      | 1–500 chars | Space-separated Bengali keywords/phrases related to the query. These words should be terms that exist in the government database text as they appear together in a related document. Example: "এসএসসি সার্টিফিকেট নাম পরিবর্তন ঢাকা শিক্ষা বোর্ড ফি" |
| `top_k`            | int    | no       | 5–50        | Number of BM25 passages to retrieve before reranking (default: 20) |

**Example request:**
```bash
curl -X POST "http://172.22.11.241:9210/search" \
  -H "Content-Type: application/json" \
  -d '{
    "formal_query": "এসএসসি সার্টিফিকেটে নিজের নাম পরিবর্তন করতে ঢাকা শিক্ষা বোর্ডে কত টাকা জমা দিতে হয় ? ",
    "keyword_string": "এসএসসি সার্টিফিকেট নাম পরিবর্তন ঢাকা শিক্ষা বোর্ড ফি"
  }'
```

**Pipeline stages:**

1. **Chroma retrieval** — `keyword_string` is embedded and used to find top-5 similar keyword documents in ChromaDB.
2. **LLM Chroma filter** — LLM selects which Chroma results are relevant to the query (returns 0-based node IDs).
3. **Keyword collection** — Unique keywords are extracted from the filtered Chroma documents.
4. **Synonym expansion** — Words in `keyword_string` not found in collected keywords are replaced with synonyms from `synonym_groups.json`.
5. **BM25 retrieval** — Elasticsearch BM25 with field boosting (`keywords^5`, `text^4`, `text_keywords^3`, `topic^1`) using the expanded keyword string.
6. **Reranking** — LLM binary relevance classifier scores each passage against `formal_query`, keeps only those scoring above 0.5.
7. **Combined context cutting** — All matching results are combined, and an LLM selects the most relevant line ranges.
8. **Context trimming** — Extracted context is trimmed to 1000 words if it exceeds that limit.

---

## Response Schema

**`SearchResponse`:**

| Field                      | Type           | Description                                                |
|----------------------------|----------------|------------------------------------------------------------|
| `formal_query`             | string         | Echo of the input formal_query                             |
| `keyword_string`           | string         | Echo of the input keyword_string                           |
| `expanded_keyword_string`  | string         | The keyword_string after synonym expansion applied         |
| `chroma_keywords`          | string         | Space/comma-separated unique keywords collected from Chroma|
| `chroma_details`           | string         | Chroma top-5 results formatted as node_id/node/summary blocks (for debugging) |
| `chroma_filter_ids`        | string         | 0-based node IDs retained by LLM filter (for debugging)    |
| `results`                  | array          | Top results after reranking (see below)                    |
| `combined`                 | string         | All result texts combined with Node/Text headers           |
| `combined_sed`             | array          | Line ranges selected by context cutter (for debugging)     |
| `combined_context`         | string         | **The answer** — extracted relevant text, trimmed to 1000 words max |
| `latency_ms`               | float          | Total pipeline latency in milliseconds                     |

**Each result item (inside `results` array):**

| Field    | Type   | Description                                              |
|----------|--------|----------------------------------------------------------|
| `node`   | string | Hierarchical node path (pipe-separated), e.g. "শিক্ষা\|নথি\|নাম পরিবর্তন" |
| `text`   | string | Full passage text from the government database           |
| `score`  | float  | Relevance score from LLM reranker (0–1, higher = more relevant) |
| `reason` | string | "LLM output: 1" means relevant, "LLM output: 0" means not relevant |

---

## Example Response

```json
{
  "formal_query": "এসএসসি সার্টিফিকেটে নিজের নাম পরিবর্তন করতে ঢাকা শিক্ষা বোর্ডে কত টাকা জমা দিতে হয় ? ",
  "keyword_string": "এসএসসি সার্টিফিকেট নাম পরিবর্তন ঢাকা শিক্ষা বোর্ড ফি",
  "expanded_keyword_string": "এসএসসি সনদ নাম পরিবর্তন ঢাকা শিক্ষা বোর্ড ফি",
  "chroma_keywords": "Fee, Name, Correction, সার্টিফিকেট, সনদ, পরিবর্তন, ফি, জমা",
  "chroma_details": "node_id: 0\nnode: শিক্ষা সম্পর্কিত সেবা|...\nsummary: জেএসসি, এসএসসি বা এইচএসসি ...\n---\n...",
  "chroma_filter_ids": "0, 4",
  "results": [
    {
      "node": "শিক্ষা সম্পর্কিত সেবা|শিক্ষা ক্ষেত্রে নথি ও সনদ সম্পর্কিত সেবা|শিক্ষা বিষয়ক নথির যাবতীয় সংশোধন|ঢাকা শিক্ষাবোর্ডের যাবতীয় ফি",
      "text": "ঢাকা শিক্ষাবোর্ডের যাবতীয় ফি : ঢাকা শিক্ষা বোর্ডে নাম, বয়স বা জন্মতারিখ সংশোধন করতে নির্ধারিত ফি জমা দিতে হবে...\n- নাম সংশোধন (নিজ/পিতা/মাতা): ৫০০ টাকা\n- বয়স/জন্মতারিখ সংশোধন: ১০০০ টাকা",
      "score": 0.9995,
      "reason": "LLM output: 1"
    },
    {
      "node": "শিক্ষা সম্পর্কিত সেবা|শিক্ষা ক্ষেত্রে নথি ও সনদ সম্পর্কিত সেবা|জেএসসি/এসএসসি/ এইচএসসি পরীক্ষার আবেদনের ক্ষেত্রে নাম সংশোধন",
      "text": "জেএসসি/এসএসসি/ এইচএসসি পরীক্ষার আবেদনের ক্ষেত্রে নাম সংশোধন...",
      "score": 0.8176,
      "reason": "LLM output: 1"
    }
  ],
  "combined_context": "Node: শিক্ষা সম্পর্কিত সেবা|শিক্ষা ক্ষেত্রে নথি ও সনদ সম্পর্কিত সেবা|শিক্ষা বিষয়ক নথির যাবতীয় সংশোধন|ঢাকা শিক্ষাবোর্ডের যাবতীয় ফি\nText: ঢাকা শিক্ষাবোর্ডের যাবতীয় ফি :\nঢাকা শিক্ষা বোর্ডে নাম, বয়স বা জন্মতারিখ সংশোধন করতে নির্ধারিত ফি জমা দিতে হবে...\n\n- নাম সংশোধন (নিজ/পিতা/মাতা): ৫০০ টাকা\n- বয়স/জন্মতারিখ সংশোধন: ১০০০ টাকা\n- রেজিস্ট্রেশন পত্র সংশোধন: ২০০ টাকা\n  - (১৯৯৫ সালের আগের সার্টিফিকেট হলে: ১০০০ টাকা)\n- নকল সার্টিফিকেট উত্তোলন ফি: ৫০০ টাকা",
  "latency_ms": 2593.2
}
```

**How to use the response:** The agent should read `combined_context` for the direct answer. The `results` array provides source documents with scores — use the top result's `node` to identify which service/board the information comes from, and `text` for the full detail if needed.

---

## Error Responses

| Status | Condition                      | Response                                          |
|--------|--------------------------------|---------------------------------------------------|
| 422    | Missing or invalid body fields | `{"detail": [{"msg": "Field required", ...}]}`    |
| 500    | Pipeline error (ES/LLM/Chroma) | `{"detail": "Search failed: ..."}`                |

---

## Notes for Agent Tool Wrappers

- The `combined_context` field contains the answer text that was found during execution
- If `results` is empty,  no matching government service information was found.
- The `node` field uses `|` as a separator for hierarchical paths (category → sub-category → service → specific topic).
- All text is in Bengali (বাংলা) — this is the language of the Bangladesh government service database.
- The database covers topics: education, passports, NID, birth/death registration, trade licenses, metro rail, land records, and more.
