# Wikipedia Search API

A Bengali-language Wikipedia search service that retrieves relevant articles from a Bangladesh-focused Wikipedia test database, extracts precise context using LLM-powered document cutting, and returns structured results.

---

## Base URL

```
http://172.22.11.241:9220
```

---

## Endpoints

### POST `/search`

Retrieve relevant Wikipedia articles and condensed context for a given Bengali query.

#### Request Body

```json
{
  "formal_query": "...",
  "keyword_string": "..."
}
```

| Field            | Type   | Required | Description                                                                                                                              |
| ---------------- | ------ | -------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `formal_query`   | string | Yes      | The specific information the agent is looking for, expressed in formal Bengali. This represents the absolute question being answered.     |
| `keyword_string` | string | Yes      | A space- or pipe-separated list of keywords related to the query that should be present in the Wikipedia database text.                   |

#### Response Body

```json
{
  "results": [
    {
      "title": "...",
      "url": "...",
      "context": "...",
      "sed": [[1, 5], [8, 12]],
      "published_at": "2025-06-15"
    }
  ],
  "combined_context": "..."
}
```

| Field              | Type      | Description                                                                                                          |
| ------------------ | --------- | -------------------------------------------------------------------------------------------------------------------- |
| `results`          | array     | List of matching Wikipedia articles (one per passage the LLM found relevant content for).                             |
| `combined_context` | string    | All article contexts joined together as `"### {title}\n\n{context}"`, truncated to ~1000 words.                       |

Each item in `results` contains:

| Field            | Type              | Description                                                                                                      |
| ---------------- | ----------------- | --------------------------------------------------------------------------------------------------------------- |
| `title`          | string            | Wikipedia article title.                                                                                        |
| `url`            | string            | Wikipedia article URL.                                                                                          |
| `context`        | string            | LLM-extracted, condensed text — only the relevant lines from the article for this query.                         |
| `sed`            | array of tuples   | The line ranges returned by the LLM cutter, e.g. `[[1, 5], [8, 12]]`. `[[0, 0]]` means nothing relevant.        |
| `published_at`   | string or null    | The article's `last_updated` timestamp from Elasticsearch (may be null).                                         |

#### Health Check

```
GET /health
```

Response:

```json
{
  "status": "ok"
}
```

---

## Usage Notes for Agents

1. **`formal_query`** — Write the agent's question in formal, natural Bengali. The query should express the exact information the agent is trying to find. It will be used by the LLM to generate a final answer from the returned context.

2. **`keyword_string`** — Provide 3–8 keywords (space-separated) extracted from the `formal_query`. These drive the Elasticsearch retrieval. Pipe-separated keywords also work: `"keyword1 keyword2 | keyword3"`.

3. The API returns **only the most relevant lines** from each article (not the full text). The `combined_context` field is pre-filtered by an LLM context-cutter to minimize token usage.

4. The database is a **Bangladesh-focused Wikipedia test set** — expect content primarily about Bangladesh (government, history, geography, policy, etc.).

---

## Examples

### Example 1: Current Prime Minister

**Request:**

```json
{
  "formal_query": "বাংলাদেশের বর্তমান প্রধানমন্ত্রী?",
  "keyword_string": "বাংলাদেশ প্রধানমন্ত্রী বর্তমান দায়িত্ব"
}
```

**Response:**

```json
{
  "results": [
    {
      "title": "বাংলাদেশের প্রধানমন্ত্রী",
      "url": "https://bn.wikipedia.org/wiki/বাংলাদেশের_প্রধানমন্ত্রী",
      "context": "বাংলাদেশের প্রধানমন্ত্রী হলেন ... [condensed relevant passage]",
      "sed": [[1, 3], [5, 8]],
      "published_at": "2025-06-15"
    }
  ],
  "combined_context": "### বাংলাদেশের প্রধানমন্ত্রী\n\nবাংলাদেশের প্রধানমন্ত্রী হলেন ... [condensed relevant passage]"
}
```

---

### Example 2: July Movement 2024

**Request:**

```json
{
  "formal_query": "জুলাই আন্দোলন ২০২৪ কী এবং এর প্রধান দাবিগুলো কী কী?",
  "keyword_string": "জুলাই আন্দোলন ছাত্র প্রতিবাদ ২০২৪ সরকার পতন"
}
```

**Response:**

```json
{
  "results": [
    {
      "title": "জুলাই আন্দোলন",
      "url": "https://bn.wikipedia.org/wiki/জুলাই_আন্দোলন",
      "context": "জুলাই আন্দোলন ২০২৪ ... [condensed relevant passage]",
      "sed": [[2, 6], [9, 14]],
      "published_at": "2025-08-05"
    }
  ],
  "combined_context": "### জুলাই আন্দোলন\n\nজুলাই আন্দোলন ২০২৪ ... [condensed relevant passage]"
}
```

---

### Example 3: Padma Bridge

**Request:**

```json
{
  "formal_query": "পদ্মা সেতু কত সালে উদ্বোধন করা হয় এবং এর দৈর্ঘ্য কত?",
  "keyword_string": "পদ্মা সেতু উদ্বোধন দৈর্ঘ্য"
}
```

---

### Example 4: National ID Card

**Request:**

```json
{
  "formal_query": "জাতীয় পরিচয়পত্র বানানোর প্রক্রিয়া ও প্রয়োজনীয় কাগজপত্র কী কী?",
  "keyword_string": "জাতীয় পরিচয়পত্র কাগজপত্র প্রক্রিয়া নির্বাচন কমিশন"
}
```

---

### Example 5: Dhaka Population

**Request:**

```json
{
  "formal_query": "ঢাকা মহানগরীর আনুমানিক জনসংখ্যা কত?",
  "keyword_string": "ঢাকা জনসংখ্যা মহানগরী বাসিন্দা"
}
```

---

## Full Example Dataset

Below is a collection of common queries the API is designed to handle:

```json
[
  {
    "formal_query": "বাংলাদেশের বর্তমান প্রধানমন্ত্রী?",
    "keyword_string": "বাংলাদেশ প্রধানমন্ত্রী বর্তমান দায়িত্ব"
  },
  {
    "formal_query": "বাংলাদেশের বর্তমান রাষ্ট্রপতি কে?",
    "keyword_string": "বাংলাদেশ রাষ্ট্রপতি বর্তমান দায়িত্ব"
  },
  {
    "formal_query": "বাংলাদেশের বর্তমান পররাষ্ট্র মন্ত্রী কে?",
    "keyword_string": "বাংলাদেশ পররাষ্ট্র মন্ত্রী বর্তমান দায়িত্ব"
  },
  {
    "formal_query": "বাংলাদেশের বর্তমান তথ্য ও যোগাযোগ প্রযুক্তি মন্ত্রী কে?",
    "keyword_string": "বাংলাদেশ তথ্য যোগাযোগ প্রযুক্তি মন্ত্রী বর্তমান দায়িত্ব"
  },
  {
    "formal_query": "জুলাই আন্দোলন ২০২৪ কী এবং এর প্রধান দাবিগুলো কী কী?",
    "keyword_string": "জুলাই আন্দোলন ছাত্র প্রতিবাদ ২০২৪ সরকার পতন"
  },
  {
    "formal_query": "জুলাই আন্দোলনের নেতা ওসমান হাদি কে ছিলেন?",
    "keyword_string": "ওসমান হাদি জুলাই আন্দোলন"
  },
  {
    "formal_query": "ঢাকা মেট্রোরেলের টিকিটের মূল্য কত?",
    "keyword_string": "ঢাকা মেট্রো রেল ভাড়া টিকিট মূল্য"
  },
  {
    "formal_query": "পদ্মা সেতু কত সালে উদ্বোধন করা হয় এবং এর দৈর্ঘ্য কত?",
    "keyword_string": "পদ্মা সেতু উদ্বোধন দৈর্ঘ্য"
  },
  {
    "formal_query": "বাংলাদেশ জাতীয় সংসদে মোট আসন সংখ্যা কত?",
    "keyword_string": "জাতীয় সংসদ আসন সংখ্যা বাংলাদেশ"
  },
  {
    "formal_query": "জাতীয় পরিচয়পত্র বানানোর প্রক্রিয়া ও প্রয়োজনীয় কাগজপত্র কী কী?",
    "keyword_string": "জাতীয় পরিচয়পত্র কাগজপত্র প্রক্রিয়া নির্বাচন কমিশন"
  },
  {
    "formal_query": "ঢাকা মহানগরীর আনুমানিক জনসংখ্যা কত?",
    "keyword_string": "ঢাকা জনসংখ্যা মহানগরী বাসিন্দা"
  }
]
```

---

## Implementation Tips

- **Content type:** `application/json`
- **Timeout:** Set request timeout to at least 30 seconds (LLM processing can be slow).
- **Error handling:** On invalid input, the API returns HTTP 400 with a `detail` field. On server errors, it returns HTTP 500.
- **Curl example:**

```bash
curl -X POST http://172.22.11.241:9220/search \
  -H "Content-Type: application/json" \
  -d '{
    "formal_query": "বাংলাদেশের বর্তমান প্রধানমন্ত্রী?",
    "keyword_string": "বাংলাদেশ প্রধানমন্ত্রী বর্তমান দায়িত্ব"
  }'
```
