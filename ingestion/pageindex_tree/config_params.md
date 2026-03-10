# Configuration Parameters

This document explains each configuration parameter in the PageIndex pipeline.

## Model Configuration

### `model`
**Default:** `gpt-4o-2024-11-20`

The AI model to use for generating summaries and document descriptions. This can be:
- An OpenAI model identifier (e.g., `gpt-4o`, `gpt-4.1`)
- A local model identifier (e.g., `qwen35` when using vLLM)

**Effect:** Determines the quality of generated summaries and the document structure analysis.

---

## PDF Processing Parameters

### `toc_check_page_num`
**Default:** `20`

Number of initial pages to scan when looking for a table of contents in PDF documents.

**Effect:** Higher values allow the system to analyze more pages for TOC detection, which helps in identifying the document structure. Only used for PDF processing.

---

### `max_page_num_each_node`
**Default:** `10`

Maximum number of pages that can be grouped under a single node in the document tree.

**Effect:** Controls the granularity of the tree structure:
- Lower values create more nodes with fewer pages each
- Higher values create fewer nodes with more pages each
- Only used for PDF processing

---

### `max_token_num_each_node`
**Default:** `20000`

Maximum number of tokens allowed in the text content of a single node.

**Effect:** Limits the size of content per node:
- Lower values create more granular tree structures
- Higher values allow more content per node
- Only used for PDF processing

---

## Node Output Options

### `if_add_node_id`
**Default:** `yes`

Controls whether each node includes a unique identifier.

**Options:**
- `yes`: Each node gets a `node_id` field (e.g., "0001", "0002")
- `no`: Nodes will not have unique identifiers

**Effect:** Node IDs are useful for tracking and referencing specific sections in the output.

---

### `if_add_node_summary`
**Default:** `yes`

Controls whether each node includes an AI-generated summary of its content.

**Options:**
- `yes`: Summaries are generated for all nodes
- `no`: Nodes will not have summaries

**Effect:**
- When enabled, the system generates a Bengali description of each section
- Increases processing time but provides concise descriptions
- Summaries appear in the `summary` field for leaf nodes

---

### `if_add_doc_description`
**Default:** `no`

Controls whether the entire document gets a one-sentence overall description.

**Options:**
- `yes`: Generates a document-level description
- `no`: No document description is generated

**Effect:** When enabled, creates a distinguishing description for the entire document in the output.

---

### `if_add_node_text`
**Default:** `no`

Controls whether the full text content is included in each node.

**Options:**
- `yes`: Full text content of each node is preserved
- `no`: Only title, node_id, and summary are included

**Effect:** When disabled, creates a lighter output with just structural information and summaries.

---

## Markdown-Specific Parameters

### `if_thinning`
**Default:** `no` (in config.yaml, controlled via CLI for MD files)

Whether to apply tree thinning optimization for markdown documents.

**Options:**
- `yes`: Nodes with fewer tokens than the threshold are merged with their children
- `no`: No thinning is applied

**Effect:** Merges small sections with parent sections to create more meaningful nodes. Only used for markdown processing.

---

### `summary_token_threshold`
**Default:** `200`

Token threshold for generating node summaries.

**Effect:** Nodes with more tokens than this threshold will have AI-generated summaries. Nodes below this threshold will use their original text as the summary.

---

## Environment Variables

These variables are loaded from `.env` for local model operation:

### `OPENAI_BASE_URL`
**Example:** `http://localhost:5000/v1/`

The API endpoint for the local model server (vLLM).

---

### `CHATGPT_API_KEY`
**Example:** `OcRGXNELyOV+0zUbHg3ZvRuUz8P0qXxB5rNjJB9GhaU=`

API key for authentication. For local vLLM servers, this can be a placeholder value.

---

### `VLLM_MODEL_NAME`
**Example:** `qwen35`

The identifier for the local model being served by vLLM.

---

## Usage Summary

| Parameter | PDF Only | Markdown Only | General |
|-----------|----------|---------------|---------|
| `model` | Yes | Yes | Yes |
| `toc_check_page_num` | Yes | No | - |
| `max_page_num_each_node` | Yes | No | - |
| `max_token_num_each_node` | Yes | No | - |
| `if_add_node_id` | Yes | Yes | Yes |
| `if_add_node_summary` | Yes | Yes | Yes |
| `if_add_doc_description` | Yes | Yes | Yes |
| `if_add_node_text` | Yes | Yes | Yes |
| `if_thinning` | No | Yes | - |
| `summary_token_threshold` | No | Yes | - |
