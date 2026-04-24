# Generic Graph Exploration Tools for CogOpsCB

## Graph Schema Dump (For Use by Any Agent)

This dump describes the Neo4j graph schema, access patterns, and query conventions so any agent can explore the graph in any fashion.

### Database Connection
- **URI**: `bolt+ssc://localhost:7687`
- **Database**: `qwen34neo4j`
- **User**: `neo4j`
- **Password**: From `.env` → `NEO4J_PASSWORD`
- **Docker container**: `neo4j_local_secure`

### Graph Statistics
| Metric | Count |
|--------|-------|
| Entity nodes | 11,143 |
| Episodic nodes | 1,223 |
| RELATES_TO edges | 28,733 |
| Distinct relation-name values | 2,041 |
| Entities with embeddings | 11,143 (100%) |

### Node Types

#### 1. `:Entity` (11,143 nodes)
Properties:
| Property | Type | Description |
|----------|------|-------------|
| `uuid` | string (UUID) | Unique identifier |
| `name` | string | Display name (Bengali/English) |
| `summary` | string | Brief description |
| `name_embedding` | float[] | Vector embedding for similarity search |
| `labels` | list | Always empty list `[]` |
| `group_id` | string | Grouping identifier |
| `created_at` | datetime | Creation timestamp |

**Query pattern:**
```cypher
MATCH (e:Entity {name: $name}) RETURN e.uuid, e.name, e.summary, e.name_embedding
MATCH (e:Entity) WHERE toLower(e.name) CONTAINS toLower($partial) RETURN e.name, e.summary ORDER BY ... LIMIT 10
```

**Vector similarity:**
```cypher
MATCH (n:Entity) WITH n, vector.similarity.cosine(n.name_embedding, $search_vector) AS score
WHERE score > 0.5 RETURN n.name, n.summary, score ORDER BY score DESC LIMIT 10
```
No pre-built vector index — cosine computed on-the-fly across all 11,143 nodes.

#### 2. `:Episodic` (1,223 nodes)
Properties:
| Property | Type | Description |
|----------|------|-------------|
| `uuid` | string (UUID) | Unique identifier |
| `name` | string | Episodic identifier |
| `content` | string (JSON) | Raw passage data |
| `source` | string | Source URL |
| `source_description` | string | Source description |
| `entity_edges` | string[] (UUIDs) | UUIDs of related Entity nodes (NOT Neo4j relationships) |
| `valid_at` | datetime | Validity timestamp |
| `created_at` | datetime | Creation timestamp |
| `group_id` | string | Grouping identifier |

**`content` JSON structure:**
```json
{
  "category": "জন্ম নিবন্ধন",
  "sub_category": "জন্ম নিবন্ধন",
  "service": "জন্ম নিবন্ধনের আবেদন",
  "alternate_variants": "জন্মনিবন্ধন, জন্ম রেজিস্ট্রেশন, বার্থ সার্টিফিকেট, ...",
  "keyword": "জন্ম নিবন্ধন আবেদন",
  "topic": "জন্ম নিবন্ধন আবেদন প্রক্রিয়া",
  "text": "Full passage text with embedded Category/Sub-Category/Service/Topic metadata",
  "text_keywords": "keywords separated by pipes",
  "url": "source URL(s)"
}
```

**CRITICAL: Episodic nodes are NOT connected to Entity nodes via Neo4j relationships.** They are standalone nodes. The connection is stored as embedded UUIDs in `entity_edges` property. The Graphiti `_search()` method handles this via its internal indexing.

### Edge Type: `RELATES_TO` (28,733 edges)

**CRITICAL DISCOVERY:** The graph uses a SINGLE Neo4j relationship type `RELATES_TO` for ALL semantic relationships. The meaning is stored in the `r.name` property on the edge.

Edge properties:
| Property | Type | Description |
|----------|------|-------------|
| `uuid` | string (UUID) | Unique identifier |
| `name` | string | **Semantic relation type** (2,041 distinct values) |
| `fact` | string | Relationship text in Bengali/English |
| `fact_embedding` | float[] | Vector embedding for the fact |
| `created_at` | datetime | Creation timestamp |
| `group_id` | string | Grouping identifier |

**This is the Graphiti library pattern** — all relationships use a single `RELATES_TO` edge type. The `r.name` property carries the semantic meaning.

**Query pattern:**
```cypher
// WRONG — this won't work:
MATCH (a)-[:RELATES_DOCUMENTS]->(b)

// CORRECT — use WHERE on r.name:
MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity)
WHERE r.name = 'REQUIRES_DOCUMENT'
RETURN a.name, r.fact, b.name
```

### Top Relation Types by Frequency

| Relation Name | Count | Category |
|--------------|-------|----------|
| REQUIRES | 3,744 | Process/requirement |
| REQUIRES_DOCUMENT | 2,163 | Document requirement |
| IS_A | 855 | Type hierarchy |
| INVOLVES | 723 | Process step |
| HAS_ALTERNATIVE_NAME | 679 | Name variants |
| PROVIDES_SERVICE | 630 | Service provision |
| HAS_ALTERNATE_NAME | 485 | Name variants |
| INCLUDES | 412 | Process inclusion |
| HAS_ALTERNATE_VARIANT | 392 | Name variants |
| HAS_URL | 365 | URL reference |
| IS_PART_OF | 355 | Part-of hierarchy |
| HAS_SERVICE | 340 | Service association |
| HAS_FEE | 291 | Fee/cost |
| HAS_SOURCE | 277 | Source reference |
| APPLIES_TO | 266 | Applicability |
| HAS_PART | 262 | Composition |
| RELATED_TO | 257 | General relation |
| PROVIDES | 241 | Service provision |
| HAS_EXEMPTION | 233 | Exemption info |
| ALTERNATIVE_NAME | 211 | Name variants |
| HAS_KEYWORD | 209 | Keyword tagging |
| SUPPORTED_BY | 202 | Support |
| HAS_STEP | 195 | Process step |
| HAS_LOCATION | 189 | Location |
| ASSOCIATED_WITH | 185 | Association |
| HAS_FEATURE | 180 | Feature |
| HAS_TYPE | 170 | Type classification |
| SUPPORTS | 169 | Support |
| USED_FOR | 165 | Purpose |
| PAYMENT_METHOD | 159 | Payment info |

**Total: 2,041 distinct relation-name values.** The full distribution spans from high-frequency ones (REQUIRES at 3,744) down to dozens with 1-2 edges each.

### Top Entities by Degree (connections)

| Entity Name | Degree |
|-------------|--------|
| উত্তরাধিকার/সাকসেশন সনদ | 217 |
| https://mofa.gov.bd/ | 213 |
| আবেদন | 209 |
| প্রয়োজনীয় কাগজপত্র | 207 |
| করদাতা | 196 |
| https://nbr.gov.bd/all-faq/eng | 193 |
| বয়স সংশোধন | 181 |
| প্রত্যয়নপত্র | 173 |
| জাতীয় অর্থোপেডিক হাসপাতাল ও পুনর্বাসন কেন্দ্র | 170 |
| জন্ম নিবন্ধন সনদ | 166 |

### Key Semantic Clusters

**Name Variants (alias clusters):**
- `HAS_ALIAS`, `ALIAS_OF`, `ALTERNATIVE_NAME`, `HAS_ALTERNATE_NAME`, `HAS_ALTERNATIVE_NAME`, `IS_ALTERNATIVE_NAME_OF`, `HAS_ALTERNATE_VARIANT`, `ALSO_KNOWN_AS`, `RELATED_TO`
- Example: "Death Certificate" → "মৃত্যু সনদ" → "ডেথ সার্টিফিকেট" → "মৃত্যু নিবন্ধন সনদ"

**Service Provision:**
- `PROVIDES_SERVICE` (630), `PROVIDES` (241), `PROVIDED_BY` (84), `ISSUED_BY` (81), `ISSUES_CERTIFICATE`

**Documents/Requirements:**
- `REQUIRES_DOCUMENT` (2,163), `REQUIRES` (3,744), `REQUIRES_RETURN_FILING`, `HAS_REQUIREMENT`, `REQUIRED_FOR`

**Process Steps:**
- `HAS_STEP` (195), `INCLUDES` (412), `INVOLVES` (723), `HAS_PART` (262)

**Hierarchical:**
- `IS_A` (855), `IS_PART_OF` (355), `HAS_PART` (262), `HAS_SUBTYPE` (147), `IS_TYPE_OF` (118), `HAS_CATEGORY` (139), `IS_CATEGORY_OF` (94)

**Fees/Cost:**
- `HAS_FEE` (291), `PAYMENT_METHOD` (159), `HAS_EXEMPTION` (233)

**URL/Source:**
- `HAS_URL` (365), `HAS_SOURCE` (277), `HAS_KEYWORD` (209), `HAS_HOTLINE_NUMBER` (111)

**Location:**
- `LOCATED_IN` (36), `LOCATED_AT` (16)

## Generic Tool Design

The tools are generic — they work on ANY part of the graph, regardless of domain. The LLM decides WHICH relation types, entities, or paths to explore based on the user's query.

### Tool 1: `graph_search` (improved existing)
- **Purpose**: Hybrid search (BM25 + vector + BFS) with cross_encoder reranking
- **Returns**: Nodes, Edges, Passages formatted as Markdown
- **Params**: `query` (string), `limit` (int, default 5), `min_score` (float, default 0.8)
- **When to use**: When user asks a question and you need to find relevant information
- **Cypher**: Uses Graphiti `client._search()` with `COMBINED_HYBRID_SEARCH_CROSS_ENCODER`

### Tool 2: `node_explore`
- **Purpose**: Given an entity name, return ALL connections (incoming + outgoing) with relation types, facts, and neighbor summaries
- **Returns**: Markdown table grouped by relation type, showing all neighbors
- **Params**: `entity_name` (string), `max_results` (int, default 100)
- **When to use**: "Show me everything about Death Certificate" or "What does birth registration connect to?"
- **Cypher**:
```cypher
MATCH (e:Entity {name: $name})<-[:RELATES_TO]-(neighbor:Entity)-[:RELATES_TO]->(e)
RETURN neighbor.name AS other_entity,
       [r IN relationships(path) | {type: r.name, fact: r.fact}] AS relations
ORDER BY other_entity
LIMIT $max
```

### Tool 3: `relation_browse`
- **Purpose**: List ALL available relation-name values with their edge counts, optionally filtered by category prefix
- **Returns**: Markdown table of relation types sorted by frequency
- **Params**: `filter_prefix` (string, optional), `top_n` (int, default 100)
- **When to use**: "What relation types exist?" or "Show me the most common relationship patterns"
- **Cypher**:
```cypher
MATCH ()-[r:RELATES_TO]->() WHERE r.name IS NOT NULL
WITH r.name AS relName, count(*) AS cnt
WHERE $filterPrefix IS NULL OR r.name STARTS WITH $filterPrefix
ORDER BY cnt DESC LIMIT $topN
RETURN relName, cnt
```

### Tool 4: `relation_filter`
- **Purpose**: Given a relation name, return ALL entity pairs connected by it with facts
- **Returns**: Markdown table of source → target pairs with facts, grouped by relation name
- **Params**: `relation_name` (string), `max_results` (int, default 50)
- **When to use**: "Show me all REQUIRES_DOCUMENT relationships" or "What things have fees?"
- **Cypher**:
```cypher
MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity)
WHERE r.name = $relName
RETURN a.name AS source, b.name AS target, r.fact AS fact
ORDER BY a.name, b.name LIMIT $maxResults
```

### Tool 5: `entity_search`
- **Purpose**: Find entities by partial/fuzzy name match, ranked by match quality
- **Returns**: Ranked list of entities with names, summaries, and match rank
- **Params**: `search_term` (string), `max_results` (int, default 10)
- **When to use**: "Find entity named 'death'" or "Search for passport-related entities"
- **Cypher**:
```cypher
MATCH (e:Entity)
WHERE toLower(e.name) CONTAINS toLower($term)
RETURN e.name, e.summary,
       CASE WHEN e.name = $term THEN 0
            WHEN toLower(e.name) = toLower($term) THEN 1
            ELSE 2 END AS match_rank
ORDER BY match_rank, e.name LIMIT $maxResults
```

### Tool 6: `entity_detail`
- **Purpose**: Get full details of a specific entity by exact name or UUID
- **Returns**: Entity properties (name, summary, UUID, group_id, created_at)
- **Params**: `identifier` (string, name or UUID)
- **When to use**: "Tell me everything about this specific entity"
- **Cypher**:
```cypher
MATCH (e:Entity)
WHERE e.name = $id OR e.uuid = $id
RETURN e.uuid, e.name, e.summary, e.group_id, e.created_at
```

### Tool 7: `similar_entities`
- **Purpose**: Given an entity name, find semantically similar entities via vector cosine similarity
- **Returns**: Ranked list of similar entities with similarity scores
- **Params**: `entity_name` (string), `max_results` (int, default 10), `min_score` (float, default 0.5)
- **When to use**: "Find entities similar to Death Certificate"
- **Cypher**:
```cypher
MATCH (target:Entity {name: $name}) RETURN target.name_embedding AS vec, target.uuid AS exclude
WITH vec, exclude LIMIT 1
MATCH (n:Entity)
WHERE n.uuid <> exclude
WITH n, vector.similarity.cosine(n.name_embedding, vec) AS score
WHERE score >= $minScore
RETURN n.name, n.summary, score
ORDER BY score DESC LIMIT $maxResults
```

### Tool 8: `path_find`
- **Purpose**: Find paths between two entities (1-N hops) showing entity chains and relation type chains
- **Returns**: Numbered paths with entity names and relation types between them
- **Params**: `start_entity` (string), `end_entity` (string), `max_hops` (int, default 3), `max_paths` (int, default 5)
- **When to use**: "Show me the path from Birth Registration to Passport application"
- **Cypher**:
```cypher
MATCH path=(start:Entity {name: $start})-[:RELATES_TO*1..$maxHops]-(end:Entity {name: $end})
WHERE start <> end
WITH path, [n IN nodes(path) | n.name] AS chain,
            [r IN relationships(path) | r.name] AS relTypes
ORDER BY size(chain) LIMIT $maxPaths
RETURN chain, relTypes
```

### Tool 9: `episodic_search`
- **Purpose**: Search raw passage data in Episodic nodes by text content or metadata fields
- **Returns**: Matching passages with category, service, topic, URL, and text snippets
- **Params**: `search_term` (string), `field` (string: "text"|"category"|"service"|"topic"|"keyword", default "text"), `max_results` (int, default 10)
- **When to use**: "Find passages about fee amounts" or "Search for birth registration procedure text"
- **Cypher**:
```cypher
MATCH (ep:Episodic)
WHERE toLower(ep.content) CONTAINS toLower($term)
RETURN ep.content, ep.url, ep.source
ORDER BY ep.created_at DESC LIMIT $maxResults
```

### Tool 10: `graph_stats`
- **Purpose**: Get graph-level statistics — node counts, relation type distribution, degree distribution
- **Returns**: Markdown summary of graph statistics
- **Params**: `detail_level` (string: "basic"|"detailed", default "basic")
- **When to use**: "How big is the graph?" or "What's the most common relation type?"
- **Cypher**: Multiple queries combined

---

## Implementation Steps

### Step 1: Add helpers and tools to `cogops/tools/graphiti_tools.py`

1. Add `resolve_entity(driver, entity_name)` helper — finds entity by exact name first, then partial match via Cypher. Returns `{uuid, name, summary}`.

2. Add 10 tool async functions — each returns Markdown string optimized for LLM consumption:
   - Tables for tabular data (relation_filter, node_explore, relation_browse)
   - Numbered lists for paths (path_find)
   - Structured blocks for entity details (entity_detail)
   - Ranked lists for similarity (similar_entities)
   - Markdown passages for episodic (episodic_search)

3. Append 9 new schemas to `graphiti_tools_list` (graph_search already exists)

4. Append 9 new entries to `available_tools_map`

### Step 2: Update prompt in `cogops/prompts/graphiti_prompt.py`

- Section 3 (Tool Usage Doctrine): Update tool trigger rules to reference all 10 tools
- Add guidance on when to use each tool:
  - `graph_search` → general information retrieval
  - `entity_search` → finding an entity when user mentions a partial name
  - `entity_detail` → when user asks about a specific known entity
  - `node_explore` → when user wants to see all connections of an entity
  - `relation_filter` → when user asks about a specific relation type
  - `relation_browse` → when user wants to explore available relation types
  - `similar_entities` → when user asks about related/similar concepts
  - `path_find` → when user asks about connections between two entities
  - `episodic_search` → when user wants raw passage/procedure text
  - `graph_stats` → when user asks about the graph itself

### Step 3: Add config in `configs/config.yml`

- `system_prompt_reservation: 3000` (up from 1600, for ~1400 tokens of 10 tool schemas)
- Per-tool config sections: `node_explore`, `relation_browse`, `relation_filter`, `entity_search`, `entity_detail`, `similar_entities`, `path_find`, `episodic_search`, `graph_stats`

### Step 4: Add REST endpoints in `api.py`

- Routes `/api/graph/{tool_name}` for each of 10 tools
- Parse query params, call tool function, return JSON
- Use existing `get_graphiti_client()` pattern from `graphiti_tools.py`

---

## Key Design Decisions

1. **Relation-name driven**: All Cypher queries use `WHERE r.name = 'X'` because Graphiti stores relation semantics as a property on single `RELATES_TO` edges. This is the fundamental constraint.

2. **No domain assumptions**: Tools don't assume certificates, boards, or institutes. They work on ANY entity and ANY relation type. The LLM decides which combinations to query.

3. **Markdown output**: All tools return Markdown optimized for LLM consumption. Tables, lists, and structured blocks make it easy for the LLM to synthesize Bengali responses.

4. **2,041 relation types**: The graph has a huge vocabulary of relation-name values. The `relation_browse` tool lets agents discover what's available. The LLM can use this to pick the right relation for any query.

5. **Episodic isolation**: Episodic nodes are NOT connected via Neo4j edges to Entity nodes. They contain embedded UUIDs of related entities. The `episodic_search` tool queries them directly by content. The Graphiti `_search()` method handles this via internal indexing.

6. **Vector similarity on demand**: No pre-built vector index. Cosine similarity computed on-the-fly across all 11,143 Entity nodes. Acceptable latency (~1-2s per query) for exploration use.

7. **Resolution strategy**: Entity lookup tries exact name match first, then case-insensitive exact match, then partial CONTAINS match. This handles Bengali/English name variations.
