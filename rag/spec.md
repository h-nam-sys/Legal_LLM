# RAG Engineer Specification

**Service Name:** Legal Document Retrieval Service  
**Owner:** Data & RAG Engineer  
**Integration Point:** Backend calls this service  
**Status:** FROZEN - Do not deviate

---

## API Endpoint

**URL:** Configurable via backend `.env` as `RAG_SERVICE_URL`  
**Default:** `http://localhost:8002`  
**Endpoint:** `POST /retrieve`

---

## Input Format (What Backend Sends)

**Exact JSON Schema:**
```json
{
  "query": "string (required)",
  "top_k": 5
}
```

**Field Definitions:**
- `query`: User's legal question. String. Required. Min 1 char, max 2000 chars.
  - Example: "What are the requirements for a marriage contract in Vietnam?"
- `top_k`: Number of top documents to return. Integer. Optional. Default 5. Range 1-20.
  - Backend always sends 5. Do not change without backend approval.

**Example Request:**
```json
{
  "query": "What are the requirements for a marriage contract in Vietnam?",
  "top_k": 5
}
```

---

## Output Format (What You Must Return)

**Exact JSON Schema:**
```json
{
  "documents": ["string", "string", "string"],
  "scores": [0.95, 0.87, 0.76]
}
```

**Field Definitions:**
- `documents`: Array of strings. Required. Each string is a full article/clause text.
  - Format: `"[ARTICLE_NUMBER], [LAW_NAME] ([YEAR]): [FULL_TEXT]"`
  - Example: `"Article 10, Law on Marriage (2020): A marriage contract must contain the full names, dates of birth, and permanent addresses of both parties..."`
  - Min length: 50 chars per document
  - Max length: 5000 chars per document
  - Total response size: < 100KB
  - No duplicates
  - Sorted by relevance (highest first)

- `scores`: Array of floats. Required. Relevance scores for each document.
  - Each score: 0.0 to 1.0
  - Must be same length as `documents` array
  - Sorted descending (highest first)
  - Example: [0.95, 0.87, 0.76, 0.65, 0.52]

**Exact Example Response:**
```json
{
  "documents": [
    "Article 10, Law on Marriage (2020): A marriage contract must contain the full names, dates of birth, and permanent addresses of both parties. Both parties must be 18 years or older.",
    "Article 11, Law on Marriage (2020): Both parties must present valid identification documents including passport or national ID card. Documents must be verified by local authorities.",
    "Article 15, Law on Marriage (2020): A marriage contract is valid only if witnessed by two authorized officials and signed by both parties in the presence of witnesses.",
    "Article 20, Law on Marriage (2020): Amendments to marriage contracts must follow the same procedures as initial contract creation and require re-registration.",
    "Decree 03/2021 on Marriage Administration: Marriage contracts must be registered within 30 days of signing. Late registration incurs a 500,000 VND penalty."
  ],
  "scores": [0.95, 0.87, 0.76, 0.65, 0.52]
}
```

---

## Document Formatting Rules (CRITICAL)

**MUST follow this format EXACTLY:**

### Format String:
```
[ARTICLE_NUMBER], [LAW_NAME] ([YEAR]): [FULL_TEXT]
```

### Components:
- `[ARTICLE_NUMBER]`: Article X, Section Y, Clause Z (as applicable)
  - Examples: "Article 10", "Article 10.1", "Section 5, Article 3"
- `[LAW_NAME]`: Full official law name
  - Examples: "Law on Marriage", "Decree on Marriage Administration", "Ordinance on Business Registration"
- `[YEAR]`: Year of law passage/amendment
  - Format: YYYY (e.g., 2020)
- `[FULL_TEXT]`: Complete article text, no abbreviations
  - Write out the full text, not a summary
  - Preserve original Vietnamese legal terminology
  - If > 5000 chars, split into multiple documents

### NO Format Exceptions:
❌ Bad: `"Marriage law says: must be 18 years old"`  
❌ Bad: `"doc_001_article_10"`  
❌ Bad: `"Article 10: Marriage requirements (from page 5)"`  
✅ Good: `"Article 10, Law on Marriage (2020): A marriage contract must contain..."`

---

## Data Quality Requirements

**You MUST ensure:**

1. **Legal Accuracy**
   - Return actual Vietnamese legal text, not summaries or translations
   - Cite official government sources only
   - No outdated/repealed laws
   - Include amendment status if applicable

2. **Relevance**
   - Retrieved documents must directly answer the user's query
   - Highest relevance scores first (sorted 0.95 → 0.52)
   - No irrelevant documents (score should reflect true relevance)

3. **No Duplicates**
   - Same article cannot appear twice in one response
   - If article appears in multiple contexts, return only once

4. **Complete Text**
   - Return full article text, not excerpts
   - If article is very long (>5000 chars), split into logical chunks:
     - Chunk 1: "Article 10, Part A..."
     - Chunk 2: "Article 10, Part B..."

5. **Metadata Included**
   - Each document must include: article number, law name, year
   - Do NOT omit this info

---

## Error Handling

### No Results Found
**Return:**
```json
{
  "documents": [],
  "scores": []
}
```
Backend will gracefully handle by notifying LLM service.

### Service Error (Database Down, etc.)
**Return HTTP 500:**
```json
{
  "error": "Vector database connection failed"
}
```
Backend will retry or gracefully degrade.

### Invalid Input (Missing query, etc.)
**Return HTTP 400:**
```json
{
  "error": "Query field is required"
}
```

---

## Performance Requirements

- **Response Time SLA:** ≤ 30 seconds
  - Backend timeout: 30 seconds (hardcoded)
  - If you exceed 30s, backend treats as timeout/error
  
- **Throughput:** Assume < 100 queries/day during sprint (no optimization needed yet)

- **Latency Breakdown:**
  - Vector search: ≤ 2 seconds
  - Scoring/ranking: ≤ 1 second
  - Response serialization: ≤ 0.5 seconds
  - Total: ≤ 3.5 seconds target (gives 26.5s buffer)

---

## Chunking Strategy (For Your Planning)

Coordinate with LLM Engineer on how to chunk documents:

**Recommended:**
- Chunk by Article (not by token count)
- Each article = one document in response
- Preserve full text of each article
- Include sub-clauses as part of same document

**NOT:**
- Random 512-token chunks (Vietnamese law articles don't split well)
- Sentence-level splits (loses legal context)
- Multi-article merges (loses retrieval precision)

---

## Coordination with LLM Engineer

**Document Format You Provide → LLM Engineer Must Parse:**

You return:
```json
{
  "documents": [
    "Article 10, Law on Marriage (2020): A marriage contract must contain..."
  ]
}
```

LLM Engineer must:
1. Parse this exact format in their prompt template
2. Extract the article number, law name, year
3. Use this to cite back in their `sources` field
4. **Exact citation format they must return:**
   ```json
   "sources": ["Article 10, Law on Marriage 2020"]
   ```

**You must match this format. They must parse it. Non-negotiable.**

---

## Testing Criteria

To verify your service meets spec:

**Test 1: Correct Format**
```bash
curl -X POST http://localhost:8002/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "marriage contract requirements", "top_k": 5}'
```
Expected: Response matches exact format above (5 documents, 5 scores, same length).

**Test 2: Document Quality**
- Each document starts with "Article X, Law on Y (YEAR):"
- Full text follows, no abbreviations
- Score is relevant to query

**Test 3: Latency**
- Measure end-to-end response time
- Must be ≤ 30 seconds
- Aim for ≤ 5 seconds

**Test 4: Empty Results**
```bash
curl -X POST http://localhost:8002/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "completely nonsensical query xyz", "top_k": 5}'
```
Expected: `{"documents": [], "scores": []}`

---

## Deployment

- **Local Development:** http://localhost:8002
- **Production:** Update backend `.env` with your service URL
- **Database:** Your choice (Qdrant, Milvus, Chroma, etc.)
- **Embedding Model:** Your choice, but must handle Vietnamese legal text well
- **Persistence:** Data must survive service restart

---

## Questions? Blocked?

- Check format examples above (exact match required)
- Ask backend lead in daily standup
- Do NOT deviate from this spec without approval