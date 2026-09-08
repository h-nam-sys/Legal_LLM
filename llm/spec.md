# LLM Engineer Specification

**Service Name:** Legal Response Generation Service  
**Owner:** AI/ML Engineer  
**Integration Point:** Backend calls this service  
**Status:** FROZEN - Do not deviate

---

## API Endpoint

**URL:** Configurable via backend `.env` as `LLM_SERVICE_URL`  
**Default:** `http://localhost:8001`  
**Endpoint:** `POST /generate`

---

## Input Format (What Backend Sends)

**Exact JSON Schema:**
```json
{
  "prompt": "string (required)",
  "context": ["string", "string"]
}
```

**Field Definitions:**
- `prompt`: User's legal question. String. Required. Min 1 char, max 2000 chars.
  - Example: "What are the requirements for a marriage contract in Vietnam?"
  - This is the actual user question, not your prompt engineering

- `context`: Array of legal documents. Array of strings. Optional (can be empty). Max 20 items.
  - Each string is a full article from RAG service
  - Format: `"Article X, Law on Y (YEAR): [full text]"`
  - Example:
    ```
    "Article 10, Law on Marriage (2020): A marriage contract must contain the full names, dates of birth, and permanent addresses of both parties..."
    ```
  - If empty array: RAG found no relevant documents. Still provide best-effort answer but acknowledge limitation.

**Exact Example Request (With Context):**
```json
{
  "prompt": "What are the requirements for a marriage contract in Vietnam?",
  "context": [
    "Article 10, Law on Marriage (2020): A marriage contract must contain the full names, dates of birth, and permanent addresses of both parties. Both parties must be 18 years or older.",
    "Article 11, Law on Marriage (2020): Both parties must present valid identification documents including passport or national ID card. Documents must be verified by local authorities.",
    "Article 15, Law on Marriage (2020): A marriage contract is valid only if witnessed by two authorized officials and signed by both parties in the presence of witnesses."
  ]
}
```

**Exact Example Request (No Context):**
```json
{
  "prompt": "What are the requirements for a marriage contract in Vietnam?",
  "context": []
}
```

---

## Output Format (What You Must Return)

**Exact JSON Schema:**
```json
{
  "answer": "string (required)",
  "confidence": 0.85,
  "sources": ["string", "string"]
}
```

**Field Definitions:**

### `answer` (Required)
- Type: String
- Min length: 10 chars
- Max length: 5000 chars (HARD LIMIT - backend will truncate if exceeded)
- Content: Concise, accurate legal advice in Vietnamese or English
- Quality: Cite specific articles from provided context
- Tone: Professional, direct, no fluff
- Structure: Short sentences, clear bullet points if needed
- If no context provided: Start with "Insufficient legal information provided, but based on general knowledge:"

**Exact Example Answer:**
```
According to Vietnamese law, a marriage contract must include: (1) Full names and dates of birth of both parties, (2) Permanent addresses of both parties, (3) Valid identification documents (passport or national ID), (4) Two authorized witnesses, (5) Signatures of both parties in presence of witnesses. Both parties must be 18 years or older. The contract must be registered within 30 days of signing.
```

### `confidence` (Required)
- Type: Float
- Range: 0.0 to 1.0
- Meaning:
  - 0.95-1.0: Highly confident, well-grounded in provided documents
  - 0.7-0.94: Confident, based on legal context
  - 0.5-0.69: Moderately confident, some extrapolation
  - 0.3-0.49: Low confidence, mostly from training data
  - 0.0-0.29: Very low confidence, potentially unreliable
- Example: If context contains all relevant articles, use 0.90+
- Example: If context is empty, use 0.40-0.60 (uncertain)
- Example: If answering outside scope, use 0.20-0.40

**Exact Example Confidence:**
```json
"confidence": 0.92
```

### `sources` (Required, Can Be Empty)
- Type: Array of strings
- Min length: 0 items (empty array if no sources used)
- Max length: 20 items
- Format: `"Article X, Law on Y YEAR"` (must match format below exactly)
- Content: Which documents from context were used to generate answer

**Exact Format for Each Source:**
```
Article NUMBER, Law Name YEAR
```

**Format Rules:**
- MUST include: Article number, Law name, Year
- NO parentheses in year: `2020` NOT `(2020)`
- NO colons: `Law on Marriage 2020` NOT `Law on Marriage: 2020`
- Full law name (not abbreviated)

**Examples of Correct Format:**
- ✅ `"Article 10, Law on Marriage 2020"`
- ✅ `"Article 10.1, Law on Marriage 2020"`
- ✅ `"Article 15, Decree on Marriage Administration 2021"`
- ✅ `"Section 5, Article 3, Ordinance on Business Registration 2019"`

**Examples of WRONG Format:**
- ❌ `"Article 10 (Law on Marriage 2020)"`
- ❌ `"Article 10: Law on Marriage (2020)"`
- ❌ `"Art. 10, Marriage Law 2020"`
- ❌ `"doc_id_001"`
- ❌ `"Law on Marriage page 5"`

**Exact Example Sources:**
```json
"sources": [
  "Article 10, Law on Marriage 2020",
  "Article 11, Law on Marriage 2020",
  "Article 15, Law on Marriage 2020"
]
```

---

## Complete Output Example

**Full Response:**
```json
{
  "answer": "According to Vietnamese law, a marriage contract must include: (1) Full names and dates of birth of both parties, (2) Permanent addresses of both parties, (3) Valid identification documents (passport or national ID), (4) Two authorized witnesses, (5) Signatures of both parties in presence of witnesses. Both parties must be 18 years or older. The contract must be registered within 30 days of signing.",
  "confidence": 0.92,
  "sources": [
    "Article 10, Law on Marriage 2020",
    "Article 11, Law on Marriage 2020",
    "Article 15, Law on Marriage 2020"
  ]
}
```

---

## Response Quality Requirements (CRITICAL)

**You MUST ensure:**

1. **Legal Accuracy (Non-Negotiable)**
   - Answer must be legally accurate per provided documents
   - Do NOT make up laws or regulations
   - Do NOT confuse or misrepresent articles
   - If unsure, lower confidence score

2. **Citation Discipline**
   - Every factual claim should reference a source
   - Sources array must include all documents used
   - Do NOT cite documents not in sources array

3. **Hallucination Prevention**
   - If context is empty: Answer "Insufficient legal information..."
   - If context doesn't cover question: Say "This specific issue is not addressed in provided documents..."
   - Do NOT invent details to fill gaps

4. **Length Constraint**
   - Keep answer focused and concise ("đúng trọng tâm")
   - Max 5000 chars (backend will truncate)
   - Typically 200-800 chars for legal questions
   - Avoid filler, be direct

5. **Language**
   - Vietnamese legal terminology preferred
   - If mixing Vietnamese/English: keep terminology consistent
   - Avoid slang or casual language

6. **Structure**
   - Use numbered lists for multiple points
   - Short sentences (< 20 words each)
   - One idea per sentence
   - Clear hierarchy (main points → sub-points)

---

## Context Parsing Rules

**You MUST parse documents in this format:**

**What you receive:**
```
Article 10, Law on Marriage (2020): A marriage contract must contain the full names, dates of birth, and permanent addresses of both parties. Both parties must be 18 years or older.
```

**How to parse:**
1. Extract: `Article 10`
2. Extract: `Law on Marriage`
3. Extract: `2020`
4. Extract: Full text after colon
5. Use all three in your citation: `Article 10, Law on Marriage 2020`

**Do NOT:**
- Change the format
- Abbreviate the law name
- Alter the article number
- Add extra punctuation

---

## Error Handling

### Normal Cases

**Case 1: Context provided, clear answer**
```json
{
  "answer": "According to Article 10, Law on Marriage 2020...",
  "confidence": 0.92,
  "sources": ["Article 10, Law on Marriage 2020"]
}
```

**Case 2: Context provided, but incomplete**
```json
{
  "answer": "Based on provided documents, marriage contracts require Article 10 elements. However, specific requirements for X are not addressed in the retrieved documents. Additional legal consultation may be needed.",
  "confidence": 0.65,
  "sources": ["Article 10, Law on Marriage 2020"]
}
```

**Case 3: Empty context (no documents found)**
```json
{
  "answer": "Insufficient legal information provided. However, based on general Vietnamese law knowledge, marriage contracts typically require: (1) Full names and DOB, (2) Valid ID documents, (3) Witness signatures. For precise requirements, a formal legal document search is recommended.",
  "confidence": 0.45,
  "sources": []
}
```

**Case 4: Question outside legal domain**
```json
{
  "answer": "This question is outside the scope of Vietnamese legal consulting. This service addresses legal questions only. Please consult appropriate domain experts for this topic.",
  "confidence": 0.20,
  "sources": []
}
```

### Service Errors

**Service Error (Model crash, OOM, etc.):**
Return HTTP 500:
```json
{
  "error": "Model inference failed: [reason]"
}
```
Backend will handle gracefully.

**Invalid Input (Missing prompt field):**
Return HTTP 400:
```json
{
  "error": "Prompt field is required"
}
```

---

## Performance Requirements

- **Response Time SLA:** ≤ 60 seconds
  - Backend timeout: 60 seconds (hardcoded)
  - If you exceed 60s, backend treats as timeout/error
  
- **Throughput:** Assume < 100 queries/day during sprint (no optimization needed yet)

- **Streaming:** Return full response only (no streaming)
  - Backend expects complete JSON object in one response
  - No server-sent events or chunked transfer

- **Latency Breakdown (Target):**
  - Token generation: ≤ 40 seconds
  - Response serialization: ≤ 1 second
  - Total: ≤ 41 seconds target (gives 19s buffer)

---

## Prompt Engineering Template (For Your Reference)

You'll need to design a prompt that:

1. **Instructs model to use context:**
   ```
   "Use the following Vietnamese legal documents to answer the question. 
   Only cite facts from these documents. 
   If information is not in the documents, say so explicitly."
   ```

2. **Formats context:**
   ```
   "Legal Documents:
   - Article 10, Law on Marriage (2020): [text]
   - Article 11, Law on Marriage (2020): [text]"
   ```

3. **Forces JSON output:**
   ```
   "Respond in the following JSON format:
   {
     "answer": "...",
     "confidence": 0.X,
     "sources": ["Article X, Law on Y YEAR"]
   }"
   ```

4. **Constrains length:**
   ```
   "Keep your answer to maximum 500 characters. Be concise and direct."
   ```

**Your responsibility:** Design this template so output matches the exact JSON format above.

---

## Coordination with RAG Engineer

**RAG Service returns documents in this format:**
```json
{
  "documents": [
    "Article 10, Law on Marriage (2020): A marriage contract must contain...",
    "Article 11, Law on Marriage (2020): Both parties must present..."
  ],
  "scores": [0.95, 0.87]
}
```

**You MUST:**
1. Parse this exact format in your prompt
2. Use the article number, law name, year for citations
3. Return sources in matching format: `"Article X, Law on Y YEAR"`
4. **Do NOT add parentheses or change punctuation**

---

## Testing Criteria

To verify your service meets spec:

**Test 1: Correct JSON Format**
```bash
curl -X POST http://localhost:8001/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is a marriage contract?", "context": ["Article 10, Law on Marriage (2020): A marriage contract must contain..."]}'
```
Expected: Response has `answer`, `confidence`, `sources` fields in exact format.

**Test 2: Source Citation Format**
- Check that every source matches: `"Article X, Law on Y YEAR"`
- NO parentheses in year
- NO colons or extra punctuation

**Test 3: Length Constraint**
- Measure answer string length
- Must be ≤ 5000 chars
- Typical answers 200-800 chars

**Test 4: Latency**
- Measure end-to-end response time
- Must be ≤ 60 seconds
- Aim for ≤ 5 seconds

**Test 5: Empty Context**
```bash
curl -X POST http://localhost:8001/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is a marriage contract?", "context": []}'
```
Expected: Response acknowledges "Insufficient legal information" but still provides answer.

---

## Deployment

- **Local Development:** http://localhost:8001
- **Inference Engine:** Your choice (vLLM, llama.cpp, etc.)
- **Model:** Your fine-tuned Vietnamese legal model
- **Quantization:** Your choice (GGUF, AWQ, etc.)
- **Persistence:** Model weights must be available on service restart

---

## Questions? Blocked?

- Check format examples above (exact match required)
- Verify JSON schema against examples (test with JSON validator)
- Ask backend lead in daily standup
- Do NOT deviate from this spec without approval