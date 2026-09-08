# Backend Engineer Specification

**Service Name:** Legal LLM Orchestration Backend  
**Owner:** Team Lead & Backend Engineer  
**Status:** IMPLEMENTED - Frozen API contracts

---

## Core Responsibility

You orchestrate the entire flow:
1. Receive user question
2. Call RAG service (/retrieve)
3. Call LLM service (/generate with RAG results)
4. Save conversation to database
5. Return response to user

---

## User-Facing API (What You Serve)

### 1. Start Conversation

**Endpoint:** `POST /start-conversation`

**Request:**
```json
{
  "user_id": "string (1-255 chars)",
  "title": "string (optional, default: 'Legal Consultation')"
}
```

**Response (200):**
```json
{
  "conversation_id": 123,
  "title": "Legal Consultation",
  "created_at": "2026-09-08T10:30:00"
}
```

---

### 2. Send Message

**Endpoint:** `POST /chat/{conversation_id}`

**Request:**
```json
{
  "user_prompt": "string (1-2000 chars)"
}
```

**Response (200):**
```json
{
  "answer": "string (≤5000 chars)",
  "status": "success",
  "sources": ["Article 10, Law on Marriage 2020"]
}
```

**Flow:**
1. Validate prompt (not empty, ≤2000 chars)
2. Save user message to DB
3. Call RAG: `POST {RAG_SERVICE_URL}/retrieve` with user_prompt
4. Call LLM: `POST {LLM_SERVICE_URL}/generate` with prompt + context from RAG
5. Save assistant response + sources to DB
6. Return response
7. Error cases: graceful fallback (mock answer if services down)

---

### 3. Get Chat History

**Endpoint:** `GET /history/{conversation_id}`

**Response (200):**
```json
{
  "id": 123,
  "title": "Legal Consultation",
  "created_at": "2026-09-08T10:30:00",
  "updated_at": "2026-09-08T10:35:00",
  "messages": [
    {
      "id": 1,
      "role": "user",
      "content": "What are marriage requirements?",
      "sources": [],
      "created_at": "2026-09-08T10:30:00"
    },
    {
      "id": 2,
      "role": "assistant",
      "content": "According to Vietnamese law...",
      "sources": ["Article 10, Law on Marriage 2020"],
      "created_at": "2026-09-08T10:30:30"
    }
  ]
}
```

---

### 4. List User Conversations

**Endpoint:** `GET /conversations/{user_id}`

**Response (200):**
```json
[
  {
    "id": 123,
    "title": "Legal Consultation",
    "created_at": "2026-09-08T10:30:00",
    "message_count": 5
  }
]
```

---

## Internal Service Contracts (What You Call)

### RAG Service Call

**You send:**
```json
{
  "query": "user's prompt from /chat endpoint",
  "top_k": 5
}
```

**You receive:**
```json
{
  "documents": ["Article 10, Law on Marriage (2020): ..."],
  "scores": [0.95]
}
```

**Timeout:** 30 seconds (hardcoded in your code)  
**Error handling:** If RAG times out or errors, pass empty documents array to LLM

---

### LLM Service Call

**You send:**
```json
{
  "prompt": "user's prompt from /chat endpoint",
  "context": [
    "Article 10, Law on Marriage (2020): ...",
    "Article 11, Law on Marriage (2020): ..."
  ]
}
```

**You receive:**
```json
{
  "answer": "According to Vietnamese law...",
  "confidence": 0.92,
  "sources": ["Article 10, Law on Marriage 2020"]
}
```

**Timeout:** 60 seconds (hardcoded in your code)  
**Error handling:** If LLM times out, return error message: "LLM service unavailable. Please try again shortly."

---

## Database Schema

**File:** `legal_llm.db` (SQLite)

### Users Table
```sql
CREATE TABLE users (
  id INTEGER PRIMARY KEY,
  user_id STRING UNIQUE NOT NULL,
  created_at DATETIME
);
```

### Conversations Table
```sql
CREATE TABLE conversations (
  id INTEGER PRIMARY KEY,
  user_id INTEGER FOREIGN KEY,
  title STRING,
  created_at DATETIME,
  updated_at DATETIME
);
```

### Messages Table
```sql
CREATE TABLE messages (
  id INTEGER PRIMARY KEY,
  conversation_id INTEGER FOREIGN KEY,
  role STRING ('user' OR 'assistant'),
  content TEXT,
  sources TEXT (JSON string),
  created_at DATETIME
);
```

---

## Environment Variables (.env)

**Required:**
```
LLM_SERVICE_URL=http://localhost:8001
RAG_SERVICE_URL=http://localhost:8002
```

**Optional:**
```
LOG_LEVEL=INFO
```

---

## Performance SLAs You Must Meet

| Endpoint | SLA | Notes |
|----------|-----|-------|
| `/start-conversation` | ≤100ms | DB write only |
| `/chat/{conversation_id}` | ≤60s | RAG (30s) + LLM (60s) |
| `/history/{conversation_id}` | ≤500ms | DB read |
| `/conversations/{user_id}` | ≤200ms | DB read |

**Chat endpoint breakdown:**
- RAG call: ≤30s
- LLM call: ≤60s
- Total orchestration: ≤60s (LLM is critical path)

---

## Error Responses You Must Return

### Invalid Input (400)
```json
{"detail": "Prompt cannot be empty"}
```

### Not Found (404)
```json
{"detail": "Conversation not found"}
```

### Server Error (500)
```json
{"detail": "Internal server error: [reason]"}
```

### Service Down (200, graceful degradation)
If RAG times out:
```json
{
  "answer": "Service temporarily unavailable. Please try again.",
  "status": "success",
  "sources": []
}
```

---

## Implementation Checklist

- ✅ Database initialization on startup
- ✅ User session management (create or get user)
- ✅ Conversation CRUD (create, read, list)
- ✅ Message persistence (save user + assistant)
- ✅ RAG service integration (call /retrieve, handle timeouts)
- ✅ LLM service integration (call /generate, handle timeouts)
- ✅ Response validation (answer ≤5000 chars, sources array)
- ✅ Error handling (graceful fallback if services down)
- ✅ Input validation (prompt length, conversation exists)
- ✅ Async operations (don't block on service calls)

---

## Coordination Points

**With RAG Engineer:**
- Expect documents in format: `"Article X, Law on Y (YEAR): [text]"`
- Pass these unchanged to LLM service
- Scores array same length as documents array

**With LLM Engineer:**
- Send exact documents from RAG (don't parse/reformat)
- Receive sources in format: `"Article X, Law on Y YEAR"`
- Pass sources directly to user response
- Answer must be ≤5000 chars (truncate if needed)

---

## Deployment

**Local:** `fastapi dev main.py`  
**Docker:** `docker build . && docker run -p 8000:8000 ...`  
**Volume:** Mount `legal_llm.db` to persist data

---

## You're Done When

- All 4 endpoints working
- Database persists data across restarts
- RAG/LLM services can be swapped without code changes (via .env)
- Timeouts handled gracefully
- All responses match exact JSON schema

---

## No Further Changes Allowed

This is your final spec. Do not add new endpoints or change response formats without team approval.