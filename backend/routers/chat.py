from fastapi import APIRouter, HTTPException, Depends, Header, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import func
from sqlalchemy.future import select
from pydantic import BaseModel
from schemas import ChatRequest, ChatResponse, StartConversationRequest, ConversationSchema
from services.llm_service import get_legal_response_stream
from mcp_service import execute_mcp_search, should_trigger_mcp
from rate_limiter import limiter
from services.db_service import (
    create_conversation,
    get_conversation,
    get_conversation_history,
    get_user_conversations,
    add_message,
    update_conversation_title,
    update_message_rating,
    log_interaction,
    Conversation
)
from database import get_db
import json
import random
import traceback
import unicodedata
import re
import difflib
import ast

router = APIRouter(tags=["Chat"])

class FeedbackRequest(BaseModel):
    rating: int

def get_current_user(x_user_id: str = Header(default=None)):
    if not x_user_id:
        raise HTTPException(status_code=401, detail="User ID missing. Please log in.")
    return x_user_id

# ==========================================
# ROBUST PARSER (Unwraps the Double-Encode Bug)
# ==========================================
def parse_sources(sources_data):
    if not sources_data:
        return []
    if isinstance(sources_data, list):
        return sources_data
    if isinstance(sources_data, str):
        try:
            parsed = json.loads(sources_data)
            # Fix for historical double-encoded strings from previous bug
            if isinstance(parsed, str):
                parsed = json.loads(parsed)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            try:
                parsed = ast.literal_eval(sources_data)
                return parsed if isinstance(parsed, list) else []
            except Exception:
                return []
    return []

# ==========================================
# GLOBAL FUZZY MATCHING HELPERS
# ==========================================
def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode('utf-8')
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    return re.sub(r'\s+', ' ', text).strip()

def fuzzy_match_any(query: str, normalized_vocab_list: list[str], threshold: float = 0.85) -> bool:
    norm_query = normalize_text(query)
    for vocab in normalized_vocab_list:
        if vocab in norm_query:
            return True
    query_words = norm_query.split()
    for vocab in normalized_vocab_list:
        vocab_word_count = len(vocab.split())
        if vocab_word_count == 0:
            continue
        for i in range(len(query_words) - vocab_word_count + 1):
            ngram = " ".join(query_words[i:i + vocab_word_count])
            similarity = difflib.SequenceMatcher(None, vocab, ngram).ratio()
            if similarity >= threshold:
                return True
    return False

RAW_FORM_TRIGGERS = ["giấy tờ", "làm giấy", "cần gì", "hồ sơ", "biểu mẫu", "tờ khai"]
NORMALIZED_FORM_TRIGGERS = [normalize_text(w) for w in RAW_FORM_TRIGGERS if normalize_text(w)]

def is_form_trigger(query: str) -> bool:
    return fuzzy_match_any(query, NORMALIZED_FORM_TRIGGERS)

# ==========================================
# ENDPOINTS
# ==========================================

@router.post("/start-conversation", response_model=dict)
async def start_conversation(
    request: StartConversationRequest,
    db: AsyncSession = Depends(get_db),
    current_user_id: str = Depends(get_current_user)
):
    try:
        conversation = await create_conversation(db, current_user_id, request.title)
        return {
            "conversation_id": conversation.id,
            "title": conversation.title,
            "created_at": conversation.created_at.isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create conversation: {str(e)}")


@router.post("/chat/{conversation_id}")
@limiter.limit("10/minute")
async def handle_chat(
    request: Request,
    conversation_id: int,
    payload: ChatRequest,
    db: AsyncSession = Depends(get_db),
    current_user_id: str = Depends(get_current_user)
):
    if not payload.user_prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    conversation = await get_conversation(db, conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    await add_message(db, conversation_id, "user", payload.user_prompt)
    context_messages = await get_conversation_history(db, conversation_id)

    if len(context_messages) == 1:
        raw_title = payload.user_prompt.strip()
        new_title = raw_title.capitalize()
        if len(new_title) > 35:
            new_title = new_title[:35] + "..."
        await update_conversation_title(db, conversation_id, new_title)

    last_proc_title = ""
    if len(context_messages) >= 2:
        for msg in reversed(context_messages[:-1]):
            if msg.role == "assistant" and msg.sources:
                sources_list = parse_sources(msg.sources)
                actual_sources = [s for s in sources_list if isinstance(s, str) and not s.startswith("RAG_SCORE")]
                if actual_sources:
                    last_proc_title = actual_sources[0]
                    break

    # =========================================================================
    # INTENT CHECK & STATELESS ROUTING
    # =========================================================================
    TOPIC_SWITCH_PATTERN = r"(đăng ký|dang ky|làm mới|lam moi|thủ tục|thu tuc|xin cấp|xin cap|cấp lại|cap lai|kết hôn|ket hon|khai sinh|khai tử|khai tu|hộ tịch|ho tich|đổi|doi|chuyển|chuyen)"

    user_prompt_clean = payload.user_prompt.strip().lower()
    is_new_topic = bool(re.search(TOPIC_SWITCH_PATTERN, user_prompt_clean, re.IGNORECASE))

    if is_new_topic or not last_proc_title:
        rag_search_query = payload.user_prompt.strip()
    else:
        # EXACT concatenation just like your manual stateless test
        rag_search_query = f"{last_proc_title} {payload.user_prompt}".strip()

    # Pass the concatenated query to both Qdrant and LLM for true statelessness
    stream_gen, max_score = await get_legal_response_stream(rag_search_query, rag_search_query)

    async def stream_and_save_to_db():
        SCORE_THRESHOLD = 0.55

        if max_score < SCORE_THRESHOLD:
            if should_trigger_mcp(payload.user_prompt):
                ask_msg = "Tôi không tìm thấy thông tin trong dữ liệu địa phương. Bạn có muốn tôi tìm kiếm trực tuyến trên Cổng Dịch vụ công Quốc gia không?"
                # FIX: Pass empty list, not json.dumps
                await add_message(db, conversation_id, "assistant", ask_msg, [])

                yield f"data: {json.dumps({'type': 'metadata', 'sources': []})}\n\n"
                yield f"data: {json.dumps({'type': 'chunk', 'text': ask_msg})}\n\n"
                yield "data: [DONE]\n\n"
                return
            else:
                nudge_msg = random.choice([
                    "Xin chào! Tôi là trợ lý hướng dẫn thủ tục hành chính. Vui lòng cho tôi biết bạn cần làm thủ tục gì nhé.",
                    "Chào bạn! Tôi ở đây để hỗ trợ giải đáp các thủ tục hành chính công. Bạn đang cần thực hiện thủ tục nào?"
                ])
                # FIX: Pass empty list, not json.dumps
                await add_message(db, conversation_id, "assistant", nudge_msg, [])

                yield f"data: {json.dumps({'type': 'metadata', 'sources': []})}\n\n"
                yield f"data: {json.dumps({'type': 'chunk', 'text': nudge_msg})}\n\n"
                yield "data: [DONE]\n\n"
                return

        full_ai_answer = ""
        final_sources = []

        async for raw_chunk in stream_gen:
            yield raw_chunk
            if raw_chunk.startswith("data: "):
                data_str = raw_chunk.replace("data: ", "").strip()
                if data_str and data_str != "[DONE]":
                    try:
                        parsed = json.loads(data_str)
                        if parsed.get("type") == "metadata":
                            final_sources = parsed.get("sources", [])
                        elif parsed.get("type") == "chunk":
                            full_ai_answer += parsed.get("text", "")
                    except json.JSONDecodeError:
                        pass

        db_sources = final_sources.copy()
        db_sources.append(f"RAG_SCORE:{max_score}")

        if is_form_trigger(payload.user_prompt):
            offer_text = "\n\n**Bạn có muốn tải biểu mẫu (file Word) cho thủ tục này không?**"
            full_ai_answer += offer_text
            yield f"data: {json.dumps({'type': 'chunk', 'text': offer_text})}\n\n"

        # FIX: Pass db_sources directly. DO NOT json.dumps it!
        await add_message(db, conversation_id, "assistant", full_ai_answer, db_sources)
        await log_interaction(db, conversation_id, payload.user_prompt, full_ai_answer, str(final_sources), False)

    return StreamingResponse(stream_and_save_to_db(), media_type="text/event-stream")


@router.post("/chat/{conversation_id}/action/mcp")
async def action_mcp(
    conversation_id: int,
    db: AsyncSession = Depends(get_db),
    current_user_id: str = Depends(get_current_user)
):
    conversation = await get_conversation(db, conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    context_messages = await get_conversation_history(db, conversation_id)

    last_proc_title = ""
    last_user_msg = ""
    for msg in reversed(context_messages):
        if msg.role == "user" and not last_user_msg:
            last_user_msg = msg.content
        if msg.role == "assistant" and msg.sources and not last_proc_title:
            sources_list = parse_sources(msg.sources)
            actual_sources = [s for s in sources_list if isinstance(s, str) and not s.startswith("RAG_SCORE")]
            if actual_sources:
                last_proc_title = actual_sources[0]

    query = f"{last_proc_title} {last_user_msg}".strip() if last_proc_title else last_user_msg

    mcp_answer, mcp_sources = await execute_mcp_search(query, conversation.location)
    # FIX: Pass mcp_sources directly
    await add_message(db, conversation_id, "assistant", mcp_answer, mcp_sources)
    await log_interaction(db, conversation_id, query, mcp_answer, str(mcp_sources), True)

    return {"text": mcp_answer, "sources": mcp_sources}


@router.post("/chat/{conversation_id}/action/form")
async def action_form(
    conversation_id: int,
    db: AsyncSession = Depends(get_db),
    current_user_id: str = Depends(get_current_user)
):
    conversation = await get_conversation(db, conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    context_messages = await get_conversation_history(db, conversation_id)

    last_proc_title = ""
    last_user_msg = ""

    for msg in reversed(context_messages):
        if msg.role == "user" and not last_user_msg:
            last_user_msg = msg.content
        if msg.role == "assistant" and msg.sources and not last_proc_title:
            sources_list = parse_sources(msg.sources)
            actual_sources = [s for s in sources_list if isinstance(s, str) and not s.startswith("RAG_SCORE")]
            if actual_sources:
                last_proc_title = actual_sources[0]
                break

    try:
        from services.forms_matcher import find_matching_form
        query = f"{last_proc_title} {last_user_msg}".strip() if last_proc_title else last_user_msg
        matched_form = find_matching_form(procedure_title=last_proc_title, query=query)
        if matched_form:
            link_msg = (
                f"Đây là biểu mẫu bạn cần: \n\n"
                f"📄 **[{matched_form['display_name']} (.docx)](/static/forms/{matched_form['filename']})**"
            )
        else:
            link_msg = "Xin lỗi, hiện tại tôi chưa có sẵn file Word cho biểu mẫu này."
    except ImportError:
        link_msg = "Xin lỗi, module biểu mẫu chưa được cấu hình."

    # FIX: Pass empty list, not json.dumps
    await add_message(db, conversation_id, "assistant", link_msg, [])

    return {"text": link_msg, "sources": []}


@router.get("/conversations", response_model=list[dict])
async def get_user_chats(
    db: AsyncSession = Depends(get_db),
    current_user_id: str = Depends(get_current_user)
):
    try:
        conversations = await get_user_conversations(db, current_user_id)
        return [
            {
                "id": conv.id,
                "title": conv.title,
                "created_at": conv.created_at.isoformat() if conv.created_at else "",
                "message_count": 0
            }
            for conv in conversations
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching conversations: {str(e)}")


@router.get("/history/{conversation_id}", response_model=ConversationSchema)
async def get_chat_history(
    conversation_id: int,
    db: AsyncSession = Depends(get_db),
    current_user_id: str = Depends(get_current_user)
):
    try:
        conversation = await get_conversation(db, conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        messages = await get_conversation_history(db, conversation_id)

        return ConversationSchema(**{
            "id": conversation.id,
            "title": conversation.title,
            "location": conversation.location,
            "current_procedure": conversation.current_procedure,
            "checklist": conversation.checklist or {},
            "needs_search_permission": conversation.needs_search_permission,
            "created_at": conversation.created_at.isoformat() if conversation.created_at else "",
            "updated_at": conversation.updated_at.isoformat() if conversation.updated_at else "",
            "messages": [
                {
                    "id": msg.id,
                    "role": msg.role,
                    "content": msg.content,
                    "sources": parse_sources(msg.sources),
                    "created_at": msg.created_at.isoformat() if msg.created_at else "",
                    "rating": getattr(msg, 'rating', None)
                }
                for msg in messages
            ]
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching history: {str(e)}")


@router.post("/feedback/{message_id}")
async def submit_feedback(
    message_id: int,
    payload: FeedbackRequest,
    db: AsyncSession = Depends(get_db),
    current_user_id: str = Depends(get_current_user)
):
    try:
        await update_message_rating(db, message_id, payload.rating)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
