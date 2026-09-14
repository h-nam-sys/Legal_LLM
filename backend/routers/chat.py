from fastapi import APIRouter, HTTPException, Depends, Header
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session
from pydantic import BaseModel
from schemas import ChatRequest, ChatResponse, StartConversationRequest, ConversationSchema
from services.llm_service import get_legal_response
from services.db_service import (
    create_conversation,
    get_conversation,
    get_conversation_history,
    add_message,
    get_or_create_user,
    get_user_conversations,
    update_conversation_title,
    update_active_procedure,
    update_message_rating
)
from database import get_db
import json

def detect_procedure(query: str):
    """Local procedure detector so we don't rely on the external RAG server."""
    if not query:
        return None

    query_lower = query.lower().strip()
    procedures = {
        "xác nhận tình trạng hôn nhân": "Thủ tục xác nhận tình trạng hôn nhân",
        "tình trạng hôn nhân": "Thủ tục xác nhận tình trạng hôn nhân",
        "đăng ký khai sinh": "Thủ tục đăng ký khai sinh",
        "khai sinh": "Thủ tục đăng ký khai sinh",
        "đăng ký khai tử": "Thủ tục đăng ký khai tử",
        "khai tử": "Thủ tục đăng ký khai tử",
        "đăng ký kết hôn": "Thủ tục đăng ký kết hôn",
        "kết hôn": "Thủ tục đăng ký kết hôn",
    }

    # Sort by longest string first to prevent partial matches
    sorted_procedures = sorted(procedures.items(), key=lambda item: len(item[0]), reverse=True)

    for keyword, procedure_name in sorted_procedures:
        if keyword in query_lower:
            return procedure_name

    return None

router = APIRouter(tags=["Chat"])

class FeedbackRequest(BaseModel):
    rating: int

def get_current_user(x_user_id: str = Header(default=None)):
    if not x_user_id:
        raise HTTPException(status_code=401, detail="User ID missing. Please log in.")
    return x_user_id

@router.post("/start-conversation", response_model=dict)
async def start_conversation(
    request: StartConversationRequest,
    db: Session = Depends(get_db),
    current_user_id: str = Depends(get_current_user)
):
    try:
        conversation = await run_in_threadpool(create_conversation, db, current_user_id, request.title)
        return {
            "conversation_id": conversation.id,
            "title": conversation.title,
            "created_at": conversation.created_at.isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create conversation: {str(e)}")

@router.post("/chat/{conversation_id}", response_model=ChatResponse)
async def handle_chat(
    conversation_id: int,
    payload: ChatRequest,
    db: Session = Depends(get_db),
    current_user_id: str = Depends(get_current_user)
):
    if not payload.user_prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    if len(payload.user_prompt) > 2000:
        raise HTTPException(status_code=400, detail="Prompt exceeds 2000 characters")

    conversation = await run_in_threadpool(get_conversation, db, conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    try:
        await run_in_threadpool(add_message, db, conversation_id, "user", payload.user_prompt)

        context_messages = await run_in_threadpool(get_conversation_history, db, conversation_id)
        is_first_message = len(context_messages) == 1

        user_messages = [msg.content for msg in context_messages if msg.role == "user"]
        full_user_history = " . ".join(user_messages)

        detected_proc = detect_procedure(payload.user_prompt)
        active_proc = getattr(conversation, 'active_procedure', None)

        if detected_proc and detected_proc != active_proc:
            active_proc = detected_proc
            await run_in_threadpool(update_active_procedure, db, conversation_id, active_proc)

        if active_proc:
            rag_search_query = f"{active_proc}: {payload.user_prompt}"
        else:
            rag_search_query = payload.user_prompt

        ai_answer, raw_sources, max_score = await get_legal_response(
            prompt=full_user_history,
            search_query=rag_search_query
        )

        db_sources = raw_sources.copy() if raw_sources else []
        db_sources.append(f"RAG_SCORE:{max_score}")

        bot_msg = await run_in_threadpool(add_message, db, conversation_id, "assistant", ai_answer, db_sources)

        if is_first_message:
            new_title = payload.user_prompt[:35] + ("..." if len(payload.user_prompt) > 35 else "")
            await run_in_threadpool(update_conversation_title, db, conversation_id, new_title)

        return ChatResponse(answer=ai_answer, status="success", sources=db_sources, message_id=bot_msg.id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/conversations", response_model=list[dict])
async def get_user_chats(
    db: Session = Depends(get_db),
    current_user_id: str = Depends(get_current_user)
):
    try:
        conversations = await run_in_threadpool(get_user_conversations, db, current_user_id)
        return [{"id": conv.id, "title": conv.title, "created_at": conv.created_at.isoformat(), "message_count": len(conv.messages)} for conv in conversations]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching conversations: {str(e)}")

@router.get("/history/{conversation_id}", response_model=ConversationSchema)
async def get_chat_history(
    conversation_id: int,
    db: Session = Depends(get_db),
    current_user_id: str = Depends(get_current_user)
):
    try:
        conversation = await run_in_threadpool(get_conversation, db, conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        messages = await run_in_threadpool(get_conversation_history, db, conversation_id)
        return ConversationSchema(
            id=conversation.id,
            title=conversation.title,
            created_at=conversation.created_at.isoformat(),
            updated_at=conversation.updated_at.isoformat(),
            messages=[{"id": msg.id, "role": msg.role, "content": msg.content, "sources": json.loads(msg.sources) if msg.sources else [], "created_at": msg.created_at.isoformat(), "rating": getattr(msg, 'rating', None)} for msg in messages]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching history: {str(e)}")

@router.post("/feedback/{message_id}")
async def submit_feedback(
    message_id: int,
    payload: FeedbackRequest,
    db: Session = Depends(get_db),
    current_user_id: str = Depends(get_current_user)
):
    try:
        await run_in_threadpool(update_message_rating, db, message_id, payload.rating)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
