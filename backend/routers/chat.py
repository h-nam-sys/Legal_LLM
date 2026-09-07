from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from schemas import ChatRequest, ChatResponse, StartConversationRequest, ConversationSchema
from services.llm_service import get_legal_response
from services.db_service import (
    create_conversation,
    get_conversation,
    get_conversation_history,
    add_message,
    get_or_create_user,
    get_user_conversations
)
from database import get_db
import json

router = APIRouter(tags=["Chat"])


@router.post("/start-conversation", response_model=dict)
async def start_conversation(
    request: StartConversationRequest,
    db: Session = Depends(get_db)
):
    """Start a new conversation"""
    try:
        conversation = create_conversation(db, request.user_id, request.title)
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
    db: Session = Depends(get_db)
):
    """Handle a chat message in an existing conversation"""

    # Validate input
    if not payload.user_prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    if len(payload.user_prompt) > 2000:
        raise HTTPException(status_code=400, detail="Prompt exceeds 2000 characters")

    # Get conversation
    conversation = get_conversation(db, conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    try:
        # Save user message to DB
        add_message(db, conversation_id, "user", payload.user_prompt)

        # Get last 5 messages for context
        context_messages = get_conversation_history(db, conversation_id)
        context_text = "\n".join([
            f"{msg.role.upper()}: {msg.content}"
            for msg in context_messages[-5:]
        ])

        # Get LLM response
        ai_answer, sources = await get_legal_response(payload.user_prompt)

        # Save assistant response to DB
        add_message(db, conversation_id, "assistant", ai_answer, sources)

        return ChatResponse(
            answer=ai_answer,
            status="success",
            sources=sources
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.get("/conversations/{user_id}", response_model=list[dict])
async def get_user_chats(
    user_id: str,
    db: Session = Depends(get_db)
):
    """Get all conversations for a user"""
    try:
        conversations = get_user_conversations(db, user_id)
        return [
            {
                "id": conv.id,
                "title": conv.title,
                "created_at": conv.created_at.isoformat(),
                "message_count": len(conv.messages)
            }
            for conv in conversations
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching conversations: {str(e)}")


@router.get("/history/{conversation_id}", response_model=ConversationSchema)
async def get_chat_history(
    conversation_id: int,
    db: Session = Depends(get_db)
):
    """Get full chat history for a conversation"""
    try:
        conversation = get_conversation(db, conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        messages = get_conversation_history(db, conversation_id)
        return ConversationSchema(
            id=conversation.id,
            title=conversation.title,
            created_at=conversation.created_at.isoformat(),
            updated_at=conversation.updated_at.isoformat(),
            messages=[
                {
                    "id": msg.id,
                    "role": msg.role,
                    "content": msg.content,
                    "sources": json.loads(msg.sources) if msg.sources else [],
                    "created_at": msg.created_at.isoformat()
                }
                for msg in messages
            ]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching history: {str(e)}")
