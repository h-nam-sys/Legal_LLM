from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from database import User, Conversation, Message, AuditLog
import json
from datetime import datetime

# ============ User Operations ============

async def get_or_create_user(db: AsyncSession, user_id: str) -> User:
    result = await db.execute(select(User).where(User.user_id == user_id))
    user = result.scalar_one_or_none()

    if not user:
        user = User(user_id=user_id)
        db.add(user)
        await db.commit()
        await db.refresh(user)
    return user

# ============ Conversation Operations ============

async def create_conversation(db: AsyncSession, user_id: str, title: str = "Legal Consultation") -> Conversation:
    user = await get_or_create_user(db, user_id)
    conversation = Conversation(user_id=user.id, title=title)
    db.add(conversation)
    await db.commit()
    await db.refresh(conversation)
    return conversation

async def get_conversation(db: AsyncSession, conversation_id: int) -> Conversation:
    result = await db.execute(select(Conversation).where(Conversation.id == conversation_id))
    return result.scalar_one_or_none()

async def get_user_conversations(db: AsyncSession, user_id: str) -> list[Conversation]:
    result = await db.execute(select(User).where(User.user_id == user_id))
    user = result.scalar_one_or_none()

    if not user:
        return []

    result = await db.execute(
        select(Conversation)
        .where(Conversation.user_id == user.id)
        .order_by(Conversation.created_at.desc())
    )
    return list(result.scalars().all())

async def update_conversation_title(db: AsyncSession, conversation_id: int, new_title: str):
    conv = await get_conversation(db, conversation_id)
    if conv:
        conv.title = new_title
        await db.commit()

async def update_search_permission(db: AsyncSession, conversation_id: int, permission: bool):
    conv = await get_conversation(db, conversation_id)
    if conv:
        conv.needs_search_permission = permission
        await db.commit()

async def update_form_offer_flag(db: AsyncSession, conversation_id: int, state: bool):
    conv = await get_conversation(db, conversation_id)
    if conv:
        conv.offered_form_download = state
        await db.commit()

# ============ Message Operations ============

async def add_message(db: AsyncSession, conversation_id: int, role: str, content: str, sources: list[str] = None) -> Message:
    message = Message(
        conversation_id=conversation_id,
        role=role,
        content=content,
        sources=json.dumps(sources or [])
    )
    db.add(message)
    await db.commit()
    await db.refresh(message)
    return message

async def get_conversation_history(db: AsyncSession, conversation_id: int) -> list[Message]:
    result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc())
    )
    return list(result.scalars().all())

async def get_last_n_messages(db: AsyncSession, conversation_id: int, n: int = 10) -> list[Message]:
    result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.desc())
        .limit(n)
    )
    messages = list(result.scalars().all())
    return messages[::-1]

# ============ Audit Log Operations ============

async def log_interaction(db: AsyncSession, conversation_id: int, user_query: str, llm_response: str, local_context: str = None, used_online_search: bool = False):
    audit_entry = AuditLog(
        conversation_id=conversation_id,
        user_query=user_query,
        retrieved_local_context=local_context,
        llm_final_response=llm_response,
        used_online_search=used_online_search
    )
    db.add(audit_entry)
    await db.commit()

# ============ Feedback Operations ============

async def update_message_rating(db: AsyncSession, message_id: int, rating: int):
    result = await db.execute(select(Message).where(Message.id == message_id))
    msg = result.scalar_one_or_none()

    if not msg or not hasattr(msg, 'rating'):
        return

    msg.rating = rating
    await db.commit()

    if msg.role == "assistant":
        # Find the preceding user message
        result = await db.execute(
            select(Message)
            .where(
                Message.conversation_id == msg.conversation_id,
                Message.id < msg.id,
                Message.role == "user"
            )
            .order_by(Message.id.desc())
            .limit(1)
        )
        user_msg = result.scalar_one_or_none()

        if user_msg:
            raw_sources = json.loads(msg.sources) if msg.sources else []
            clean_sources = [s for s in raw_sources if not s.startswith("RAG_SCORE:")]

            entry = {
                "timestamp": datetime.now().isoformat(),
                "rating": rating,
                "instruction": "Bạn là cán bộ hướng dẫn thủ tục hành chính công. Trả lời ngắn gọn, lịch sự, đúng trọng tâm và báo rõ giấy tờ còn thiếu.",
                "input": user_msg.content,
                "output": msg.content,
                "rag_context": clean_sources
            }

            with open("all_feedback_dataset.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            if rating == 1:
                with open("sft_positive_dataset.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            elif rating == 0:
                with open("review_negative_dataset.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
