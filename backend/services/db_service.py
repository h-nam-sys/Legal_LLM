from sqlalchemy.orm import Session
from database import User, Conversation, Message
import json
from datetime import datetime

# ============ User Operations ============

def get_or_create_user(db: Session, user_id: str) -> User:
    """Get existing user or create new one based on the deterministic UUID"""
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        user = User(user_id=user_id)
        db.add(user)
        db.commit()
        db.refresh(user)
    return user

# ============ Conversation Operations ============

def create_conversation(db: Session, user_id: str, title: str = "Legal Consultation") -> Conversation:
    """Create a new conversation for a user"""
    user = get_or_create_user(db, user_id)
    conversation = Conversation(user_id=user.id, title=title)
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    return conversation

def get_conversation(db: Session, conversation_id: int) -> Conversation:
    """Get conversation by ID"""
    return db.query(Conversation).filter(Conversation.id == conversation_id).first()

def get_user_conversations(db: Session, user_id: str) -> list[Conversation]:
    """Get all conversations for a user"""
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        return []
    return db.query(Conversation).filter(Conversation.user_id == user.id).order_by(Conversation.created_at.desc()).all()

def update_conversation_title(db: Session, conversation_id: int, new_title: str):
    """Update the title of an existing conversation"""
    conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if conv:
        conv.title = new_title
        db.commit()

def update_active_procedure(db: Session, conversation_id: int, procedure_name: str):
    """Update the active legal procedure (Requires 'active_procedure' column in database.py)"""
    conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if conv and hasattr(conv, 'active_procedure'):
        conv.active_procedure = procedure_name
        db.commit()

# ============ Message Operations ============

def add_message(
    db: Session,
    conversation_id: int,
    role: str,
    content: str,
    sources: list[str] = None
) -> Message:
    """Add a message to a conversation"""
    message = Message(
        conversation_id=conversation_id,
        role=role,
        content=content,
        sources=json.dumps(sources or [])
    )
    db.add(message)
    db.commit()
    db.refresh(message)
    return message

def get_conversation_history(db: Session, conversation_id: int) -> list[Message]:
    """Get all messages in a conversation"""
    return db.query(Message).filter(Message.conversation_id == conversation_id).order_by(Message.created_at.asc()).all()

def get_last_n_messages(db: Session, conversation_id: int, n: int = 10) -> list[Message]:
    """Get last N messages from a conversation (for context)"""
    return db.query(Message).filter(
        Message.conversation_id == conversation_id
    ).order_by(Message.created_at.desc()).limit(n).all()[::-1]

def update_message_rating(db: Session, message_id: int, rating: int):
    """Update message rating in SQLite (0=Không hài lòng, 1=Hài lòng) and append to fine-tuning datasets."""
    msg = db.query(Message).filter(Message.id == message_id).first()
    if not msg or not hasattr(msg, 'rating'):
        return

    msg.rating = rating
    db.commit()

    if msg.role == "assistant":
        user_msg = db.query(Message).filter(
            Message.conversation_id == msg.conversation_id,
            Message.id < msg.id,
            Message.role == "user"
        ).order_by(Message.id.desc()).first()

        if user_msg:
            # Clean RAG sources by removing the internal score marker
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

            # Master log with all rated interactions
            with open("all_feedback_dataset.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            # Golden dataset for standard Supervised Fine-Tuning (SFT)
            if rating == 1:
                with open("sft_positive_dataset.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            # Defect dataset for error review or DPO negative pairs
            elif rating == 0:
                with open("review_negative_dataset.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
