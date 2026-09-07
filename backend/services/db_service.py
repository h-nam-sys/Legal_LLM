from sqlalchemy.orm import Session
from database import User, Conversation, Message
import json
from datetime import datetime


# ============ User Operations ============

def get_or_create_user(db: Session, user_id: str) -> User:
    """Get existing user or create new one"""
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
    ).order_by(Message.created_at.desc()).limit(n).all()[::-1]  # Reverse to get chronological order
