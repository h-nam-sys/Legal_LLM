from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import os

# SQLite database file (creates in backend folder)
DATABASE_URL = "sqlite:///./legal_llm.db"

engine = create_engine(
    DATABASE_URL, 
    connect_args={"check_same_thread": False},
    echo=False  # Set to True to see SQL queries
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ============ Models ============

class User(Base):
    """Represents a user session"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, unique=True, index=True)  # Unique identifier (can be anonymous)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    conversations = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")


class Conversation(Base):
    """Represents a chat session/conversation"""
    __tablename__ = "conversations"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    title = Column(String, default="Legal Consultation")  # Topic of conversation
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship
    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")


class Message(Base):
    """Represents a single message in a conversation"""
    __tablename__ = "messages"
    
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), index=True)
    role = Column(String, index=True)  # "user" or "assistant"
    content = Column(Text)  # The actual message
    sources = Column(Text, default="[]")  # JSON string of document sources used
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    conversation = relationship("Conversation", back_populates="messages")


# ============ Create Tables ============

def init_db():
    """Create all tables if they don't exist"""
    Base.metadata.create_all(bind=engine)


# ============ Dependency for FastAPI ============

def get_db():
    """FastAPI dependency to inject DB session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
