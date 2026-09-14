from datetime import datetime
from sqlalchemy import create_engine, event
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime
from sqlalchemy.sql import func

SQLALCHEMY_DATABASE_URL = "sqlite:///./legal_llm.db"

# 1. Add a timeout to prevent instant failures if a lock does happen
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={
        "check_same_thread": False,
        "timeout": 15  # Wait 15 seconds instead of crashing instantly
    }
)

# 2. Force SQLite into WAL mode every time it connects
@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA busy_timeout=5000")
    cursor.close()

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
    active_procedure = Column(String, nullable=True) # Tracks the detected RAG procedure
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
    rating = Column(Integer, nullable=True) # Stores user feedback (1=Không hài lòng, 3=Hài lòng)
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
