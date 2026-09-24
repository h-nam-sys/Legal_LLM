from datetime import datetime
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime, Boolean, JSON, event

# Switched to the aiosqlite driver
SQLALCHEMY_DATABASE_URL = "sqlite+aiosqlite:///./legal_llm.db"

# 1. Create Async Engine
engine = create_async_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"timeout": 15}
)

# 2. Force SQLite into WAL mode (Must be attached to the underlying sync engine)
@event.listens_for(engine.sync_engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA busy_timeout=5000")
    cursor.close()

# 3. Create Async Session Factory
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False
)

Base = declarative_base()

# ============ Models ============

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, unique=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    conversations = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")

class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    title = Column(String, default="Legal Consultation")
    location = Column(String, nullable=True)
    current_procedure = Column(String, nullable=True)
    checklist = Column(JSON, default=dict)
    needs_search_permission = Column(Boolean, default=False)
    offered_form_download = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")

class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), index=True)
    role = Column(String, index=True)
    content = Column(Text)
    sources = Column(Text, default="[]")
    rating = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    conversation = relationship("Conversation", back_populates="messages")

class AuditLog(Base):
    __tablename__ = "audit_logs"
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), index=True)
    user_query = Column(Text)
    retrieved_local_context = Column(Text, nullable=True)
    llm_final_response = Column(Text)
    used_online_search = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

# ============ Database Lifecycle ============

async def init_db():
    """Create all tables asynchronously"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

async def get_db():
    """FastAPI async dependency to inject DB session"""
    async with AsyncSessionLocal() as db:
        try:
            yield db
        finally:
            await db.close()
