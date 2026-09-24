from pydantic import BaseModel, Field
from typing import Optional

# ============ User-facing API ============

class LoginRequest(BaseModel):
    username: str = Field(..., description="The user's username for deterministic ID generation")

class ChatRequest(BaseModel):
    user_prompt: str = Field(..., description="The legal question from the user")

class ChatResponse(BaseModel):
    answer: str = Field(..., description="The model's legal advice response")
    status: str = Field("success", description="Status of the request")
    sources: list[str] = Field(default=[], description="Legal documents used for the answer")
    message_id: int | None = None  # Add this line

# ============ RAG Service Contract ============

class RAGServiceRequest(BaseModel):
    query: str = Field(..., description="Legal query to search for")
    top_k: int = Field(default=5, description="Number of top results to return")

class RAGServiceResponse(BaseModel):
    documents: list[str] = Field(..., description="Retrieved legal document excerpts")
    scores: list[float] = Field(..., description="Relevance scores for each document")

# ============ LLM Service Contract ============

class LLMServiceRequest(BaseModel):
    system_prompt: str = Field(..., description="The system instructions (formatting, constraints)")
    user_prompt: str = Field(..., description="The user's question and RAG context")
    context: list[str] = Field(default=[], description="Retrieved legal documents as context")
    max_tokens: int = Field(default=300, description="Max tokens to generate")
    temperature: float = Field(default=0.0, description="Generation temperature")

class LLMServiceResponse(BaseModel):
    answer: str = Field(..., description="Generated legal response")
    confidence: float = Field(..., description="Model confidence score (0-1)")
    sources: list[str] = Field(default=[], description="Which documents were referenced")

# ============ Database/Conversation Schemas ============

class MessageSchema(BaseModel):
    """Represents a single message in chat history"""
    id: int
    role: str
    content: str
    sources: list[str] = []
    created_at: str

class ConversationSchema(BaseModel):
    """Represents a conversation with full history and stateful tracking"""
    id: int
    title: str
    location: Optional[str] = None
    current_procedure: Optional[str] = None
    checklist: dict = Field(default_factory=dict)
    needs_search_permission: bool = False
    created_at: str
    updated_at: str
    messages: list[MessageSchema] = []

class StartConversationRequest(BaseModel):
    """Request to start a new conversation"""
    title: str = Field(default="Legal Consultation", description="Conversation title")

# ============ Audit Log Schemas ============

class AuditLogSchema(BaseModel):
    """Represents an admin audit log entry"""
    id: int
    conversation_id: int
    user_query: str
    retrieved_local_context: Optional[str] = None
    llm_final_response: str
    used_online_search: bool
    created_at: str
