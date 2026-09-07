from pydantic import BaseModel, Field

# ============ User-facing API ============

class ChatRequest(BaseModel):
    user_prompt: str = Field(..., description="The legal question from the user")

class ChatResponse(BaseModel):
    answer: str = Field(..., description="The model's legal advice response")
    status: str = Field("success", description="Status of the request")
    sources: list[str] = Field(default=[], description="Legal documents used for the answer")

# ============ RAG Service Contract ============

class RAGServiceRequest(BaseModel):
    query: str = Field(..., description="Legal query to search for")
    top_k: int = Field(default=5, description="Number of top results to return")

class RAGServiceResponse(BaseModel):
    documents: list[str] = Field(..., description="Retrieved legal document excerpts")
    scores: list[float] = Field(..., description="Relevance scores for each document")

# ============ LLM Service Contract ============

class LLMServiceRequest(BaseModel):
    prompt: str = Field(..., description="The legal question")
    context: list[str] = Field(default=[], description="Retrieved legal documents as context")

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
    """Represents a conversation with full history"""
    id: int
    title: str
    created_at: str
    updated_at: str
    messages: list[MessageSchema] = []

class StartConversationRequest(BaseModel):
    """Request to start a new conversation"""
    user_id: str = Field(..., description="Unique user identifier")
    title: str = Field(default="Legal Consultation", description="Conversation title")
