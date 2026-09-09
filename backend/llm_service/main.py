import os
import re
from fastapi import FastAPI
from pydantic import BaseModel, Field
from llama_cpp import Llama

app = FastAPI(title="Legal LLM Service (Port 8001)")

MODEL_PATH = os.getenv("MODEL_PATH", r"models\qwen_legal_q4_k_m.gguf")

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=4,
    verbose=False
)

class LLMServiceRequest(BaseModel):
    prompt: str = Field(..., description="The user prompt")
    context: list[str] = Field(default=[], description="Retrieved documents")

class LLMServiceResponse(BaseModel):
    answer: str = Field(..., description="Generated response")
    confidence: float = Field(..., description="Model confidence score (0-1)")
    sources: list[str] = Field(default=[], description="Which documents were referenced")

def clean_source_citation(doc_text: str) -> str:
    """Extracts the actual procedure name from the document chunk instead of the label"""
    for line in doc_text.split("\n"):
        if "Tên thủ tục hành chính" in line:
            parts = line.split(":", 1)
            if len(parts) > 1:
                return parts[1].strip()

    # Fallback if line isn't found
    header = doc_text.split("\n")[0]
    return header.replace("Tên thủ tục hành chính:", "").strip()

@app.post("/generate", response_model=LLMServiceResponse)
async def generate(payload: LLMServiceRequest):
    if not payload.context:
        return LLMServiceResponse(
            answer="Không tìm thấy tài liệu liên quan.",
            confidence=0.0,
            sources=[]
        )

    # Join the retrieved RAG chunks
    context_str = "\n\n".join([f"Tài liệu {i+1}:\n{doc}" for i, doc in enumerate(payload.context)])

    system_prompt = (
        "Bạn là trợ lý tư vấn thủ tục hành chính và pháp luật Việt Nam. "
        "Dựa vào thông tin tham khảo được cung cấp bởi hệ thống dưới đây, hãy trả lời câu hỏi của người dùng một cách chính xác, tự nhiên bằng tiếng Việt."
    )

    prompt_template = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"[Thông tin tham khảo từ hệ thống]:\n{context_str}\n\n"
        f"Câu hỏi: {payload.prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    output = llm(
        prompt_template,
        max_tokens=300,
        temperature=0.1,
        repeat_penalty=1.1,
        stop=["<|im_end|>"]
    )

    raw_answer = output["choices"][0]["text"].strip()

    return LLMServiceResponse(
        answer=raw_answer,
        confidence=0.90,
        sources=[clean_source_citation(doc) for doc in payload.context]
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
