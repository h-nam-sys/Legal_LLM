import os
import re
from fastapi import FastAPI
from pydantic import BaseModel, Field
from llama_cpp import Llama

app = FastAPI(title="Legal LLM Service (Port 8001)")

MODEL_PATH = os.getenv("MODEL_PATH", r"models\qwen2.5-0.5b-instruct-q4_k_m.gguf")

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=4,
    verbose=False
)

class LLMServiceRequest(BaseModel):
    prompt: str = Field(..., description="The legal question")
    context: list[str] = Field(default=[], description="Retrieved legal documents")

class LLMServiceResponse(BaseModel):
    answer: str = Field(..., description="Generated legal response")
    confidence: float = Field(..., description="Model confidence score (0-1)")
    sources: list[str] = Field(default=[], description="Which documents were referenced")

def clean_source_citation(doc_text: str) -> str:
    header = doc_text.split(":", 1)[0]
    return re.sub(r"\((\d{4})\)", r"\1", header).strip()

@app.post("/generate", response_model=LLMServiceResponse)
async def generate(payload: LLMServiceRequest):
    if not payload.context:
        return LLMServiceResponse(
            answer="Không đủ thông tin pháp lý từ tài liệu được cung cấp.",
            confidence=0.0,
            sources=[]
        )

    context_str = "\n".join([f"- {doc}" for doc in payload.context])

    prompt_template = (
        f"<|im_start|>system\n"
        f"Bạn là trợ lý pháp lý. Trả lời câu hỏi ngắn gọn bằng tiếng Việt, CHỈ sử dụng thông tin trong phần Tài liệu dưới đây.\n"
        f"Không chào hỏi, không thêm thông tin ngoài.\n\n"
        f"Tài liệu:\n{context_str}<|im_end|>\n"
        f"<|im_start|>user\n{payload.prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    output = llm(
        prompt_template,
        max_tokens=350,
        temperature=0.1,
        stop=["<|im_end|>"]
    )

    raw_answer = output["choices"][0]["text"].strip()

    return LLMServiceResponse(
        answer=raw_answer[:5000],
        confidence=0.90,
        sources=[clean_source_citation(doc) for doc in payload.context]
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
