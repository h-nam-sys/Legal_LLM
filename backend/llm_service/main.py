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

def extract_sources(docs: list[str]) -> list[str]:
    sources = []
    for doc in docs:
        match = re.search(r"Tên thủ tục hành chính:\s*([^\n\r]+)", doc)
        if match:
            proc_name = match.group(1).strip()
            if proc_name:
                sources.append(proc_name)
    return sources

@app.post("/generate", response_model=LLMServiceResponse)
async def generate(payload: LLMServiceRequest):
    if not payload.context:
        return LLMServiceResponse(
            answer="Không tìm thấy tài liệu liên quan.",
            confidence=0.0,
            sources=[]
        )

    context_str = "\n\n".join(payload.context)

    system_prompt = (
        "Bạn là trợ lý giải đáp thủ tục hành chính Việt Nam. "
        "Hãy dựa vào tài liệu được cung cấp để trả lời câu hỏi trực tiếp và ngắn gọn nhất có thể."
    )

    prompt_template = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"{context_str}\n\n"
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
        sources=extract_sources(payload.context)
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
