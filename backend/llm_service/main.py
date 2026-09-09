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
    prompt: str = Field(..., description="The user prompt")
    context: list[str] = Field(default=[], description="Retrieved documents or routing instructions")

class LLMServiceResponse(BaseModel):
    answer: str = Field(..., description="Generated response")
    confidence: float = Field(..., description="Model confidence score (0-1)")
    sources: list[str] = Field(default=[], description="Which documents were referenced")

def clean_source_citation(doc_text: str) -> str:
    header = doc_text.split(":", 1)[0]
    return re.sub(r"\((\d{4})\)", r"\1", header).strip()

@app.post("/generate", response_model=LLMServiceResponse)
async def generate(payload: LLMServiceRequest):
    # 1. Routing / Chit-Chat Mode (Intercepts single-string instructions from backend)
    if len(payload.context) == 1 and ("Phân loại tin nhắn" in payload.context[0] or "Bạn là trợ lý tư vấn" in payload.context[0]):
        system_prompt = payload.context[0]
        prompt_template = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{payload.prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        output = llm(prompt_template, max_tokens=256, temperature=0.01, repeat_penalty=1.1, stop=["<|im_end|>"])
        return LLMServiceResponse(answer=output["choices"][0]["text"].strip(), confidence=0.0, sources=[])

    # 2. Standard Legal RAG Mode
    if not payload.context:
        return LLMServiceResponse(
            answer="Không đủ thông tin pháp lý từ tài liệu được cung cấp.",
            confidence=0.0,
            sources=[]
        )

    context_str = "\n".join([f"- {doc}" for doc in payload.context])

    system_prompt = (
        "Bạn là một trợ lý pháp lý chuyên nghiệp. Tất cả câu trả lời BẮT BUỘC phải viết bằng TIẾNG VIỆT.\n\n"
        "Yêu cầu trả lời:\n"
        "1. Trực tiếp đưa ra câu trả lời ngắn gọn.\n"
        "2. Chỉ trích dẫn thông tin có trong phần \"Tài liệu\".\n"
        "3. Nếu \"Tài liệu\" không có thông tin, CHỈ trả lời đúng câu: \"Không đủ thông tin pháp lý từ tài liệu được cung cấp.\""
    )

    prompt_template = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"Tài liệu:\n{context_str}\n\n"
        f"Câu hỏi: {payload.prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    output = llm(
        prompt_template,
        max_tokens=300,
        temperature=0.01,
        repeat_penalty=1.1,
        stop=["<|im_end|>"]
    )

    raw_answer = output["choices"][0]["text"].strip()

    if "Không đủ thông tin pháp lý" in raw_answer:
        return LLMServiceResponse(answer=raw_answer[:5000], confidence=0.2, sources=[])

    return LLMServiceResponse(
        answer=raw_answer[:5000],
        confidence=0.90,
        sources=[clean_source_citation(doc) for doc in payload.context]
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
