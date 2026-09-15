import os
import re
import asyncio
from fastapi import FastAPI
from pydantic import BaseModel, Field
from llama_cpp import Llama
from transformers import AutoTokenizer

app = FastAPI(title="Legal LLM Service (Port 8002)")

# Path to the GGUF model
MODEL_PATH = os.getenv("MODEL_PATH", r"models\qwen_legal_q4_k_m.gguf")
TOKENIZER_NAME = os.getenv("TOKENIZER_NAME", "Qwen/Qwen3-0.6B")

print(f"[INFO] Loading Tokenizer ({TOKENIZER_NAME}) for prompt formatting...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

print("[INFO] Loading Llama.cpp engine (CPU mode)...")
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=4,
    n_gpu_layers=0,  # 0 forces CPU execution
    verbose=False
)

# Robust schema accepting both old and new payload structures
class LLMServiceRequest(BaseModel):
    system_prompt: str | None = Field(default=None, description="System instructions")
    user_prompt: str | None = Field(default=None, description="User prompt + context")
    prompt: str | None = Field(default=None, description="Fallback raw prompt")
    context: list[str] = Field(default=[], description="Retrieved context chunks")
    max_tokens: int = Field(default=300, description="Max tokens to generate")
    temperature: float = Field(default=0.0, description="Sampling temperature")

class LLMServiceResponse(BaseModel):
    answer: str
    confidence: float
    sources: list[str]

def extract_sources(docs: list[str]) -> list[str]:
    sources = []
    for doc in docs:
        match = re.search(r"Tên thủ tục hành chính:\s*([^\n\r]+)", doc)
        if match:
            proc_name = match.group(1).strip()
            if proc_name:
                sources.append(proc_name)
    return sources

def generate_text_sync(system: str | None, user: str, max_tokens: int, temp: float) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    # Apply ChatML formatting with thinking tokens explicitly suppressed
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    response = llm(
        formatted_prompt,
        max_tokens=max_tokens,
        temperature=temp,
        repeat_penalty=1.1,
        stop=["<|im_end|>"]
    )
    return response["choices"][0]["text"].strip()

@app.post("/generate", response_model=LLMServiceResponse)
async def generate(payload: LLMServiceRequest):
    if not payload.context and not payload.user_prompt and not payload.prompt:
        return LLMServiceResponse(
            answer="Không tìm thấy tài liệu liên quan.",
            confidence=0.0,
            sources=[]
        )

    # Determine user content from either user_prompt or fallback prompt
    user_content = payload.user_prompt or payload.prompt or ""

    raw_answer = await asyncio.to_thread(
        generate_text_sync,
        payload.system_prompt,
        user_content,
        payload.max_tokens,
        payload.temperature
    )

    return LLMServiceResponse(
        answer=raw_answer,
        confidence=0.90,
        sources=extract_sources(payload.context)
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
