import os
import re
import json
import asyncio
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from llama_cpp import Llama
from transformers import AutoTokenizer

app = FastAPI(title="Legal LLM Service (Port 8002)")

MODEL_PATH = os.getenv("MODEL_PATH", r"models\qwen3_1.7b_q8_0.gguf")
TOKENIZER_NAME = os.getenv("TOKENIZER_NAME", "Qwen/Qwen3-0.6B")

print(f"[INFO] Loading Tokenizer ({TOKENIZER_NAME}) for prompt formatting...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

print("[INFO] Loading Llama.cpp engine (CPU mode)...")
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=4096,
    n_threads=8,
    n_gpu_layers=-1,
    verbose=False
)

class LLMServiceRequest(BaseModel):
    system_prompt: str | None = Field(default=None, description="System instructions")
    user_prompt: str | None = Field(default=None, description="User prompt + context")
    prompt: str | None = Field(default=None, description="Fallback raw prompt")
    context: list[str] = Field(default=[], description="Retrieved context chunks")
    max_tokens: int = Field(default=300, description="Max tokens to generate")
    temperature: float = Field(default=0.0, description="Sampling temperature")

def extract_sources(docs: list[str]) -> list[str]:
    sources = []
    for doc in docs:
        match = re.search(r"Tên thủ tục hành chính:\s*([^\n\r]+)", doc)
        if match:
            proc_name = match.group(1).strip()
            if proc_name:
                sources.append(proc_name)
    return sources

@app.post("/generate_stream")
async def generate_stream(payload: LLMServiceRequest):
    user_content = payload.user_prompt or payload.prompt or ""

    messages = []
    if payload.system_prompt:
        messages.append({"role": "system", "content": payload.system_prompt})
    messages.append({"role": "user", "content": user_content})

    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )

    def stream_generator():
        # Yield the sources first so the frontend has them instantly
        sources = extract_sources(payload.context)
        yield f"data: {json.dumps({'type': 'metadata', 'sources': sources})}\n\n"

        # Stream the text chunks as they generate
        for chunk in llm(formatted_prompt, max_tokens=payload.max_tokens, temperature=payload.temperature, stream=True, stop=["<|im_end|>"]):
            text = chunk["choices"][0]["text"]
            if text:
                yield f"data: {json.dumps({'type': 'chunk', 'text': text})}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(stream_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
