import os
import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from llama_cpp import Llama
from transformers import AutoTokenizer

app = FastAPI(title="Legal LLM Service (Local Fallback - Port 8002)")

MODEL_PATH = os.getenv("MODEL_PATH", r"models/qwen3_1.7b_q8_0.gguf")
TOKENIZER_NAME = os.getenv("TOKENIZER_NAME", "Qwen/Qwen3-0.6B")

print(f"[INFO] Loading Tokenizer ({TOKENIZER_NAME}) for prompt formatting...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

print("[INFO] Loading Llama.cpp engine (CPU fallback mode)...")
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=4096,
    n_threads=8,
    n_gpu_layers=-1,
    verbose=False
)

class OpenAIMessage(BaseModel):
    role: str
    content: str

class OpenAIRequest(BaseModel):
    messages: list[OpenAIMessage]
    stream: bool = True
    max_tokens: int = Field(default=800, description="Max tokens to generate")
    temperature: float = Field(default=0.0, description="Sampling temperature")

@app.post("/v1/chat/completions")
async def chat_completions(payload: OpenAIRequest):
    # Định dạng prompt
    messages_list = [{"role": msg.role, "content": msg.content} for msg in payload.messages]

    formatted_prompt = tokenizer.apply_chat_template(
        messages_list, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )

    def stream_generator():
        # Stream text chunks đúng chuẩn OpenAI schema
        for chunk in llm(formatted_prompt, max_tokens=payload.max_tokens, temperature=payload.temperature, stream=True, stop=["<|im_end|>"]):
            text = chunk["choices"][0]["text"]
            if text:
                response_chunk = {
                    "choices": [{"delta": {"content": text}}]
                }
                yield f"data: {json.dumps(response_chunk)}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(stream_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
