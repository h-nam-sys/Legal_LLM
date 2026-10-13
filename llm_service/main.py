"""Existing port/startup command; supports both JSON and SSE responses."""
import json
import os
from pathlib import Path
from threading import Lock

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from llama_cpp import Llama
from transformers import AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / "backend/.env")
load_dotenv(ROOT_DIR / ".env")
app = FastAPI(title="Legal LLM Service (Port 8002)")
MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/qwen3_1.7b_q8_0.gguf"))
if not MODEL_PATH.is_absolute():
    MODEL_PATH = ROOT_DIR / MODEL_PATH
# Preserve the existing tokenizer choice; override TOKENIZER_NAME if your GGUF needs another.
TOKENIZER_NAME = os.getenv("TOKENIZER_NAME", "Qwen/Qwen3-0.6B")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
llm = Llama(model_path=str(MODEL_PATH), n_ctx=4096, n_threads=8, n_gpu_layers=-1, verbose=False)
_model_lock = Lock()


class OpenAIMessage(BaseModel):
    role: str
    content: str


class OpenAIRequest(BaseModel):
    messages: list[OpenAIMessage]
    stream: bool = True
    max_tokens: int = Field(default=800, ge=1, le=4096)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, gt=0, le=1)


def format_prompt(payload):
    return tokenizer.apply_chat_template(
        [{"role": message.role, "content": message.content} for message in payload.messages],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


@app.post("/v1/chat/completions")
def chat_completions(payload: OpenAIRequest):
    # Sync endpoint/generator keeps CPU inference off FastAPI's event loop.
    if not payload.stream:
        with _model_lock:
            result = llm(format_prompt(payload), max_tokens=payload.max_tokens,
                         temperature=payload.temperature, top_p=payload.top_p, stream=False, stop=["<|im_end|>"])
        return {"choices": [{"index": 0, "message": {"role": "assistant", "content": result["choices"][0]["text"]},
                              "finish_reason": result["choices"][0].get("finish_reason", "stop")} ]}

    def generate():
        with _model_lock:
            output = llm(format_prompt(payload), max_tokens=payload.max_tokens,
                         temperature=payload.temperature, top_p=payload.top_p, stream=True, stop=["<|im_end|>"])
            try:
                for chunk in output:
                    text = chunk["choices"][0]["text"]
                    if text:
                        yield f"data: {json.dumps({'choices': [{'delta': {'content': text}}]}, ensure_ascii=False)}\n\n"
            finally:
                close = getattr(output, "close", None)
                if close:
                    close()
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
