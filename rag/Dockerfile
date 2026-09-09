FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen --no-cache

COPY . .

EXPOSE 8002
CMD ["uv", "run", "uvicorn", "src.rag_service:app", "--host", "0.0.0.0", "--port", "8002"]