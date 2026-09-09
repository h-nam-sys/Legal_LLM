FROM python:3.11-slim

WORKDIR /app

# Point pip directly to pre-compiled CPU wheels for llama-cpp-python
ENV PIP_EXTRA_INDEX_URL="https://abetlen.github.io/llama-cpp-python/whl/cpu"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["fastapi", "run", "main.py", "--host", "0.0.0.0", "--port", "8000"]