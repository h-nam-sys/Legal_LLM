# Downgrade to 3.11 for maximum AI library and CUDA wheel compatibility
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# FIX: Added the abetlen CUDA 12.1 index URL so it fetches the GPU-accelerated llama-cpp-python
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

COPY . .
