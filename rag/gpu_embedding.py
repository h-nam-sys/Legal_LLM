"""LlamaIndex adapter preserving BGE-M3 dense embedding semantics."""
import asyncio
from typing import Any
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.bridge.pydantic import PrivateAttr

from .gpu_runtime import EMBED_MODEL, read_active, make_session, encode_embeddings


class LocalDirectMLEmbedding(BaseEmbedding):
    _session: Any = PrivateAttr()
    _tokenizer: Any = PrivateAttr()
    _metadata: dict = PrivateAttr()

    def __init__(self):
        from transformers import AutoTokenizer
        directory, metadata = read_active()
        super().__init__(model_name=EMBED_MODEL, embed_batch_size=1)
        self._metadata = metadata['embedding']
        self._tokenizer = AutoTokenizer.from_pretrained(directory / 'embedding', local_files_only=True)
        self._session = make_session(directory / 'embedding' / 'model.onnx', device_id=metadata['device_id'], precision=metadata['precision'])

    def _get_query_embedding(self, query):
        return encode_embeddings(self._session, self._tokenizer, [query], self._metadata, 'query')[0]

    async def _aget_query_embedding(self, query):
        return await asyncio.to_thread(self._get_query_embedding, query)

    def _get_text_embedding(self, text):
        return encode_embeddings(self._session, self._tokenizer, [text], self._metadata, 'text')[0]

    async def _aget_text_embedding(self, text):
        return await asyncio.to_thread(self._get_text_embedding, text)

    def _get_text_embeddings(self, texts):
        return encode_embeddings(self._session, self._tokenizer, texts, self._metadata, 'text')
