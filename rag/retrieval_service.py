"""Lazy model loading; serialized CPU inference; no implicit domain fallback."""
import logging
import math
import os
from functools import lru_cache
from threading import RLock

from .domain_config import RAGUnavailableError, load_domain_config

logger = logging.getLogger(__name__)
_MODEL_LOCK = RLock()


from rag.timing import timed, span, emit, TimedLock, trace_headers, TimingMiddleware, trace_id
from rag.hardware_diagnostics import memory_stage, retrieval_device

@lru_cache(maxsize=2)
@timed('rag.retrieval_service._embedding')
def _embedding(model_name):
    from .gpu_runtime import embedding_model
    retrieval_device("rag.embedding_load", model_name, "directml")
    with memory_stage("rag.embedding_load", model=model_name, configured_device="directml"):
        return embedding_model(model_name)


@lru_cache(maxsize=1)
@timed('rag.retrieval_service._reranker')
def _reranker():
    from .gpu_runtime import DirectMLReranker
    retrieval_device("rag.reranker_load", "BAAI/bge-reranker-v2-m3", "directml")
    with memory_stage("rag.reranker_load", model="BAAI/bge-reranker-v2-m3", configured_device="directml"):
        return DirectMLReranker()


@lru_cache(maxsize=4)
@timed('rag.retrieval_service._index')
def _index(vector_dir, model_name, revision):
    from llama_index.core import StorageContext, load_index_from_storage
    context = StorageContext.from_defaults(persist_dir=vector_dir)
    return load_index_from_storage(context, embed_model=_embedding(model_name))


def _sigmoid(value):
    value = float(value)
    if not math.isfinite(value):
        raise RAGUnavailableError("Reranker returned a non-finite score")
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    exp = math.exp(value)
    return exp / (1 + exp)


def _positive_integer(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    if isinstance(value, str):
        try:
            value = int(value)
        except ValueError as exc:
            raise ValueError(f"{name} must be a positive integer") from exc
    if not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


@timed('rag.retrieval_service.retrieve_and_rerank')
def retrieve_and_rerank(query, top_k=None, domain=None, *, config=None,
                        candidate_k=None):
    """Return a ranked list of (record_id, score, metadata), not one tuple.

    candidate_k: vector-search candidates (default 20).
    top_k: maximum UNIQUE source records after reranking (default 5).
    Every returned record also meets config.minimum_score.
    """
    config = config or load_domain_config(domain)
    candidate_k = _positive_integer(
        candidate_k if candidate_k is not None else config.retrieval_option('candidate_k', 'RAG_CANDIDATE_K', '20'),
        "candidate_k",
    )
    top_k = _positive_integer(
        top_k if top_k is not None else config.retrieval_option('result_k', 'RAG_RESULT_K', '5'), "top_k",
    )
    if top_k > candidate_k:
        raise ValueError("top_k cannot exceed candidate_k")
    if not query.strip():
        return []
    if not config.vector_dir.is_dir():
        raise RAGUnavailableError(f"Index unavailable for domain {config.name}")
    try:
        with TimedLock(_MODEL_LOCK, 'rag.model_lock'):
            with span('rag.index_lookup_or_load'):
                index = _index(str(config.vector_dir), config.embedding_model, config.revision)
            with memory_stage('rag.embedding_and_vector_search'), span('rag.embedding_and_vector_search'):
                candidates = index.as_retriever(similarity_top_k=candidate_k).retrieve(query)
            if not candidates:
                return []
            with memory_stage('rag.reranker_inference'), span('rag.reranker_inference'):
                scores = _reranker().compute_score([[query, node.text] for node in candidates])
            if hasattr(scores, "tolist"):
                scores = scores.tolist()
            if isinstance(scores, (int, float)):
                scores = [scores]
            if len(scores) != len(candidates):
                raise RAGUnavailableError("Reranker returned the wrong number of scores")
            ranked = sorted(
                [(node, _sigmoid(score)) for node, score in zip(candidates, scores)],
                key=lambda item: item[1], reverse=True,
            )
            results = []
            seen_ids = set()
            for node, score in ranked:
                if score < config.minimum_score:
                    break
                record_id = node.metadata.get("id")
                if record_id is None:
                    raise RAGUnavailableError("Retrieved vector has no source record ID")
                # Multiple chunks from one record should not consume multiple slots.
                identity = str(record_id)
                if identity in seen_ids:
                    continue
                seen_ids.add(identity)
                results.append((record_id, score, dict(node.metadata)))
                if len(results) == top_k:
                    break
            logger.info(
                "RAG domain=%s candidates=%d kept=%d cutoff=%.4f scores=%s",
                config.name, len(candidates), len(results), config.minimum_score,
                [round(score, 4) for _, score, _ in results],
            )
            return results
    except (RAGUnavailableError, ValueError):
        raise
    except Exception as exc:
        raise RAGUnavailableError(f"Retrieval failed for domain {config.name}") from exc
