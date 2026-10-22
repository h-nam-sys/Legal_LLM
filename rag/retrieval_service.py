"""Lazy model loading; serialized CPU inference; no implicit domain fallback."""
import logging
import math
import os
from functools import lru_cache
from threading import RLock

from .domain_config import RAGUnavailableError, load_domain_config
from .record_context import fetch_public_rows, record_excerpt

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


def _diagnostic_node(node, config):
    """Best-effort metadata only; never change retrieval if diagnostics fail."""
    try:
        metadata = node.metadata or {}
        return {'record_id': metadata.get('id'), 'title': metadata.get(config.primary_entity),
                'vector_score': getattr(node, 'score', None)}
    except Exception:
        return {'metadata_unavailable': True}


@timed('rag.retrieval_service.retrieve_and_rerank')
def retrieve_and_rerank(query, top_k=None, domain=None, *, config=None,
                        candidate_k=None, defer_rerank=False):
    """Return a ranked list of (record_id, score, metadata), not one tuple.

    candidate_k: vector-search candidates (default 20).
    top_k: maximum UNIQUE source records after reranking (default 5).
    Every returned record also meets config.minimum_score.
    With defer_rerank=True, return the unique vector pool with placeholder
    scores; ranking and score thresholds are applied after cross-query merging.
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
    diagnostic_task = 'rag.retrieval.diagnostic'
    emit(diagnostic_task, 'request', domain=config.name, generation=getattr(config, 'generation', None),
         query=query, candidate_k=candidate_k, result_k=top_k, minimum_score=config.minimum_score,
         rerank_deferred=bool(defer_rerank))
    if not query.strip():
        emit(diagnostic_task, 'empty', reason='empty_query', domain=config.name)
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
                emit(diagnostic_task, 'empty', reason='vector_search_returned_no_candidates', domain=config.name)
                return []
            emit(diagnostic_task, 'vector_candidates', domain=config.name,
                 candidates=[_diagnostic_node(node, config) for node in candidates])
            ids = [node.metadata.get('id') for node in candidates]
            if any(record_id is None for record_id in ids):
                raise RAGUnavailableError('Retrieved vector has no source record ID')
            if defer_rerank:
                # Discovery scores from different phrases are not comparable.
                # Keep the vector pool; the complete question scores it later.
                seen = set()
                pool = []
                for node in candidates:
                    identity = str(node.metadata['id'])
                    if identity not in seen:
                        pool.append((node.metadata['id'], 0.0, dict(node.metadata)))
                        seen.add(identity)
                emit(diagnostic_task, 'discovered', domain=config.name,
                     retained_ids=[record[0] for record in pool])
                return pool
            rows = fetch_public_rows(ids, config)
            if any(str(record_id) not in rows for record_id in ids):
                raise RAGUnavailableError('Retrieved vector has no matching SQL record; rebuild this domain')
            scope = config.profile.get('description', '')[:180]
            passages = [('Dataset scope: ' + scope + '\n' if scope else '') +
                        record_excerpt(rows[str(record_id)], config) for record_id in ids]
            emit(diagnostic_task, 'source_passages', domain=config.name,
                 records=[{'id': record_id, 'passage': passage} for record_id, passage in zip(ids, passages)])
            with memory_stage('rag.reranker_inference'), span('rag.reranker_inference'):
                scores = _reranker().compute_score([[query, passage] for passage in passages])
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
            emit(diagnostic_task, 'reranked', domain=config.name, candidates=[
                {**_diagnostic_node(node, config), 'raw_reranker_score': raw, 'relevance_score': _sigmoid(raw)}
                for node, raw in zip(candidates, scores)])
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
            # Explain every candidate, including entries beyond the early cutoff.
            decisions, diagnostic_seen, retained_count = [], set(), 0
            for node, score in ranked:
                item = _diagnostic_node(node, config)
                identity = str(item.get('record_id'))
                if score < config.minimum_score:
                    reason = 'below_minimum_score'
                elif retained_count >= top_k:
                    reason = 'result_limit_reached'
                elif identity in diagnostic_seen:
                    reason = 'duplicate_record'
                else:
                    reason = 'kept'
                    diagnostic_seen.add(identity)
                    retained_count += 1
                decisions.append({**item, 'relevance_score': score, 'reason': reason})
            emit(diagnostic_task, 'result', domain=config.name, candidates=decisions,
                 retained_ids=[record[0] for record in results])
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


@timed('rag.retrieval_service.rerank_records')
def rerank_records(question, matches, *, config, top_k):
    """One score scale for the merged pool; apply the result limit afterwards."""
    if not matches:
        return []
    top_k = _positive_integer(top_k, 'top_k')
    task = 'rag.ranking.diagnostic'
    try:
        with TimedLock(_MODEL_LOCK, 'rag.model_lock'):
            rows = fetch_public_rows([m[0] for m in matches], config)
            if any(str(m[0]) not in rows for m in matches):
                raise RAGUnavailableError('Candidate has no matching SQL record')
            scope = config.profile.get('description','')[:180]
            passages = [('Dataset scope: '+scope+'\n' if scope else '') +
                        record_excerpt(rows[str(m[0])],config) for m in matches]
            emit(task,'request',domain=config.name,question=question,
                 candidate_ids=[m[0] for m in matches],result_limit=top_k)
            # Existing DirectML execution uses its configured bounded batch size.
            # Chunk RPC payloads as well; this does not load another model.
            scores = []
            with memory_stage('rag.merged_reranker_inference'):
                for offset in range(0,len(matches),16):
                    raw = _reranker().compute_score([[question,p] for p in passages[offset:offset+16]])
                    if hasattr(raw,'tolist'):
                        raw = raw.tolist()
                    if isinstance(raw,(int,float)):
                        raw = [raw]
                    if len(raw) != len(passages[offset:offset+16]):
                        raise RAGUnavailableError('Reranker returned the wrong number of scores')
                    scores.extend(_sigmoid(score) for score in raw)
            ranked = sorted([(m[0],score,m[2]) for m,score in zip(matches,scores)],
                            key=lambda m:m[1],reverse=True)
            kept = [m for m in ranked if m[1]>=config.minimum_score][:top_k]
            emit(task,'complete',domain=config.name,
                 ranked=[{'id':m[0],'score':m[1]} for m in ranked],
                 retained_ids=[m[0] for m in kept])
            return kept
    except (RAGUnavailableError, ValueError):
        raise
    except Exception as exc:
        raise RAGUnavailableError('Merged candidate ranking unavailable') from exc
