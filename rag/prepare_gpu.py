"""One-time export and local DirectML validation; never starts CPU fallback."""
import argparse
from collections import Counter
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
import uuid

import numpy as np

from .domain_config import ROOT_DIR
from .gpu_runtime import (EMBED_MODEL, RERANK_MODEL, runtime_root, make_session,
                          encode_embeddings, score_pairs, prepare_pairs)

QUESTIONS = [
    'Thủ tục đăng ký khai sinh cần hồ sơ gì?',
    'Thủ tục đăng ký kết hôn mất bao lâu?',
    'Đăng ký kết hôn có yếu tố nước ngoài',
    'Thủ tục xác nhận tình trạng hôn nhân có lệ phí không?',
    'Thủ tục nhận cha mẹ con', 'Giải quyết chế độ trợ cấp thờ cúng liệt sĩ',
    'Mượn sách thư viện cần gì?', 'Xin cấp lại thẻ sinh viên',
    'dang ky ket hon can giay to gi va mat bao lau',
    '  Thủ tục đăng ký khai tử\n', 'Còn lệ phí của thủ tục đó?',
    '  Hồ sơ\nđăng ký khai sinh  ',
]
PASSAGES = [
    'Thủ tục đăng ký khai sinh. Thành phần hồ sơ gồm giấy chứng sinh và giấy tờ tùy thân.',
    'Đăng ký kết hôn. Thời gian giải quyết trong ngày làm việc, có điều kiện sau 15 giờ.',
    'Đăng ký kết hôn có yếu tố nước ngoài. Thời gian giải quyết 5 ngày làm việc.',
    'Mượn sách thư viện. Sinh viên mang thẻ sinh viên đến quầy phục vụ.',
    'Thủ tục đăng ký khai tử. Thời gian giải quyết trong ngày làm việc.',
]
PAIRS = [(q, p) for q in QUESTIONS[:4] for p in PASSAGES] + [
    ('giấy tờ ' * 500, 'khai sinh ' * 700),
    ('Mượn sách thư viện', 'thẻ sinh viên ' * 700),
    ('What documents are required?', 'Birth registration documentation'),
]


def write_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def export_native(model, output, *, kind, max_length):
    """Export with the installed PyTorch/Transformers; no Optimum dependency."""
    import torch
    version = torch.__version__.split('+')[0].split('.')
    if tuple(map(int, version[:2])) < (2, 6):
        raise RuntimeError('Native GPU export requires PyTorch >= 2.6. Your Torch was not upgraded; report its version before proceeding.')
    # Eager attention exports ordinary operators rather than vendor attention kernels.
    if hasattr(model, 'set_attn_implementation'):
        model.set_attn_implementation('eager')
    else:
        raise RuntimeError('This model cannot select eager attention with the installed Transformers; no export was activated.')

    class ExportWrapper(torch.nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone

        def forward(self, input_ids, attention_mask):
            result = self.backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            if kind == 'embedding':
                return torch.nn.functional.normalize(result.last_hidden_state[:, 0].float(), p=2, dim=1, eps=1e-12)
            return result.logits.float()

    wrapper = ExportWrapper(model).eval()
    # Non-padding IDs and sample batch > 1 allow genuinely dynamic batch/sequence axes.
    inputs = (torch.full((2, 16), 10, dtype=torch.int64),
              torch.ones((2, 16), dtype=torch.int64))
    batch = torch.export.Dim('batch', min=1, max=32)
    sequence = torch.export.Dim('sequence', min=2, max=max_length)
    with torch.no_grad():
        torch.onnx.export(wrapper, inputs, str(output / 'model.onnx'),
            input_names=['input_ids', 'attention_mask'],
            output_names=['dense_embedding' if kind == 'embedding' else 'logits'],
            dynamic_shapes=({0: batch, 1: sequence}, {0: batch, 1: sequence}),
            dynamo=True, external_data=True, opset_version=18,
            report=True, artifacts_dir=str(output))
    import onnx
    graph = onnx.load(str(output / 'model.onnx'), load_external_data=False)
    if any(entry.version > 20 for entry in graph.opset_import if entry.domain in ('', 'ai.onnx')):
        raise RuntimeError('Export exceeds DirectML ONNX opset support; no GPU activation.')
    from .onnx_shape_compat import rewrite_graph
    changes = rewrite_graph(graph)
    if changes:
        onnx.save_model(graph, str(output / 'model.onnx'))
        print(f'Canonicalized {len(changes)} attention reshapes for DirectML.', flush=True)
    onnx.checker.check_model(str(output / 'model.onnx'))
    return len(changes)


def export_one(kind, output, *, precision='fp32'):
    """Runs in a short-lived process, so CPU reference weights are released."""
    output.mkdir(parents=True, exist_ok=False)
    if kind == 'embedding':
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        reference = HuggingFaceEmbedding(model_name=EMBED_MODEL, device='cpu')
        model = reference._model[0].auto_model
        metadata = {
            'max_length': int(reference._model.max_seq_length),
            'prompts': {key: reference._model.prompts.get(key, '') or '' for key in ('query', 'text')},
            'do_lower_case': bool(getattr(reference._model[0], 'do_lower_case', False)),
            'source_revision': getattr(model.config, '_commit_hash', None),
        }
        query = [reference.get_query_embedding(q) for q in QUESTIONS]
        text = [reference.get_text_embedding(q) for q in QUESTIONS]
        np.savez(output / 'reference.npz', query=np.asarray(query), text=np.asarray(text))
        reference._model.tokenizer.save_pretrained(output)
    else:
        from FlagEmbedding import FlagReranker
        reference = FlagReranker(RERANK_MODEL, devices=['cpu'], use_fp16=False)
        model = reference.model
        metadata = {
            'max_length': int(getattr(reference, 'max_length', 512)),
            'query_max_length': int(getattr(reference, 'query_max_length', None) or getattr(reference, 'max_length', 512) * 3 // 4),
            'source_revision': getattr(model.config, '_commit_hash', None),
        }
        if metadata['max_length'] != 512:
            raise RuntimeError('Unexpected installed FlagReranker defaults; refusing to change truncation.')
        observed_tokens = set()
        def capture_inputs(module, args, kwargs):
            ids = kwargs['input_ids'].detach().cpu().tolist()
            masks = kwargs['attention_mask'].detach().cpu().tolist()
            observed_tokens.update(tuple(token for token, mask in zip(row, attention) if mask)
                                   for row, attention in zip(ids, masks))
        hook = model.register_forward_pre_hook(capture_inputs, with_kwargs=True)
        try:
            values = reference.compute_score([list(pair) for pair in PAIRS], batch_size=1)
        finally:
            hook.remove()
        # Detect actual installed FlagEmbedding tokenization rather than assuming a version.
        for mode in ('split', 'joint'):
            try:
                metadata['pair_mode'] = mode
                rows = prepare_pairs(reference.tokenizer, PAIRS, metadata)
                expected_tokens = {tuple(row['input_ids']) for row in rows}
                if expected_tokens == observed_tokens:
                    break
            except (AttributeError, TypeError, ValueError):
                continue
        else:
            raise RuntimeError('Installed reranker tokenization differs from supported GPU preprocessing; no activation.')
        np.savez(output / 'reference.npz', logits=np.asarray(values).reshape(-1))
        reference.tokenizer.save_pretrained(output)
    metadata['precision'] = precision
    model.eval()
    if precision == 'fp16':
        model.half()
    else:
        model.float()
    # Reference outputs above always come from the original FP32 CPU model.
    print(f'Exporting {kind}; services should be stopped during this one-time CPU export.', flush=True)
    model.config.save_pretrained(output)
    shape_count = export_native(model, output, kind=kind, max_length=metadata['max_length'])
    if shape_count:
        metadata['directml_shape_compat'] = 1
        metadata['directml_shape_compat_nodes'] = shape_count
    del model, reference
    gc.collect()
    # Bind validation to the actual exported model and tokenizer artifacts.
    files = {p.relative_to(output).as_posix(): sha256(p) for p in output.rglob('*') if p.is_file()}
    write_json(output / 'exported.json', {'metadata': metadata, 'files': files})


def checked_export(path):
    data = json.loads((path / 'exported.json').read_text(encoding='utf-8'))
    for filename, digest in data['files'].items():
        file = path / filename
        if not file.resolve().is_relative_to(path.resolve()) or sha256(file) != digest:
            raise RuntimeError('Export changed or incomplete: ' + str(file))
    return data['metadata']


def check_numeric(actual, expected, *, embedding=False):
    """Check usable outputs; report small precision differences without blocking setup."""
    actual, expected = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    if actual.shape != expected.shape or not actual.size or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise RuntimeError('GPU/reference shape mismatch, empty output or non-finite values')
    error = float(np.max(np.abs(actual - expected)))
    metrics = {'max_absolute_error': error}
    if embedding:
        norms = np.linalg.norm(actual, axis=-1)
        if np.any(norms < 1e-8):
            raise RuntimeError('GPU returned a zero embedding')
        cosine = np.sum(actual * expected, axis=-1) / np.maximum(
            norms * np.linalg.norm(expected, axis=-1), 1e-12)
        metrics['minimum_cosine_similarity'] = float(cosine.min())
        # Broad sanity guards, not a claim of measured retrieval-quality equivalence.
        if cosine.min() < 0.99 or error > 0.05:
            raise RuntimeError('GPU embeddings differ substantially from the reference: ' + str(metrics))
        metrics['small_precision_difference'] = bool(cosine.min() < 0.99999 or error > 0.001)
    else:
        if error > 0.5:
            raise RuntimeError('GPU reranker logits differ substantially from the reference: ' + str(metrics))
        metrics['small_precision_difference'] = bool(error > 0.002)
        changed = []
        for offset in range(0, min(20, actual.size), 5):
            if not np.array_equal(np.argsort(-actual[offset:offset + 5]), np.argsort(-expected[offset:offset + 5])):
                changed.append(offset // 5)
        # Report rank changes for review; tiny logit shifts can swap close candidates.
        metrics['reference_order_changed_groups'] = changed
    return metrics


def check_profile(path):
    """Allow CPU shape bookkeeping/transfers, reject CPU model tensor arithmetic."""
    events = json.loads(Path(path).read_text(encoding='utf-8'))
    counts = Counter()
    rejected = []
    gpu_nodes = 0
    for event in events:
        args = event.get('args', {})
        provider = args.get('provider')
        if not provider:
            continue
        counts[provider] += 1
        if provider == 'DmlExecutionProvider':
            gpu_nodes += 1
            continue
        op = args.get('op_name', '')
        if op.startswith('Memcpy'):
            continue
        shapes = args.get('output_type_shape', [])
        shape_only = bool(shapes) and all(set(item).issubset({'int64', 'int32', 'bool'}) for item in shapes)
        if not shape_only:
            rejected.append({'name': event.get('name'), 'operator': op, 'provider': provider})
    if not gpu_nodes or rejected:
        raise RuntimeError('GPU placement validation failed; CPU tensor inference is not allowed: ' + str(rejected[:10]))
    return dict(counts)


def verify_existing_indexes(session, tokenizer, metadata):
    """Compare GPU document vectors with the active persisted index, without rewriting it."""
    from .domain_config import list_domain_configs
    from llama_index.core import StorageContext
    from llama_index.core.schema import MetadataMode
    count = 0
    for config in list_domain_configs():
        if config.embedding_model != EMBED_MODEL:
            raise RuntimeError('Unsupported active embedding model: ' + config.embedding_model)
        storage = StorageContext.from_defaults(persist_dir=str(config.vector_dir))
        store = storage.vector_store.to_dict()
        vectors = store.get('embedding_dict')
        if not isinstance(vectors, dict) or not vectors:
            raise RuntimeError('Cannot validate persisted vectors for ' + config.name)
        ids = sorted(vectors)
        # Spread 32 checks across each domain, rather than checking only its first records.
        indices = np.unique(np.linspace(0, len(ids) - 1, min(32, len(ids)), dtype=int))
        for index in indices:
            node_id = ids[int(index)]
            node = storage.docstore.get_node(node_id)
            text = node.get_content(metadata_mode=MetadataMode.EMBED)
            actual = encode_embeddings(session, tokenizer, [text], metadata, 'text')
            check_numeric(actual, [vectors[node_id]], embedding=True)
            count += 1
    if not count:
        raise RuntimeError('No active persisted vectors were checked; refusing to activate against an unknown index.')
    return count


def validate_generation(directory, device_id):
    from transformers import AutoTokenizer
    # Validate under the actual Qwen VRAM allocation, rather than on an empty GPU.
    from .domain_config import llm_url
    with urllib.request.urlopen(llm_url().rstrip('/') + '/v1/rag/status', timeout=5) as response:
        status = json.load(response)
    if status.get('gpu_protocol') != 1 or not status.get('qwen_ready'):
        raise RuntimeError('Start the patched LLM service and wait for Qwen before --validate.')
    if status.get('embedding_loaded') or status.get('reranker_loaded') or status.get('gpu_queue_busy'):
        raise RuntimeError('Restart the LLM service and do not chat before validation, to avoid duplicate GPU retrieval weights.')
    embedding_meta = checked_export(directory / 'embedding')
    reranker_meta = checked_export(directory / 'reranker')
    embed_tokenizer = AutoTokenizer.from_pretrained(directory / 'embedding', local_files_only=True)
    rerank_tokenizer = AutoTokenizer.from_pretrained(directory / 'reranker', local_files_only=True)
    # Both sessions coexist throughout checks/benchmark, just as in chat.
    precision = embedding_meta.get('precision', 'fp32')
    if precision != reranker_meta.get('precision', 'fp32'):
        raise RuntimeError('Mixed export precisions require a separate validation; no activation.')
    embed = make_session(directory / 'embedding' / 'model.onnx', device_id=device_id, profile=True, precision=precision)
    rerank = make_session(directory / 'reranker' / 'model.onnx', device_id=device_id, profile=True, precision=precision)
    reference = np.load(directory / 'embedding' / 'reference.npz')
    report = {}
    for prompt in ('query', 'text'):
        report['embedding_' + prompt] = check_numeric(
            encode_embeddings(embed, embed_tokenizer, QUESTIONS, embedding_meta, prompt),
            reference[prompt], embedding=True)
    expected = np.load(directory / 'reranker' / 'reference.npz')['logits']
    report['reranker'] = check_numeric(score_pairs(rerank, rerank_tokenizer, PAIRS, reranker_meta), expected)
    report['existing_index_vectors_checked'] = verify_existing_indexes(embed, embed_tokenizer, embedding_meta)
    # Tune a small batch set on the real 20-candidate workload; never drop candidates.
    timings = {}
    failures = {}
    for batch in (1, 2, 4, 8):
        try:
            # Also require the chosen batch to fit preserved 512-token passages.
            stress = score_pairs(rerank, rerank_tokenizer, [PAIRS[-2]] * batch, reranker_meta, batch)
            check_numeric(stress, np.repeat(expected[-2], batch))
            values = score_pairs(rerank, rerank_tokenizer, PAIRS[:20], reranker_meta, batch)
            check_numeric(values, expected[:20])
            samples = []
            for _ in range(2):
                start = time.perf_counter()
                score_pairs(rerank, rerank_tokenizer, PAIRS[:20], reranker_meta, batch)
                samples.append((time.perf_counter() - start) * 1000)
            timings[batch] = float(np.median(samples))
        except Exception as exc:
            # Failed larger batches are NOT used; batch one must always have passed.
            if not any(word in str(exc).lower() for word in ('memory', 'allocat', 'resource', 'out of')):
                raise
            failures[batch] = type(exc).__name__ + ': ' + str(exc)[:200]
            break
    if not timings:
        raise RuntimeError('No DirectML reranker batch succeeded')
    batch_size = min(timings, key=timings.get)
    report['embedding_max_length_preserved'] = embedding_meta['max_length']
    embedding_samples = []
    for question in QUESTIONS[:3]:
        start = time.perf_counter()
        encode_embeddings(embed, embed_tokenizer, [question], embedding_meta, 'query')
        embedding_samples.append((time.perf_counter() - start) * 1000)
    report['embedding_warm_ms'] = float(np.median(embedding_samples))
    report['reranker_20_pairs_ms_by_batch'] = timings
    report['larger_batch_failures'] = failures
    report['placement_embedding'] = check_profile(embed.end_profiling())
    report['placement_reranker'] = check_profile(rerank.end_profiling())
    result = {
        'schema': 1, 'models': [EMBED_MODEL, RERANK_MODEL], 'provider': 'DmlExecutionProvider',
        'precision': precision, 'device_id': device_id, 'reranker_batch_size': batch_size,
        'embedding': embedding_meta, 'reranker': reranker_meta, 'report': report,
    }
    import onnxruntime as ort
    result['onnxruntime_version'] = ort.__version__
    write_json(directory / 'validated.json', result)
    (ROOT_DIR / 'logs').mkdir(exist_ok=True)
    write_json(ROOT_DIR / 'logs' / 'gpu_validation.json', result)
    write_json(runtime_root() / 'current.json', {'generation': directory.name})
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print('GPU retrieval activated. Setup will now release its test sessions; you can then chat normally.')


def shared_service_check(directory):
    from .gpu_client import gpu_call
    expected_embedding = np.load(directory / 'embedding' / 'reference.npz')['query'][:1]
    expected_logits = np.load(directory / 'reranker' / 'reference.npz')['logits'][:20]
    samples = []
    for _ in range(2):
        start = time.perf_counter()
        vectors = gpu_call('/v1/rag/embeddings', {'texts': QUESTIONS[:1], 'kind': 'query'})['embeddings']
        embedding_ms = (time.perf_counter() - start) * 1000
        check_numeric(vectors, expected_embedding, embedding=True)
        start = time.perf_counter()
        logits = gpu_call('/v1/rag/rerank', {'pairs': PAIRS[:20]})['logits']
        reranker_ms = (time.perf_counter() - start) * 1000
        check_numeric(logits, expected_logits)
        samples.append({'embedding_ms': embedding_ms, 'reranker_20_pairs_ms': reranker_ms})
    result = json.loads((ROOT_DIR / 'logs' / 'gpu_validation.json').read_text(encoding='utf-8'))
    result['report']['shared_service_runs'] = samples
    write_json(ROOT_DIR / 'logs' / 'gpu_validation.json', result)
    print(json.dumps({'shared_service_runs': samples}, indent=2))
    print('Shared GPU service passed output sanity checks and is warm. You can chat normally now.')


def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT_DIR / 'backend/.env')
    load_dotenv(ROOT_DIR / '.env')
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--export', action='store_true')
    mode.add_argument('--validate', action='store_true')
    mode.add_argument('--fix-shapes', action='store_true')
    mode.add_argument('--worker', choices=['embedding', 'reranker', 'validate'], help=argparse.SUPPRESS)
    parser.add_argument('--output', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--device-id', type=int, default=0)
    parser.add_argument('--precision', choices=['fp32', 'fp16'], default='fp32')
    args = parser.parse_args()
    if args.worker:
        if args.worker == 'validate':
            validate_generation(args.output, args.device_id)
        else:
            export_one(args.worker, args.output, precision=args.precision)
        return
    if args.fix_shapes:
        from .onnx_shape_compat import patch_pending
        patch_pending()
        return
    root = runtime_root()
    root.mkdir(parents=True, exist_ok=True)
    if args.export:
        try:
            import torch, onnx, onnxscript
        except ImportError as exc:
            raise RuntimeError('Export dependencies are missing. Run python install_gpu_dependencies.py successfully before --export.') from exc
        generation = uuid.uuid4().hex
        directory = root / generation
        directory.mkdir()
        for kind in ('embedding', 'reranker'):
            subprocess.run([sys.executable, '-m', 'rag.prepare_gpu', '--worker', kind,
                            '--output', str(directory / kind), '--precision', args.precision], cwd=ROOT_DIR, check=True)
        write_json(root / 'pending.json', {'generation': generation})
        print('Export complete. Start the usual services, wait for Qwen, then run python -m rag.prepare_gpu --validate.')
    else:
        pending = json.loads((root / 'pending.json').read_text(encoding='utf-8'))['generation']
        if not isinstance(pending, str) or not pending.isalnum():
            raise RuntimeError('Invalid pending generation')
        directory = root / pending
        # Exit the benchmark process before loading shared-service GPU models.
        subprocess.run([sys.executable, '-m', 'rag.prepare_gpu', '--worker', 'validate',
                        '--output', str(directory), '--device-id', str(args.device_id)],
                       cwd=ROOT_DIR, check=True)
        shared_service_check(directory)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(f'GPU setup stopped: {error}. Active domain data/indexes were not modified.', file=sys.stderr)
        raise
