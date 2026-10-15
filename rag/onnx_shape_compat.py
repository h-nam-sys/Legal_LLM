"""Canonicalize proven BGE attention reshapes; never load external weight data."""
from datetime import datetime, timezone
import hashlib
import json
from math import prod
from pathlib import Path
import os
import tempfile


def tensor_shape(info):
    kind = info.type.tensor_type
    return [dim.dim_value if dim.HasField('dim_value') else dim.dim_param or None
            for dim in kind.shape.dim]


def rewrite_graph(model):
    import onnx
    from onnx import helper, numpy_helper
    graph = model.graph
    shapes = {info.name: tensor_shape(info) for info in list(graph.input) + list(graph.output) + list(graph.value_info)}
    producers = {out: node for node in graph.node for out in node.output}
    initializers = {item.name: item for item in graph.initializer}
    names = set(initializers) | set(producers)
    changes = []

    def attr(node, name, default=None):
        value = next((value for value in node.attribute if value.name == name), None)
        return helper.get_attribute_value(value) if value is not None else default

    def constant_vector(name):
        value = initializers.get(name)
        if value is None:
            node = producers.get(name)
            if node is None or node.op_type != 'Constant': return None
            value = attr(node, 'value')
        # Shape constants only. Never read a model weight or external file.
        if value is None or value.data_type != onnx.TensorProto.INT64 or value.data_location == onnx.TensorProto.EXTERNAL:
            return None
        if len(value.dims) > 1 or prod(value.dims) > 16: return None
        return numpy_helper.to_array(value).reshape(-1).tolist()

    def requested_shape(node, prefix):
        shape_source = producers.get(node.input[1])
        if shape_source is None or shape_source.op_type != 'Concat' or attr(shape_source, 'axis', 0) != 0:
            return None
        inputs = list(shape_source.input)
        if len(inputs) < 3: return None
        # Prove the first two entries are the same batch and sequence dimensions.
        for dimension in (0, 1):
            part = producers.get(inputs[dimension])
            if part is None or part.op_type != 'Shape' or len(part.input) != 1: return None
            if attr(part, 'start', 0) != dimension or attr(part, 'end') != dimension + 1: return None
            if shapes.get(part.input[0], [])[:2] != prefix: return None
        tail = []
        for value in inputs[2:]:
            values = constant_vector(value)
            if values is None: return None
            tail.extend(values)
        return tail

    for node in graph.node:
        if node.op_type != 'Reshape' or len(node.input) != 2 or len(node.output) != 1:
            continue
        source = shapes.get(node.input[0], [])
        target = shapes.get(node.output[0], [])
        # BGE attention splits/joins only; do not rewrite arbitrary model reshapes.
        if (len(source), len(target)) not in ((3, 4), (4, 3)): continue
        if source[:2] != target[:2] or any(value is None or value == '' for value in source[:2]): continue
        if not all(isinstance(value, int) and value > 0 for value in source[2:] + target[2:]): continue
        if prod(source[2:]) != prod(target[2:]): continue
        tail = requested_shape(node, source[:2])
        if tail is None or len(tail) != len(target) - 2 or tail.count(-1) != 1: continue
        if any(value != -1 and value <= 0 for value in tail): continue
        known = prod(value for value in tail if value != -1)
        inferred, remainder = divmod(prod(source[2:]), known)
        if remainder or [inferred if value == -1 else value for value in tail] != target[2:]: continue
        new_shape = [0, 0] + target[2:]
        name = 'dml_attention_shape_' + str(len(changes))
        while name in names: name += '_'
        names.add(name)
        graph.initializer.append(helper.make_tensor(name, onnx.TensorProto.INT64, [len(new_shape)], new_shape))
        original_shape_input = node.input[1]
        node.input[1] = name
        attributes = [value for value in node.attribute if value.name != 'allowzero']
        del node.attribute[:]
        node.attribute.extend(attributes)
        node.attribute.append(helper.make_attribute('allowzero', 0))
        changes.append({'node': node.name, 'source': source, 'target': target,
                        'old_shape_input': original_shape_input, 'new_shape': new_shape})
    return changes


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def replace_bytes(path, data):
    fd, name = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream: stream.write(data)
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)


def patch_pending():
    import onnx
    from .gpu_runtime import runtime_root
    root = runtime_root()
    generation = json.loads((root / 'pending.json').read_text(encoding='utf-8'))['generation']
    if not isinstance(generation, str) or not generation.isalnum(): raise ValueError('Invalid pending generation')
    current = root / 'current.json'
    if current.exists() and json.loads(current.read_text(encoding='utf-8')).get('generation') == generation:
        raise RuntimeError('Pending generation is active. Refusing to edit live model files.')
    directory = root / generation
    plans = []
    reports = {}
    for kind in ('embedding', 'reranker'):
        graph_path = directory / kind / 'model.onnx'
        manifest_path = directory / kind / 'exported.json'
        original = graph_path.read_bytes()
        original_manifest = manifest_path.read_bytes()
        manifest = json.loads(original_manifest)
        if manifest['files'].get('model.onnx') != sha256(original):
            raise RuntimeError(kind + ' graph differs from its export manifest; no files changed.')
        if manifest.get('metadata', {}).get('directml_shape_compat') == 1:
            print(kind + ': compatible reshape form already applied.')
            continue
        model = onnx.load(str(graph_path), load_external_data=False)
        # Preserve the serialized original tensors, including external weight references.
        tensors_before = [value.SerializeToString() for value in model.graph.initializer]
        changes = rewrite_graph(model)
        if not changes: raise RuntimeError('No proven attention reshape pattern found in ' + kind + '; no files changed.')
        failing = next((node for node in model.graph.node if node.name == 'node_view_1'), None)
        if failing is not None and not any(change['node'] == failing.name for change in changes):
            raise RuntimeError('Known failing node was not safely matched in ' + kind + '; no files changed.')
        if [value.SerializeToString() for value in model.graph.initializer[:len(tensors_before)]] != tensors_before:
            raise RuntimeError('Original tensors changed unexpectedly; no files changed.')
        patched = model.SerializeToString()
        manifest['files']['model.onnx'] = sha256(patched)
        manifest['metadata']['directml_shape_compat'] = 1
        reports[kind] = changes
        manifest['metadata']['directml_shape_compat_nodes'] = len(changes)
        updated = (json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
        # Check the candidate graph in its original directory so external weight paths resolve.
        temporary = graph_path.with_name('model.shape_candidate.onnx')
        try:
            temporary.write_bytes(patched)
            onnx.checker.check_model(str(temporary))
        finally:
            temporary.unlink(missing_ok=True)
        plans.extend([(graph_path, original, patched), (manifest_path, original_manifest, updated)])
    if not plans:
        print('Pending graphs already use the compatible reshape form. Run --validate.')
        return
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    backup = directory / ('shape_backups_' + stamp)
    backup.mkdir()
    for path, old, new in plans:
        saved = backup / path.relative_to(directory)
        saved.parent.mkdir(parents=True, exist_ok=True)
        saved.write_bytes(old)
    written = []
    try:
        for path, old, new in plans:
            replace_bytes(path, new)
            written.append((path, old))
    except BaseException:
        for path, old in reversed(written): replace_bytes(path, old)
        raise
    (backup / 'changes.json').write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding='utf-8')
    for kind, changes in reports.items(): print(f'{kind}: canonicalized {len(changes)} attention reshapes.')
    print('External weights unchanged. Backups: ' + str(backup))
    print('Keep Qwen running, do not chat yet, and run: python -m rag.prepare_gpu --validate')
