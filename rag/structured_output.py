"""Decode small model values without executing code or repairing their meaning."""
import ast
import json


def parse_model_value(text):
    if not isinstance(text, str) or len(text) > 65536:
        raise ValueError('Expected bounded model text')
    text = text.strip()
    if text.startswith('```') and text.endswith('```'):
        text = '\n'.join(text.splitlines()[1:-1]).strip()
    try:
        return json.loads(text)
    except (ValueError, RecursionError):
        pass
    # Some local models use Python-style string quotes. Accept literal containers
    # only; callers still validate every key, type, index and allowed value.
    try:
        tree = ast.parse(text, mode='eval')
        nodes = list(ast.walk(tree))
        if len(nodes) > 4096 or any(not isinstance(node, (
            ast.Expression, ast.List, ast.Dict, ast.Constant, ast.Load,
        )) for node in nodes):
            raise ValueError('Unsupported model literal')
        return ast.literal_eval(tree)
    except (SyntaxError, ValueError, TypeError, RecursionError) as exc:
        raise ValueError('Invalid structured model output') from exc
