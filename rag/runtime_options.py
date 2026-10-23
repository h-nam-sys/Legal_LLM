"""Reversible engineering changes; enabled by default, no .env edits required."""
import os

OPTIONS = ('RAG_RESUME_CLARIFICATION', 'RAG_LOW_SCORE_RECOVERY',
           'RAG_ANSWER_GUIDANCE', 'RAG_ASCII_QUERY_GUARD')


def enabled(name):
    if name not in OPTIONS:
        raise ValueError('Unknown runtime option: ' + name)
    return os.getenv(name, '1').strip().lower() in {'1', 'true', 'yes', 'on'}


def signature():
    """A cache entry must not cross an ablation/configuration boundary."""
    return tuple(enabled(name) for name in OPTIONS)
