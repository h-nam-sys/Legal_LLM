"""Bounded public-record context from the same immutable retrieval snapshot."""
import re
import sqlite3
from .domain_config import RAGUnavailableError


def fetch_public_rows(record_ids, config):
    identifiers = list(dict.fromkeys(str(i) for i in record_ids))
    if not identifiers:
        return {}
    if not all(re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', name) for name in (config.table, config.id_column)):
        raise RAGUnavailableError('Invalid source database identifiers')
    try:
        with sqlite3.connect(config.db_path.resolve().as_uri() + '?mode=ro', uri=True) as db:
            db.row_factory = sqlite3.Row
            result = {}
            # Bounded chunks also work with older SQLite parameter limits.
            for start in range(0, len(identifiers), 100):
                batch = identifiers[start:start + 100]
                rows = db.execute(f'SELECT * FROM "{config.table}" WHERE "{config.id_column}" IN (' +
                                  ','.join('?' for _ in batch) + ')', batch)
                for row in rows:
                    result[str(row[config.id_column])] = {
                        key: row[key] for key in config.evidence_fields if key in row.keys()}
            return result
    except sqlite3.Error as exc:
        raise RAGUnavailableError(f'Source database unavailable for {config.name}') from exc


def record_excerpt(row, config, max_chars=1800):
    """Use mapped public labels; bound both individual values and total context."""
    keys = list(dict.fromkeys([config.primary_entity] + list(config.evidence_fields)))
    populated = [(config.schema[key], str(row[key]).strip()) for key in keys
                 if key in config.schema and row.get(key) is not None and str(row[key]).strip()]
    limit = max(40, min(400, max_chars // max(1, min(len(populated), 8))))
    text = '\n'.join(label + ': ' + (value if len(value) <= limit else value[:limit] + '…')
                     for label, value in populated)
    return text if len(text) <= max_chars else text[:max_chars - 1] + '…'


def field_samples(rows, config):
    return {key: [str(row[key])[:180] for row in rows[:2]
                  if row.get(key) is not None and str(row[key]).strip()]
            for key in config.evidence_fields if key != config.primary_entity}
