"""Domain settings and immutable SQL/vector snapshots. Paths are repo-relative."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import math
import os
import re

ROOT_DIR = Path(__file__).resolve().parents[1]


class RAGConfigurationError(RuntimeError):
    pass


class RAGUnavailableError(RuntimeError):
    pass


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise RAGConfigurationError(f"Invalid SQL identifier: {value!r}")
    return value


def read_json(path):
    try:
        with Path(path).open(encoding="utf-8-sig") as file:
            return json.load(file)
    except (OSError, ValueError) as exc:
        raise RAGConfigurationError(f"Cannot read JSON: {path}") from exc


@dataclass(frozen=True)
class DomainConfig:
    name: str
    profile: dict
    db_path: Path
    vector_dir: Path
    table: str
    id_column: str
    generation: str
    revision: str
    pointer_path: Path

    @property
    def schema(self):
        return self.profile["schema_mapping"]

    @property
    def primary_entity(self):
        return self.profile["primary_entity"]

    @property
    def minimum_score(self):
        return float(self.profile.get("minimum_score", 0.01))

    @property
    def embedding_model(self):
        return self.profile.get("embedding_model", "BAAI/bge-m3")

    @property
    def evidence_fields(self):
        excluded = set(self.profile.get("excluded_fields", ["timestamp", "email", "email_address"]))
        return [key for key in self.schema if key not in excluded]


def validate_profile(profile):
    if not isinstance(profile, dict):
        raise RAGConfigurationError("Profile must be a JSON object")
    name = profile.get("domain_name", "")
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise RAGConfigurationError("domain_name must contain letters, numbers, '_' or '-'")
    schema = profile.get("schema_mapping")
    if not isinstance(schema, dict) or not schema:
        raise RAGConfigurationError("schema_mapping must be a nonempty object")
    for key, column in schema.items():
        identifier(key)
        if key == "id" or not isinstance(column, str) or not column.strip():
            raise RAGConfigurationError("Schema keys cannot be 'id'; column labels must be nonempty")
    if profile.get("primary_entity") not in schema:
        raise RAGConfigurationError("primary_entity must occur in schema_mapping")
    score = float(profile.get("minimum_score", 0.01))
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise RAGConfigurationError("minimum_score must be between 0 and 1")
    for option in ("excluded_fields", "embedding_fields"):
        fields = profile.get(option, [])
        if not isinstance(fields, list) or any(not isinstance(x, str) for x in fields):
            raise RAGConfigurationError(f"{option} must be a list of strings")
    if profile["primary_entity"] in profile.get("excluded_fields", ["timestamp", "email", "email_address"]):
        raise RAGConfigurationError("primary_entity cannot be excluded from evidence")
    for key in profile.get("embedding_fields", []):
        if key not in schema:
            raise RAGConfigurationError(f"Unknown embedding field: {key}")
    if profile.get("source_id_field") and profile["source_id_field"] not in schema:
        raise RAGConfigurationError("source_id_field must be a schema key")
    return profile


def load_domain_config(domain=None, *, root=None, use_active=True):
    root = Path(root or ROOT_DIR).resolve()
    default = validate_profile(read_json(root / "data/domain_profile.json"))
    name = domain or default["domain_name"]
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise RAGConfigurationError("Invalid domain name")
    is_default = name == default["domain_name"]
    profile = default if is_default else validate_profile(read_json(root / "data/domains" / f"{name}.json"))
    if profile["domain_name"] != name:
        raise RAGConfigurationError("Domain name does not match profile filename")
    pointer = root / "data/rag_generations" / name / "current.json"
    generation = "legacy"
    if use_active and pointer.exists():
        generation = read_json(pointer).get("generation", "")
        if not isinstance(generation, str) or not re.fullmatch(r"[a-f0-9]{32}", generation):
            raise RAGConfigurationError("Invalid active generation")
        directory = pointer.parent / generation
        profile = validate_profile(read_json(directory / "profile.json"))
        if profile["domain_name"] != name:
            raise RAGConfigurationError("Snapshot belongs to another domain")
        db_path, vector_dir = directory / "records.db", directory / "vectors"
        table, id_column = "records", "id"
    else:
        storage = profile.get("storage", {})
        db_path = root / storage.get("database", "data/legal_llm.db" if is_default else f"data/domains/{name}/records.db")
        vector_dir = root / storage.get("vectors", "data/vector_storage" if is_default else f"data/domains/{name}/vectors")
        table = identifier(storage.get("table", "procedures" if is_default else "records"))
        id_column = identifier(storage.get("id_column", "id"))
    # Runtime policy overrides do not rewrite immutable ingestion snapshots.
    if use_active and os.getenv("RAG_MIN_SCORE") is not None:
        profile = dict(profile)
        try:
            profile["minimum_score"] = float(os.environ["RAG_MIN_SCORE"])
        except ValueError as exc:
            raise RAGConfigurationError("RAG_MIN_SCORE must be a number between 0 and 1") from exc
        validate_profile(profile)
    # Includes profile changes and legacy DB/index metadata in the cache identity.
    signature = [json.dumps(profile, sort_keys=True, ensure_ascii=False), str(db_path), str(vector_dir), generation]
    if use_active:
        signature.extend(os.getenv(key, default) for key, default in (
            ("RAG_PREPARE_QUERY", "0"), ("RAG_CANDIDATE_K", "20"), ("RAG_RESULT_K", "5")))
    for path in (db_path, Path(str(db_path) + "-wal"), vector_dir / "docstore.json", vector_dir / "index_store.json"):
        if path.exists():
            stat = path.stat()
            signature.append(f"{path}:{stat.st_size}:{stat.st_mtime_ns}")
    revision = hashlib.sha256("\n".join(signature).encode()).hexdigest()
    return DomainConfig(name, profile, db_path, vector_dir, table, id_column, generation, revision, pointer)


def llm_url():
    return os.getenv("LLM_SERVICE_URLS", os.getenv("LLM_SERVICE_URL", "http://127.0.0.1:8002")).split(",")[0].strip().rstrip("/")
