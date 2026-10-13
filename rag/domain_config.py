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

    def retrieval_option(self, name, env_name, default):
        return self.profile.get(name, os.getenv(env_name, default))


def atomic_json(path, value):
    """Replace one JSON file only after a complete, flushed write."""
    import uuid
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('w', encoding='utf-8') as file:
            json.dump(value, file, ensure_ascii=False, indent=2, allow_nan=False)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def mode_settings(config, mode):
    return config.profile.get('modes', {}).get(mode, {})


def config_for_mode(config, mode):
    from dataclasses import replace
    overrides = {k: v for k, v in mode_settings(config, mode).items()
                 if k in {'minimum_score', 'candidate_k', 'result_k'}}
    if not overrides:
        return config
    profile = validate_profile({**config.profile, **overrides})
    revision = hashlib.sha256((config.revision + mode + json.dumps(overrides, sort_keys=True)).encode()).hexdigest()
    return replace(config, profile=profile, revision=revision)


# These settings may change without rebuilding the immutable SQL/vector pair.
RUNTIME_KEYS = {'description', 'routing_examples', 'default_mode', 'minimum_score',
                'candidate_k', 'result_k', 'selection_score_margin', 'field_aliases',
                'title_prefixes', 'exact_field_output', 'exact_output_block_terms',
                'query_preparation_prompt', 'record_selection_prompt', 'field_selection_prompt',
                'modes', 'fallback_message', 'allow_forms', 'allow_external_search'}


def validate_runtime(settings, base):
    if not isinstance(settings, dict) or set(settings) - RUNTIME_KEYS:
        raise RAGConfigurationError('Runtime settings contain unsupported or structural keys')
    validate_profile({**base, **settings})
    return settings


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
    aliases = profile.get('field_aliases', {})
    if not isinstance(aliases, dict) or any(k not in schema or not isinstance(v, list)
            or any(not isinstance(x, str) or not x.strip() for x in v) for k, v in aliases.items()):
        raise RAGConfigurationError('field_aliases must map schema keys to lists of phrases')
    for key in ('title_prefixes', 'exact_output_block_terms', 'routing_examples'):
        if key in profile and (not isinstance(profile[key], list) or any(not isinstance(v, str) for v in profile[key])):
            raise RAGConfigurationError(f'{key} must be a list of strings')
    for key in ('exact_field_output', 'allow_forms', 'allow_external_search'):
        if key in profile and type(profile[key]) is not bool:
            raise RAGConfigurationError(f'{key} must be a boolean')
    for key in ('description', 'embedding_prefix', 'fallback_message', 'query_preparation_prompt',
                'record_selection_prompt', 'field_selection_prompt'):
        if key in profile and not isinstance(profile[key], str):
            raise RAGConfigurationError(f'{key} must be text')
    def validate_options(options):
        if not isinstance(options, dict):
            raise RAGConfigurationError('Mode settings must be objects')
        for key in ('candidate_k', 'result_k', 'max_tokens'):
            if key in options and (type(options[key]) is not int or not 1 <= options[key] <= (4096 if key == 'max_tokens' else 200)):
                raise RAGConfigurationError(f'{key} is outside its supported range')
        if options.get('result_k', profile.get('result_k', 5)) > options.get('candidate_k', profile.get('candidate_k', 20)):
            raise RAGConfigurationError('result_k cannot exceed candidate_k')
        for key, low, high in [('minimum_score', 0, 1), ('temperature', 0, 2), ('top_p', 0.01, 1)]:
            if key in options and (isinstance(options[key], bool) or not isinstance(options[key], (int, float))
                                  or not math.isfinite(options[key]) or not low <= options[key] <= high):
                raise RAGConfigurationError(f'Invalid {key}')
        for key in ('system_prompt', 'user_template', 'direct_prompt'):
            if key in options and not isinstance(options[key], str):
                raise RAGConfigurationError(f'{key} must be text')
        if 'user_template' in options:
            import string
            parsed = list(string.Formatter().parse(options['user_template']))
            if any(spec or conversion for _, name, spec, conversion in parsed if name is not None):
                raise RAGConfigurationError('user_template placeholders cannot use format specs or conversions')
            fields = [name for _, name, spec, conversion in parsed if name is not None]
            if any(name not in {'context', 'query'} for name in fields) or not {'context', 'query'}.issubset(fields):
                raise RAGConfigurationError('user_template must use {context} and {query} only')
    validate_options(profile)
    if profile.get('default_mode', 'strict') not in {'strict', 'friendly'}:
        raise RAGConfigurationError('default_mode must be strict or friendly')
    modes = profile.get('modes', {})
    if not isinstance(modes, dict) or set(modes) - {'strict', 'friendly'}:
        raise RAGConfigurationError('modes must contain strict/friendly only')
    for options in modes.values():
        if set(options) - {'system_prompt', 'user_template', 'direct_prompt', 'temperature', 'top_p',
                           'max_tokens', 'minimum_score', 'candidate_k', 'result_k'}:
            raise RAGConfigurationError('Unknown mode setting')
        validate_options(options)
    if 'selection_score_margin' in profile:
        margin = profile['selection_score_margin']
        if isinstance(margin, bool) or not isinstance(margin, (float, int)) or not math.isfinite(margin) or not 0 <= margin <= 1:
            raise RAGConfigurationError('selection_score_margin must be between 0 and 1')
    return profile


def domain_status(name, *, root=None):
    """Lifecycle state is separate from prompts/settings and survives their reset."""
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', name):
        raise RAGConfigurationError('Invalid domain name')
    path = Path(root or ROOT_DIR) / 'data/domain_lifecycle' / f'{name}.json'
    if not path.exists():
        return 'enabled'
    status = read_json(path).get('status')
    if status not in {'enabled', 'disabled', 'deleted'}:
        raise RAGConfigurationError('Invalid domain lifecycle state')
    return status


def load_domain_config(domain=None, *, root=None, use_active=True, allow_inactive=False):
    root = Path(root or ROOT_DIR).resolve()
    default_path = root / 'data/domain_profile.json'
    default = validate_profile(read_json(default_path)) if default_path.exists() else None
    name = domain or (default['domain_name'] if default else None)
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise RAGConfigurationError("Invalid domain name")
    status = domain_status(name, root=root)
    if status == 'deleted' or (status == 'disabled' and not allow_inactive):
        raise RAGConfigurationError(f'Domain {name} is {status}; select another domain')
    is_default = bool(default and name == default['domain_name'])
    pointer = root / 'data/rag_generations' / name / 'current.json'
    profile_path = default_path if is_default else root / 'data/domains' / f'{name}.json'
    if profile_path.exists():
        profile = validate_profile(read_json(profile_path))
    elif use_active and pointer.exists():
        # Allows a newly published snapshot to be resolved before registry metadata is written.
        active = read_json(pointer).get('generation', '')
        if not isinstance(active, str) or not re.fullmatch(r'[a-f0-9]{32}', active):
            raise RAGConfigurationError('Invalid active generation')
        profile = validate_profile(read_json(pointer.parent / active / 'profile.json'))
    else:
        raise RAGConfigurationError(f'No profile for domain {name}')
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
    runtime_path = root / 'data/domain_settings' / f'{name}.json'
    if use_active and runtime_path.exists():
        settings = validate_runtime(read_json(runtime_path), profile)
        profile = validate_profile({**profile, **settings})
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


def list_domain_configs(*, root=None, include_inactive=False):
    root = Path(root or ROOT_DIR)
    names = set()
    default = root / 'data/domain_profile.json'
    if default.exists():
        names.add(validate_profile(read_json(default))['domain_name'])
    names.update(p.stem for p in (root / 'data/domains').glob('*.json'))
    names.update(p.parent.name for p in (root / 'data/rag_generations').glob('*/current.json'))
    result = []
    for name in sorted(names):
        try:
            result.append(load_domain_config(name, root=root, allow_inactive=include_inactive))
        except RAGConfigurationError:
            continue
    return result


def llm_url():
    return os.getenv("LLM_SERVICE_URLS", os.getenv("LLM_SERVICE_URL", "http://127.0.0.1:8002")).split(",")[0].strip().rstrip("/")
