# Setup guide

## Environment

The current local configuration targets Windows and Python 3.12. Qwen uses Vulkan; embedding and reranking use DirectML. Model downloads and package installation require network access. Local inference uses the downloaded files.

Follow the installation and `.env` steps in the [README](../README.md). Keep a working existing environment rather than reinstalling it unnecessarily. To record that environment’s installed package versions:

```bat
python -m pip freeze > requirements-review-lock.txt
```

The shared model service listens on port 8001 by default. Backend and admin use 8000 and 8501. Stop old project processes before starting another instance.

## GPU retrieval models

Skip preparation if `models/rag_directml/` already contains artifacts validated for the installed ONNX Runtime.

For a new setup, activate `env` and export before starting the services:

```bat
python -m rag.prepare_gpu --export
```

Start the normal launcher:

```bat
start_all_services.bat
```

Wait for Qwen to be ready. In another terminal, activate the same environment and validate:

```bat
env\Scripts\activate.bat
python -m rag.prepare_gpu --validate
```

Complete validation before ingestion or dataset chat. The current retrieval runtime has CPU model fallback disabled. A responding `/v1/rag/status` endpoint alone does not prove both retrieval models can load; validate them and try a dataset question.

Model files require more runtime memory than their download sizes suggest. If reranker loading fails with a memory allocation error, inspect available RAM/GPU memory, close unnecessary applications and check for duplicate project processes.

## Add data

Existing ingested datasets can be used directly. On a new machine, obtain the permitted source files and profiles, or the corresponding databases and vector generations.

1. Open <http://localhost:8501> and go to dataset management.
2. Upload `.xlsx`, `.xlsm`, `.csv`, `.tsv`, `.json` or `.jsonl` data.
3. Select an existing domain or create one. Review field mappings, the primary entity, embedding fields, excluded fields and source-ID configuration.
4. Run ingestion and inspect its progress and active generation.
5. Ask a question with a known answer and inspect the returned source.

JSON data must be flat row objects: an array for JSON or one object per line for JSONL. Reingesting an existing domain replaces its active dataset after successful publication. Retain generations and active pointers for rollback.

For a reviewed JSON profile, the CLI supports validation before ingestion. Replace these example paths and domain ID:

```bat
python -m rag.ingest_pipeline --domain YOUR_DOMAIN --input "path\dataset.xlsx" --profile-file "path\profile.json" --dry-run
python -m rag.ingest_pipeline --domain YOUR_DOMAIN --input "path\dataset.xlsx" --profile-file "path\profile.json"
```

## Optional configuration

| Setting | Purpose |
|---|---|
| `LLM_PORT` | Model-service listening port; default `8001` |
| `LLM_SERVICE_URLS` | Model-service URL used by clients; keep it aligned with the listener |
| `RAG_CONTEXT_MODE` | Defaults to `full`; `selected` is also supported |
| `RAG_CANDIDATE_K` / `RAG_RESULT_K` | Retrieval candidate/result settings, subject to domain configuration |
| `GEMINI_API_KEY` | Gemini-assisted profile proposals and applicable optional integrations |
| `DOMAIN_PROFILE_MODEL` | Profile-proposal model; source default `gemini-2.5-flash` |
| `TOKENIZER_NAME` | Prompt-formatting tokenizer; source default `Qwen/Qwen3-0.6B` |

The tokenizer identifier is separate from the Qwen3.5-4B generation model. Validate template compatibility when changing models or tokenizer configuration. Both root `.env` and `backend/.env` are read by the launcher/model service; avoid conflicting settings.

External search additionally requires MCP configuration. Keep keys private and out of Git. Local dataset inference does not require enabling external search.

## Files and diagnostics

| Location | Contents |
|---|---|
| `backend/` | API, browser interface, admin and conversation handling |
| `llm_service/` | Shared model service |
| `rag/` | Profiles, ingestion, retrieval and runtime helpers |
| `data/` | Dataset profiles, generations and active pointers |
| `models/` | GGUF and DirectML artifacts |
| `testing/` / `tests/` | Live regression runner / developer checks |
| `logs/services/<run>/` | Backend, LLM and admin output |
| `logs/timing/` / `logs/conversations/` | Request diagnostics and conversation records |

Model weights are excluded from Git. Source exports may also omit databases, indexes and the test assets. Keep the actual test launcher and fixtures in the development repository, and document how permitted runtime data is obtained.

The browser login is a demonstration identifier flow. It does not provide production authentication. Administrative/admissions answers reflect the loaded dataset; the application does not independently verify whether those policies are current.

## Applying the complete source bundle

Merge the source into the existing project. This bundle intentionally omits runtime assets and active dataset metadata: preserve `data/`, `models/`, `env/`, `.env`, `backend/.env`, the configured conversation database (default `legal_llm.db`), `backend/static/forms/`, raw source files and the existing Git repository. It does not contain a replacement for those files.

For a fresh administrative-domain setup, `config_examples/administrative_domain_profile.json` contains the supplied schema mapping. It is an example, not an active generation or an ingested dataset. Review/use it during ingestion; do not overwrite an existing domain's configuration with it.

Answer guidance and full context are enabled by default in the selected code. If a previous local ablation explicitly set `RAG_ANSWER_GUIDANCE=0`, remove that override or set it to `1` before the release run. Existing dataset profiles and environment overrides remain effective.
