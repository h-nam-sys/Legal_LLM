# Multi-domain ingestion and admin configuration increment

This increment restores the Gemini profiling feature from `ingest_pipeline(6).py`, while retaining the current validated SQL/vector snapshots. It also connects new domains to chat and extends the existing Streamlit admin dashboard.

## Install on your Windows project

1. Stop the backend, local LLM service and existing Streamlit dashboard.
2. Extract this ZIP into a **separate folder**, for example `C:\Users\Reines\Downloads\Legal_LLM_multidomain_admin_increment`.
3. Open Command Prompt and run:

```bat
cd /d C:\Users\Reines\Downloads\Legal_LLM_multidomain_admin_increment
python install_increment.py --project "C:\Users\Reines\Documents\GitHub\Legal_LLM"
```

The installer copies complete production files and creates a timestamped backup of every existing file it replaces under `Legal_LLM\backups\multidomain_<timestamp>`. It does not replace `.env`, databases, datasets, profiles, model weights, vectors, or form files. Keep the printed backup path.

4. Install the small admin dependency supplement in your existing environment:

```bat
cd /d C:\Users\Reines\Documents\GitHub\Legal_LLM
env\Scripts\activate
python -m pip install -r requirements_admin.txt
```

5. Keep your existing `.env` settings. Gemini profiling requires a **working** `GEMINI_API_KEY` in your local root `.env` or `backend/.env`. The key you described as outdated will need replacing locally if it no longer works. Do not put the key into a dataset or profile. Optional model override:

```dotenv
DOMAIN_PROFILE_MODEL=gemini-2.5-flash
```

6. Start the LLM service in one Command Prompt:

```bat
cd /d C:\Users\Reines\Documents\GitHub\Legal_LLM
env\Scripts\activate
python llm_service/main.py
```

7. Start the backend in a second Command Prompt:

```bat
cd /d C:\Users\Reines\Documents\GitHub\Legal_LLM
env\Scripts\activate
cd backend
uvicorn main:app --port 8000 --reload
```

8. Start the admin dashboard in a third Command Prompt:

```bat
cd /d C:\Users\Reines\Documents\GitHub\Legal_LLM
env\Scripts\activate
cd backend
python -m streamlit run admin_dashboard.py --server.address 127.0.0.1 --server.port 8501
```

Alternatively, double-click `start_domain_admin.bat` after installing dependencies.

- Chat: http://localhost:8000/chat
- Admin: http://localhost:8501

The admin dashboard remains a local tool on loopback. The existing mock login and admin authorization model are unchanged; this increment does not turn them into a deployment-ready authentication system.

## First: keep the existing legal domain working

Do **not** re-ingest legal data just to install the patch. Its existing profile and active snapshot remain in place. Repeat a familiar marriage fee/location query in chat. With only one ready domain, routing goes directly to it without another classification call.

Existing legal field aliases, exact output, locality prompt, full Sources, streaming, history, and form catalog remain. The legal-specific defaults are retained for compatibility. New generic profiles configure their own aliases, prompts and prefixes.

## Add a domain from the admin UI

1. Open **Quản lý lĩnh vực / dữ liệu → Thêm dữ liệu**.
2. Select **Tạo lĩnh vực mới**.
3. Enter a distinct domain ID, such as `campus_demo`, or leave it blank for Gemini to suggest one.
4. Upload `.xlsx`, `.xlsm`, `.csv`, `.tsv`, `.json`, or `.jsonl`.
5. Choose the Excel sheet if necessary.
6. Click **Gemini: tạo cấu hình tự động**.
7. Review the generated description, schema mapping, aliases, prompts and fields. Column mappings must match actual input column names. The English keys are internal field names.
8. Choose the primary title/topic field and fields to embed. For the current legal domain, keep the existing title representation. Changing embedded fields requires re-ingestion.
9. Check the validation report, then click **Bắt đầu ingest**.
10. Open **Tiến độ / phiên bản** and refresh until the job succeeds. Full worker logs and row validation results are shown.
11. In chat, click **Làm mới lĩnh vực**. Select the new domain explicitly first and ask a dataset question.
12. Select **Tự động** to try detection and topic switching.

Existing-domain uploads reuse that domain's current profile and do not call Gemini automatically. Each upload **replaces the whole domain dataset** after successful validation/vector build; it does not append rows. Ingest is a background subprocess, so rerunning or closing the dashboard does not cancel the worker. A per-domain writer lock prevents conflicting ingestion/rollback operations.

### A ready second-domain demonstration

`examples/campus_demo.csv` is explicitly fictional student-service data. It is not a real university policy document.

Upload it through the steps above and let Gemini generate the profile. If you want to exercise ingestion while Gemini is unavailable, import the included reviewed `examples/campus_demo_profile.json` using **Hoặc nhập profile JSON đã có → Nạp profile JSON**. This optional profile is supplied as a demonstration fixture; it is not presented as an actual Gemini result.

Try:

```text
Mượn sách thư viện cần gì, ở đâu?
Cấp lại thẻ sinh viên mất bao lâu, bao nhiêu tiền?
lệ phí bao nhiêu?
```

For automatic switching in one conversation:

```text
Đăng ký kết hôn mất bao nhiêu tiền?
Cấp lại thẻ sinh viên mất bao nhiêu tiền?
mất mấy ngày?
```

The first question should use your legal domain; the second and third should use `campus_demo`. The active domain is displayed above chat. The demonstration facts come from the CSV, not from the routing description.

## What admin can configure now

| Control | Effect |
| --- | --- |
| Domain coverage description and example questions | Automatic routing input |
| Primary entity, column mappings, embedding fields/prefix | Dataset ingestion; changes require rebuilding |
| Excluded fields | Controls public evidence and Sources after ingestion |
| Field aliases, title prefixes, exact-value output | Requested-field matching and direct formatting |
| Default Strict/Friendly mode | Domain default unless the chat request explicitly chooses a mode |
| Evidence-answer prompt and input template, separately per mode | Generated answers with retrieved records |
| Friendly direct-conversation prompt | Friendly answers without retrieval |
| Temperature, top-p, max output tokens per mode | Local LLM generation |
| Minimum score, candidate count, result count per mode | Retrieval acceptance and record limits |
| Query preparation, field-selection and record-selection prompts | Advanced runtime JSON |
| Missing-evidence message | Domain fallback |
| Snapshot activation | Rollback to a completed SQL/vector pair |

Use **Cấu hình → Áp dụng cấu hình** for the ordinary controls. Advanced settings are available as editable runtime JSON. Invalid settings are rejected before publication. Runtime edits apply at the next chat request and invalidate answer caches; they do not rebuild vectors or require a backend restart. A request already running continues with its loaded configuration.

`{context}` and `{query}` are the only supported answer-template placeholders. Use literal braces as `{{` and `}}` when needed.

When exact-value output is enabled and the question explicitly asks a recognized field, the answer is formatted from retrieved values. Generation temperature/prompt changes do not rewrite those values. Other questions still use the LLM.

Retrieval continues to use BGE-M3 plus the existing reranker. Reranker/model replacement and automatic threshold calibration are not part of this increment. Result counts remain capped by configured limits; set result count to at least eight if you intend to retain eight distinct requested records.

## Gemini profiling: what is sent

Gemini receives column names and a bounded, distributed sample of at most eight rows, with individual values truncated and a total value budget. It proposes the mapping, short coverage description, examples and aliases. The description is a draft inferred from that sample, **not a comprehensive summary of the entire dataset**. Admin can edit it.

The API is invoked only by the profiling action or onboarding a new domain via CLI. Normal retrieval still uses the local embedding/reranking models and local LLM service. New profiles are validated against actual dataset columns and rows. Invalid AI output stops onboarding; there is no silent hardcoded legal fallback.

Original record values, not the profile description, provide the answers. This uses Gemini's JSON response mode through its documented REST endpoint: https://ai.google.dev/api/generate-content.

## Routing behavior

- Explicit domain selection wins and remains selected for that conversation.
- `Tự động` allows topic switching.
- A single ready domain requires no routing classifier call.
- Recognizable field-only follow-ups reuse the current domain and active records.
- Other automatic-mode questions are classified against ready domain descriptions, entity labels, fields and examples using the existing local LLM endpoint.
- An ambiguous/invalid/unavailable classifier asks for explicit domain selection rather than guessing.
- A domain switch clears the previous active procedure context. Friendly model history is limited to the current domain segment, and the previous domain's cached evidence cannot be reused under another domain key.

Qwen 1.7B still makes the automatic routing decision. Explicit selection is deterministic; live automatic routing quality requires checking against your real datasets. A retrieval miss alone does not force a domain switch.

## CLI alternative

New domain with automatic Gemini profiling:

```bat
cd /d C:\Users\Reines\Documents\GitHub\Legal_LLM
env\Scripts\activate
set PYTHONIOENCODING=utf-8
set RAG_EMBED_DEVICE=cpu
python -m rag.ingest_pipeline --file examples/campus_demo.csv --domain campus_demo --auto-profile --save-draft data/campus_demo_draft.json
```

That command proposes/validates a profile and saves the draft; it does not activate data. Then ingest the reviewed draft:

```bat
python -m rag.ingest_pipeline --file examples/campus_demo.csv --domain campus_demo --profile-file data/campus_demo_draft.json
```

Or directly create and ingest a new domain:

```bat
python -m rag.ingest_pipeline --file examples/campus_demo.csv --domain campus_demo --auto-profile
```

Re-ingest an existing domain while reusing its profile:

```bat
python -m rag.ingest_pipeline --domain campus_demo --input examples/campus_demo.csv
```

`--input` and `--file` are aliases. Existing default commands continue to work. For a new domain, `--dry-run` can still call Gemini if no profile exists; it does not load embedding models or publish records. PDF, Word, databases and API ingestion are not implemented by this increment. Nested JSON must first be flattened into tabular row fields.

## Storage and compatibility

- Completed snapshots: `data/rag_generations/<domain>/<generation>/`
- Active snapshot pointer: `data/rag_generations/<domain>/current.json`
- Domain registry profiles: `data/domains/<domain>.json`
- The existing `data/domain_profile.json` still identifies the default domain.
- Runtime settings: `data/domain_settings/<domain>.json`
- Admin uploads and durable jobs: `data/admin_uploads/`, `data/admin_jobs/`

SQL and vectors are built before activation; a failed vector build never publishes a partial SQL/vector pair. Original rows are retained in `_raw_json`, and excluded/raw metadata is not added to public Sources. Row IDs remain matched across SQL and vectors.

No chat database migration is needed. Domain selection is stored under `Conversation.checklist._domain_state`; internal `RAG_DOMAIN:` source markers identify message-domain boundaries. The UI hides those internal markers. Existing untagged local answers belong to the original default domain.

The MCP increment replaces the legacy Gemini-only fallback with a real MCP search client/server and evidence-based local generation. See README_MCP.md for enabling domains and configuring source restrictions. The existing legal form catalog is preserved; generic domains disable legal form offers by default.

## Validation and practical limits

Automated checks exercise:

- Gemini HTTP response parsing, bounded input and rejection of invalid mappings.
- Real SQLite ingestion and record ID correspondence with fake vector persistence.
- Separate domain snapshots, failure before activation and rollback.
- Durable job success/failure and per-domain writer locking.
- Per-mode settings and cache revision changes without rebuilding.
- Explicit/sticky/automatic routing contracts and ambiguity handling.
- Real FastAPI endpoints, SSE completion, Sources and conversation persistence using controlled model/retrieval responses.
- Streamlit profile-review/ingestion controls, settings publication, version controls and preservation of the original monitoring tabs.
- Cross-domain follow-up and Friendly-history isolation.

These checks do **not** run your BGE/reranker/GGUF models or call a paid/live Gemini endpoint. The uploaded project excluded its datasets, active profiles, vector snapshots, form files and model weights; those live checks must be done on your machine. Run the second-domain demonstration before claiming production routing accuracy.

Optional local automated checks:

```bat
python -m pip install pytest
python -m pytest tests/test_domain_increment.py -q
```

To revert code, stop the services, then copy the backed-up files from the installer backup directory back to the corresponding project paths. Code rollback does not change data snapshots. Use admin snapshot activation for a data rollback.
