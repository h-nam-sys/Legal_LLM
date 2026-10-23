# Development and evaluation

The project combines Vietnamese administrative lookup with configurable retrieval over other structured datasets. Backend, local inference, ingestion and dataset administration are developed in the original repository; contribution evidence should link to actual commits and pull requests.

## Selected source snapshot — 22 October 2026

This snapshot keeps full public-record context, clarification memory, bounded low-score recovery and answer guidance enabled by default. It restores `rag/query_pipeline.py` exactly from the build used in the verified 112-turn cacheless run. The subsequent selector experiment that included unwanted branches is not selected for this release. Corrected unit checks and diagnostic on/off tools remain available.

The combined source snapshot has passed offline checks. It has not yet completed its own final live regression or whole-system acceptance pass. See [release checks](RELEASE_CHECKLIST.md).

## Recorded baseline

Source: `cacheless_comparison_20261022T071428_097130Z.zip` and its nested original/fresh exports. Both runs completed without request or stream errors, with answer-cache bypass verified.

| Measurement | Original suite | Fresh suite |
|---|---:|---:|
| Turns | 64 | 48 |
| Official automatic passes | 62 | 44 |
| Same answers with equivalent-unit checker | 62 | 46 |
| Mean response excluding first request | 13.35 s | 14.31 s |
| Answer-cache hits | 0 | 0 |

The two extra fresh passes come solely from accepting valid unit phrasing. They are not new generated answers or a model-quality improvement. Automatic counts are not a manual factual-accuracy grade. The admissions question has an ambiguous record expectation; some passing answers contain misleading caveats. Unaccented library routing and membership selection remain known limitations.

The later 65-turn paired experiment supported keeping answer guidance: paired averages were 13.18 s on and 13.09 s off, with cleaner answers in several cases. Both settings passed 28/32 automatic paired checks. That experiment also exposed extra-branch selector results, motivating restoration of the earlier selector. This is one run, not statistical proof or a new-dataset benchmark.

## Test tools

- `testing\run_cacheless_tests.bat`: unchanged original 64 + fresh 48 questions, answer-cache reads and writes bypassed; saves both response logs and comparison metadata.
- `testing\run_comprehensive_tests.bat`: retained original combined 64-turn launcher, including its read-only fixture preflight.
- `testing\run_cacheless_tests.bat --suite probe`: 17 additional source-derived wording/state probes.
- `testing\run_cacheless_tests.bat --suite targeted`: retained 65-turn guidance comparison for later diagnostics.

Fixture files are test references, not runtime ingestion inputs. The runners do not ingest them. Historical `full_context_check.py` verifies an older isolated experiment's exact hashes; its optional `--verify-full-context` mode is not the release acceptance check.

Developer checks use temporary SQLite and mocked model/network I/O. They do not establish live model accuracy, concurrent-user capacity or Windows GPU latency. Production runtime dependencies and user model/index assets are needed for live acceptance.

## Research and contribution evidence

Engineering regression questions are separate from the paper's robustness evaluation. Preserve first-run results, question-set provenance and tested code hashes. Do not report a tuned regression set as an independent research benchmark.

Development evidence: [commits](https://github.com/h-nam-sys/Legal_LLM/commits), [branches](https://github.com/h-nam-sys/Legal_LLM/branches), [pull requests](https://github.com/h-nam-sys/Legal_LLM/pulls). Add actual contribution links to the team table rather than inferring individual work from this source archive. Keep the existing repository history when applying this snapshot.
