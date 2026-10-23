# Retrieval update — candidate for live evaluation

Based on `Legal_LLM_code_review_20261022_130717_285597.zip`.

## Behavior changes

- **Clarification:** save one unresolved question in the existing conversation checklist. The existing router call decides whether a reply fills its missing subject or starts a new question. Only clarification turns use the two-field JSON contract (`route`, `resume_pending`); ordinary routing retains its one-field contract. Successful resumption, reset, a new question or no-match consumes the pending question. Social replies preserve it. Oversize questions are not truncated and reused. Explicit manual-domain and detect-once locks retain their existing routing precedence.
- **Low-score recovery:** when the merged reranker pool would otherwise be completely discarded, offer at most three candidates to the existing semantic record selector. It can reject all of them. The ordinary high-score path and pinned follow-up cutoff remain unchanged. A configured minimum score is now a candidate filter with bounded recovery on this path, rather than an absolute prohibition. The answer service no longer applies a second cutoff to selector-approved evidence. Raw relevance scores are retained, not promoted into confidence scores. False positive selection remains a live-test risk.
- **Unaccented input:** preserve the original ASCII question as a discovery query when contextual resolution differs, and tell query preparation to avoid uncertain accent restoration. Existing normalized routing/selector hints remain. No question-specific accent dictionary is added. This is not a demonstrated solution to Vietnamese robustness.
- **Answer consistency:** add non-exhaustive hints from public schema labels/aliases and guidance about conditions, units, temporal meaning and contradictory caveats to the existing generation call. All public fields remain available. No second answer-review call or broad deterministic renderer is introduced. This is a generation-prompt change, not a guarantee that contradictions are eliminated; inspect actual answers.
- **Cacheless QA:** `bypass_answer_cache` skips generated-answer cache reads and writes per request. It does not clear or alter other users' cache entries. Model/index caches remain enabled. QA verifies backend acknowledgement, records timings and bundles both exports.

## Live test

1. Copy the update files into the existing project; keep data, models, `.env`, indexes and existing testing files.
2. Restart using the normal service launcher. No ingestion or environment changes are required.
3. Run `testing\run_cacheless_tests.bat`.
4. Send the resulting `qa_exports\cacheless_comparison_*.zip`.

The default runs the unchanged original 64 turns and fresh 48 turns, all with answer caching bypassed. Source fixtures are test oracles only; the runner never ingests them. Required domains must already be available.

An additional 17-turn engineering probe is available after that comparison:

```bat
testing\run_cacheless_tests.bat --suite probe
```

These questions use known datasets and source facts. They are not a new-dataset evaluation or independent paper benchmark. Keep their first result before tuning further.

## Acceptance and reversibility

Primary frozen baseline: fresh suite 44/48 automatic passes, zero recorded answer-cache hits, 14.0111 seconds average excluding its first request. Manual caveat problems in passing answers and the ambiguous admissions oracle remain explicitly recorded. No frozen expected answers were revised. The original suite was 59/64 with 23 cache hits; its uncached subset averaged 13.9312 seconds and is not a fully matched cacheless baseline.

Review correctness, wrong records, abstention and contradictions alongside mean, median and P95 latency. A slowdown over about 10% on a matched workload triggers review, not automatic acceptance. Answering a previously refused question can legitimately take longer; a faster wrong refusal is not an improvement.

Four optional environment switches let us isolate changes without editing code. They default to `1`; normal use needs none of them:

| Switch | Set `0` to disable |
| --- | --- |
| `RAG_RESUME_CLARIFICATION` | Pending-question resumption |
| `RAG_LOW_SCORE_RECOVERY` | Candidate recovery below the cutoff |
| `RAG_ANSWER_GUIDANCE` | New answer hints/instructions |
| `RAG_ASCII_QUERY_GUARD` | Additional original-query and preparation guard |

Restart the backend after changing switches. The QA export records switches observed in the answering backend. Cache keys include the switch configuration. Disable all four for a behavioral ablation with the new cacheless harness; this is not a byte-identical restoration of the original source. Restore `rollback_before_update.zip` for the original modified files.

## Local validation

See `LOCAL_VALIDATION.txt`. Checks use actual routing/orchestration, temporary SQLite records and mocked external model/network I/O. They establish code behavior, not live model accuracy or Windows GPU latency. The full dependency-heavy suite and live model are unavailable in this workspace.
