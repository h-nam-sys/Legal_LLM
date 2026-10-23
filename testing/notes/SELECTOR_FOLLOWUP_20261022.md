# Selector and answer-guidance follow-up

Apply after `Legal_LLM_generalization_speed_update_20261022.zip`, the build verified by the latest 112-turn cacheless export.

## Next run

Merge the patch into the existing project, restart normally, then run from the project root:

```bat
testing\run_cacheless_tests.bat --suite targeted
```

Send the resulting `qa_exports\cacheless_comparison_*.zip`. No environment edits or ingestion are required. The existing BAT is reused.

This runs 65 turns: one warmup and 32 matched turns in each guidance setting. On/off ordering alternates by scenario. The warmup is excluded from the paired comparison. Each chain gets its own conversation; questions, expectations, domain settings and other request options match across its two variants. Only the answer-guidance request option differs. The backend must acknowledge that option and cache bypass. This does not require restarting between variants.

The inner system-test ZIP includes `guidance_comparison.json`, containing paired answers, selected records, check failures, timings and missing-information language flags. Such a flag can be a correct abstention; it is not a factual error grade. Check whether selected records also match before attributing an answer difference to guidance. One paired run does not establish statistical significance.

The 17 scenario pairs cover student services, workshop services, admissions and administrative fees. They include clarification, branch changes, processing versus usage duration, unit wording, unsupported deposits, wrong-dataset rejection, ASCII input and two additional ASCII question variants. These are engineering tests on known datasets, not a blind/new-dataset research evaluation.

## Changes

1. **Selector representation:** for ASCII questions, reuse the existing bounded excerpt budget for an accent-folded evidence view. Original titles and normalized titles remain available. The question is not rewritten or accent-restored. Accented source descriptions are folded in this matching view too. After selection, answering still receives the original full public records. The default selection instructions now describe arbitrary records and requested attributes, preserving action, branch and condition constraints. Custom dataset selection prompts remain respected. A valid empty selection remains empty; there is no forced acceptance, extra retry or extra model call. Existing `RAG_ASCII_QUERY_GUARD=0` disables the folded view along with its previous query guards.
2. **Guidance comparison:** optional per-request `answer_guidance` selects the existing guidance block on/off. Omission retains the runtime default. No new answer guidance or deterministic answer renderer is introduced in this follow-up. Cache entries distinguish settings; concurrent requests do not mutate a shared guidance setting. Full-context answering stays enabled.
3. **QA unit equivalence:** the checker accepts explicit equivalent wording such as `từng thẻ`, `một bản ... là 10000 đồng`, and currency/unit notation. Units are taken from the answer, not borrowed from the question. Inferred rates are not added across negation, sentence boundaries or a conflicting explicit rate. Existing amount, record, domain and contradiction checks remain. Original text is unchanged. Each new result also retains strict-unit checks under `previous_checker`.
4. **Baseline records:** preserve the original frozen manifests and older baseline. Add the latest verified cacheless baseline, keeping its official 62/64 and 44/48 counts alongside checker-only equivalents 62/64 and 46/48. No new generation produced that two-pass increase. The admissions ambiguity and misleading caveats remain flagged. See `CHECKER_REVIEW_20261022.json`.

## After targeted review

Choose the guidance setting from factual answer review plus paired timings. Do not choose from absence-language flag counts alone. If selector behavior improves without false-positive matches, run the complete unchanged 112-turn suite with the normal cacheless BAT, then the optional 17-turn probe. Compare the same full workload with the latest verified cacheless baseline: original 13.3509s and fresh 14.3128s excluding their first requests. Investigate a matched-workload slowdown above about 10%.

Restore `rollback_before_followup.zip` into the project root to undo overwritten files. New diagnostic files may remain unused. No models, data, indexes, database, environment file or normal launcher are included or changed.
