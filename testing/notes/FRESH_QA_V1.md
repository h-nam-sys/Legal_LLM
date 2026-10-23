# Fresh system retest v1

This is an engineering test of the existing application, separate from the research-paper benchmark.

## Run

Copy the `testing/` contents from this package into the existing project. Keep the current code and model/settings for the first run. Restart with `start_all_services.bat`, then run this command from the project root:

```bat
env\Scripts\python.exe qa_suite.py --cases testing\fixtures\qa_fresh_cases_v1.json --require-all-domains
```

Keep services running and avoid other chats during the test. Send the new `qa_exports/system_tests_*.zip` afterward. The existing engine saves prompts, responses, timings and diagnostics automatically.

The original 64-turn suite and its normal batch launcher remain available. This package adds no launcher or application-code changes.

## Coverage

- 48 turns in 32 separate conversation scenarios.
- Four domains: administrative procedures, Demo university services, HCMUS admissions and the Xưởng Nhỏ fixture.
- 20 distinct target records, including five that were not expected targets in the previous suite: Trạm Nắng drill renewal, Trạm Nắng projector borrowing, Trạm Nắng drill repair and membership cards at both stations.
- New combinations of processing time, usage duration, fees, units, required documents and service desks.
- Conditional eligibility, negation, comparisons, subject/branch changes, thanks between factual turns, unaccented input, clarification and manual-domain reset.
- Zero exact or normalized prompt matches with the previous 64-turn manifest. All 48 prompts are also distinct within this set.

This set is authored from known data and previous observations. It is not an independently authored blind benchmark, a new dataset or an estimate of research-paper robustness. It does not cover positive movie/product answers, UI actions, external search or concurrency.

## Expected facts and review

`qa_fresh_reference_v1.json` contains the public source records used to author the expectations. They come from the 22 October QA export and its supplied Xưởng Nhỏ fixture. They are dataset snapshots, not an independent verification of current law or admissions policy.

Each turn in `qa_fresh_cases_v1.json` contains a `manual_oracle` describing a correct answer in context. The existing test engine preserves this in the exported manifest but does not grade it automatically.

The automatic checks assess routing, selected records, saved responses, available fields and declared answer patterns. They cannot reliably establish negation, correct attribution in comparisons, or the absence of contradictory wording. Review the answers against `manual_oracle`, including turns marked PASS. No broad caveat regex from the earlier room-answer false positive is copied into this set.

All 91 required factual patterns were checked against their intended source records. JSON structure, record references, regex validity and question uniqueness were checked offline. No live model run has been performed on this set.

The first command uses `qa_suite.py` directly, so it does not invoke the old comprehensive wrapper's novel-data preflight. Keep the existing ingested data unchanged for this run. If data has changed, compare it with the reference before interpreting failures; do not reingest merely to run a test.

## Keep the first result

Save the initial result ZIP and its code/settings alongside later results. Once this set is used to guide fixes, it becomes another regression suite; use a further fresh set when checking generalization again.

Scores on 48 new turns should not be treated as directly comparable accuracy percentages with the old 64 turns: question mix and difficulty differ. Inspect behavior by category as well as the overall count.

Manifest SHA-256:

```text
17a5a8d82147ea89f6a8e15531ff0eaa24378a40d5ed8f03c74f86c54171baa0
```
