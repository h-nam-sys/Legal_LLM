# Development and evaluation

## Background and progress

The project started with Vietnamese administrative-procedure lookup and expanded into a configurable assistant for multiple structured datasets.

| Area | Current status | Evidence |
|---|---|---|
| Chat, history and sources | Implemented | `[Add relevant commit / PR]` |
| Multiple datasets and routing controls | Implemented; routing robustness under refinement | `[Add relevant commit / PR]` |
| Dataset administration and versioned ingestion | Implemented | `[Add commit / PR and admin capture]` |
| Local Qwen and shared GPU retrieval models | Implemented | `[Add runtime/model configuration commit]` |
| Comprehensive chat regression | 64-turn suite available | Recorded archive below |
| Focused routing patch, 21 Oct | 123 offline checks passed; live validation pending | `[Add patch commit and subsequent live archive]` |

Add contribution links beside each area so readers can follow the work in the original repository.

## Recorded evaluation

The completed full-context run on **20 October 2026** recorded:

| Metric | Result |
|---|---:|
| Turns / scenarios | 64 / 42 |
| Automatic PASS / FAIL | 58 / 6 |
| Mean complete-response time | 10.51 s |
| Median complete-response time | 5.09 s |

Source: `system_tests_20261020T081538_831156Z.zip`, `results.json`. `[Add a link to the retained summary/archive and the tested commit SHA.]`

Manual review graded 58 responses correct/acceptable, 4 partial and 2 incorrect. One automatic room-answer failure was a checker false positive. The suite includes greetings, refusals and repeated questions, so its pass rate is not factual accuracy over 64 independent questions. It does not measure concurrent-user capacity or validate every admin, form and external-search action.

![Recorded test summary](images/test-results.svg)

<!-- Replace with the actual dated run summary and identifier. -->

## Remaining work

- Verify the renewal-follow-up and unaccented-routing patch with the live model.
- Improve answers that omit a specific service desk or add unnecessary caveats.
- Improve clarification/domain-lock explanations and refine the broad contradiction checker.
- Evaluate concurrent users and complete acceptance coverage of admin actions, form downloads and optional search.

## Reproduce and trace a result

Use the normal launcher, then run `testing\run_comprehensive_tests.bat`. Preserve its `qa_exports` ZIP with the corresponding commit SHA. The novel-dataset preflight reads existing ingestion state; it does not import data again.

Developer routing checks:

```bat
python -m unittest discover -s tests -p test_llm_domain_routing.py
python -m unittest discover -s tests -p test_routing_contract_integration.py
```

The offline checks use mocked model/external I/O. The 20 October live measurements predate the 21 October patch.

Development history: [commits](https://github.com/h-nam-sys/Legal_LLM/commits), [branches](https://github.com/h-nam-sys/Legal_LLM/branches), [pull requests](https://github.com/h-nam-sys/Legal_LLM/pulls).
