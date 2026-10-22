Start services normally with start_all_services.bat, then run run_comprehensive_tests.bat here.
Default: the existing 64 chat turns (41 original + 23 novel-domain turns).
Prompts, answers, checks and diagnostics are saved in the project qa_exports ZIP.
Send that ZIP for review. This port update does not fix previous live suite failures.

Optional commands from the project root (one batch file supports every suite):
testing/run_comprehensive_tests.bat --suite original
testing/run_comprehensive_tests.bat --suite novel
testing/run_comprehensive_tests.bat --suite smoke
testing/run_comprehensive_tests.bat --suite review

fixtures/xuong_nho contains expected source data, profile and questions for maker_lab_test.
These are test references, not a second runtime database. Tests use the already-ingested data.
Before all/novel runs, a read-only preflight checks the active profile and records match.
It never imports, updates or deletes data. If preflight stops, send its message for review.
Keep the fixtures to reproduce this test; keep the actual data/ directory and snapshots.
This suite covers chat regression, not every UI, ingestion, MCP or concurrency feature.
