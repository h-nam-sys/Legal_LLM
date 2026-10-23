# Whole-system release check

Status: manual/live checks below are pending for this combined source snapshot. This is the next work phase, not another small prompt-tuning cycle.

- [ ] Start with the normal launcher. Chat/backend is available on 8000, the shared model service on 8001, and admin on 8501. Check the service logs for startup failures.
- [ ] Run `testing\run_cacheless_tests.bat` once against the preserved datasets. Review failed answers and sources; retain the export with this snapshot's hashes. Compare the same full suites with the recorded cacheless baseline.
- [ ] Chat: stream a known answer, expand its source, ask a follow-up, switch domains in Auto, and confirm manual selection/detect-once still stay in their chosen domain. Try a question with no source support.
- [ ] History and feedback: save an answer, submit feedback, reopen the conversation and restart the backend. Confirm the saved conversation and feedback persist.
- [ ] Dataset administration: create a disposable domain with a small permitted dataset using different field names. Review mappings/excluded fields, ingest it, inspect progress, ask known questions and verify source values. Use that disposable domain to check configuration edits, replacement ingestion and rollback.
- [ ] Forms: ask about a configured procedure with an available Word form, accept the offer and open the downloaded file. Existing `backend/static/forms` files must be preserved.
- [ ] Optional MCP: verify the search offer's decline/permission behavior and, if configured for use, an explicitly approved search. No online action is assumed tested by the chat regression suite.
- [ ] Reproducibility: follow the README/setup instructions on the intended deployment machine, confirm runtime data/model availability, and record the tested commit/snapshot. Preserve the original Git history.

Fix crashes, broken features, data loss, incorrect sources and deployment blockers discovered here. Record accepted limitations separately; wording polish and small accuracy gains can wait.
