Current release checks:
  run_cacheless_tests.bat
    Original 64 + fresh 48 questions; verified answer-cache bypass; combined export.
  run_comprehensive_tests.bat
    Retained 64-turn original combined launcher and read-only fixture preflight.

Optional diagnostics, not required for normal startup:
  run_cacheless_tests.bat --suite probe
  run_cacheless_tests.bat --suite targeted

Use the existing ingested datasets. Fixtures are test oracles; these launchers do not ingest them.
Send the resulting qa_exports ZIP for review. Automated PASS counts are not factual accuracy.
Historical notes under testing/notes describe earlier experiments. The selected release restores the earlier selector; see docs/DEVELOPMENT.md.
