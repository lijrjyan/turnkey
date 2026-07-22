# CLI reference

New users need four commands:

- `turnkey run --config <yaml>`: execute a paired evaluation.
- `turnkey dev <path.py:build>`: run a bounded external-component smoke.
- `turnkey inspect <run-dir>`: diagnose cases and events.
- `turnkey audit <run-dir>`: independently validate the public bundle.

Discovery commands include `turnkey data list`, `turnkey detector list`, `turnkey attack list`, `turnkey judge list`, and `turnkey backends`.

Advanced workflows use `turnkey detector calibrate`, `turnkey matrix plan/run/summarize/merge/analyze`, `turnkey analyze threshold-sweep`, and `turnkey analyze detector-overlap`.

Run `turnkey <command> --help` for current arguments; the CLI itself is the authoritative option reference.
