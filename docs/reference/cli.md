# CLI reference

Start a new external detector with:

- `turnkey init <new-directory>`: create `detector.py`, `run.yaml`, and a
  focused README without overwriting an existing path.

The daily development loop uses three commands:

- `turnkey dev <path.py:build>`: run a bounded external-component smoke.
- `turnkey run --config <yaml>`: execute a paired evaluation.
- `turnkey inspect <run-dir>`: diagnose cases and events.

Verification and sharing use:

- `turnkey audit <run-dir>`: independently validate the public bundle.
- `turnkey report <run-dir> --html <path>`: generate a standalone redacted
  report with expandable case timelines.

Discovery commands include `turnkey data list`, `turnkey detector list`, `turnkey attack list`, `turnkey judge list`, and `turnkey backends`.

Advanced workflows use `turnkey detector calibrate`, `turnkey matrix plan/run/summarize/merge/analyze`, `turnkey analyze threshold-sweep`, and `turnkey analyze detector-overlap`.

Run `turnkey <command> --help` for current arguments; the CLI itself is the authoritative option reference.
