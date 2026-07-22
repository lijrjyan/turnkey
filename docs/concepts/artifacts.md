# Artifact model

Every normal run writes four public artifacts.

| File | Responsibility |
|---|---|
| `run.json` | Resolved configuration, component/source identity, inputs, environment, and artifact hashes |
| `cases.jsonl` | Paired logical outcomes and judge results, redacted by default |
| `events.jsonl` | Policy, target, request/provider, cache, timing, and forward-count events |
| `metrics.json` | Recomputable safety, utility, grouping, and cost metrics |

`turnkey audit` treats these files as evidence, not as trusted prose. Optional reports and exports are derived views. Plaintext belongs only in an explicitly requested private sidecar and must not be published.
