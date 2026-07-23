# Bring Your Own Detector

Turnkey is the shortest path from “I have a detector idea” to “I can run it, inspect what happened, and audit the result.”

<div class="hero-grid" markdown>

<div class="hero-card" markdown>
## Run

Start with a CPU-only fixture and dummy backend. No GPU, token, or model download is needed.
</div>

<div class="hero-card" markdown>
## Build

Load an external `path.py:build` component without modifying Turnkey's registry.
</div>

<div class="hero-card" markdown>
## Debug

Trace missed cases through policies, target calls, typed providers, cache hits, and measured cost.
</div>

</div>

## First success

```bash
uv sync --extra dev
uv run turnkey run --config configs/runs/smoke.yaml
uv run turnkey audit outputs/<run-id>
```

The run produces a four-file public bundle: configuration and identity, paired cases, runtime events, and recomputable metrics. Plaintext prompts and responses are redacted unless a private sidecar is explicitly requested.

[Start the quickstart](get-started/quickstart.md){ .md-button .md-button--primary }
[Open the notebooks](notebooks/00_quickstart.ipynb){ .md-button }

## Choose a path

- New to Turnkey: follow [Installation](get-started/installation.md) and the [ten-minute quickstart](get-started/quickstart.md).
- Bringing a method: [build an external detector](guides/build-a-detector.md).
- Investigating a failure: [debug a run](guides/debug-a-run.md).
- Adding model-side state: learn [Policies and typed signals](concepts/runtime.md).
