# Turnkey

**Bring Your Own Detector.** Turnkey is a development and debugging harness for jailbreak detectors, guards, and model-side safety policies.

It gives detector authors one runtime for paired evaluation, typed model signals, inspectable event traces, and reproducible audit artifacts. The first run is CPU-only and downloads no model.

```bash
git clone https://github.com/lijrjyan/turnkey.git
cd turnkey && uv sync --extra dev
uv run turnkey run --config configs/runs/smoke.yaml
```

The command prints a run directory containing four public artifacts:

- `run.json`: resolved configuration, source identity, and environment;
- `cases.jsonl`: redacted paired reference/intervention outcomes;
- `events.jsonl`: target, policy, provider, cache, and timing events;
- `metrics.json`: safety, utility, and measured runtime cost.

Then inspect or independently audit it:

```bash
uv run turnkey inspect outputs/<run-id> --category missed_harm
uv run turnkey audit outputs/<run-id>
```

## Why Turnkey

- **Bring your own detector:** load `path.py:build` without forking Turnkey or editing a registry.
- **Ask for model state:** typed providers expose prompt logprobs, hidden states, gradients, and other method-owned signals with cache and lifecycle accounting.
- **Debug the failure:** connect a case to its policy, target, provider, timing, and measured-forward events.
- **Audit the claim:** recompute metrics and verify redaction, inputs, source identity, event relationships, and artifacts.
- **Start small:** the dummy backend and fixture dataset make the entire loop runnable without GPU access.

## Learn

- [Installation](docs/get-started/installation.md)
- [Ten-minute quickstart](docs/get-started/quickstart.md)
- [Build an external detector](docs/guides/build-a-detector.md)
- [Debug a missed case](docs/guides/debug-a-run.md)
- [Executable notebooks](docs/notebooks/)
- [Detector and Policy authoring reference](docs/reference/detector-authoring.md)

The complete learning site is built with `uv run mkdocs serve` after `uv sync --extra docs`.

## Development

```bash
uv sync --extra dev
uv run ruff check .
uv run pytest -q
uv run make smoke
```

GPU-backed methods are optional extras: `uv sync --extra dev --extra hf --extra judges`.

## License

Turnkey is licensed under the [Apache License 2.0](LICENSE). Third-party methods and datasets retain their own terms; see [third_party/README.md](third_party/README.md).
