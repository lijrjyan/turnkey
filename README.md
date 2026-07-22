<div align="center">
  <img src="docs/assets/turnkey-mark.svg" alt="Turnkey logo" width="112">

  <h1>Turnkey</h1>

  <p><strong>Bring Your Own Detector.</strong></p>
  <p>A development and debugging harness for jailbreak detectors, guards, and model-side safety policies.</p>

  <p>
    <a href="https://github.com/lijrjyan/turnkey/stargazers"><img src="https://img.shields.io/github/stars/lijrjyan/turnkey?style=for-the-badge&logo=github&label=stars" alt="GitHub stars"></a>
    <a href="https://github.com/lijrjyan/turnkey/blob/main/LICENSE"><img src="https://img.shields.io/github/license/lijrjyan/turnkey?style=for-the-badge" alt="license"></a>
    <a href="https://github.com/lijrjyan/turnkey/issues"><img src="https://img.shields.io/github/issues-closed-raw/lijrjyan/turnkey?style=for-the-badge&label=closed%20issues" alt="closed issues"></a>
    <a href="https://github.com/lijrjyan/turnkey/issues"><img src="https://img.shields.io/github/issues-raw/lijrjyan/turnkey?style=for-the-badge&label=open%20issues" alt="open issues"></a>
    <a href="https://deepwiki.com/lijrjyan/turnkey"><img src="https://img.shields.io/badge/Ask-DeepWiki-087fca?style=for-the-badge" alt="Ask DeepWiki"></a>
  </p>
</div>

---

<p align="center">
  <a href="#quick-start"><b>Quick Start</b></a> |
  <a href="docs/get-started/installation.md"><b>Documentation</b></a> |
  <a href="docs/notebooks/"><b>Notebooks</b></a> |
  <a href="examples/"><b>Examples</b></a> |
  <a href="https://deepwiki.com/lijrjyan/turnkey"><b>Ask DeepWiki</b></a> |
  <a href="#community--support"><b>Community</b></a>
</p>

## About

Turnkey shortens the path from “I have a detector idea” to “I can run it,
inspect what happened, and audit the result.” It gives detector authors one
runtime for paired evaluation, typed model signals, inspectable event traces,
and reproducible audit artifacts.

The first run is deliberately small: it runs on CPU, downloads no model, and
requires no API key. GPU-backed methods and remote judges are optional layers,
not prerequisites for understanding the system.

## Why Turnkey

- **Bring your own detector:** load `path.py:build` without forking Turnkey or
  editing a central registry.
- **Compare paired paths:** evaluate the reference and intervention paths on
  the same selected and attacked input.
- **Ask for model state:** typed providers expose prompt logprobs, hidden
  states, gradients, and other method-owned signals with cache and lifecycle
  accounting.
- **Debug the failure:** connect a case to its policy, target, provider,
  timing, cache, and measured-forward events.
- **Audit the claim:** recompute metrics and verify redaction, source identity,
  event relationships, and public artifacts instead of trusting a summary.
- **Start without infrastructure:** the dummy backend and fixture dataset make
  the complete loop runnable without GPU access.

```text
dataset -> attack -> target -> judge -> detector policy -> metrics -> audit
```

## Quick Start

The supported preview is currently installed from source. The `turnkey` name
on [PyPI](https://pypi.org/project/turnkey/) is reserved at `0.0.1`; it is not
yet the functional `0.1.0` source preview documented here.

```bash
git clone https://github.com/lijrjyan/turnkey.git
cd turnkey
uv sync --extra dev
uv run turnkey run --config configs/runs/smoke.yaml
```

The run prints an output directory containing four public artifacts:

| Artifact | Purpose |
| --- | --- |
| `run.json` | Resolved configuration, source identity, and environment |
| `cases.jsonl` | Redacted paired reference/intervention outcomes |
| `events.jsonl` | Target, policy, provider, cache, and timing events |
| `metrics.json` | Safety, utility, and measured runtime cost |

Inspect one failure category and independently audit the bundle:

```bash
uv run turnkey inspect outputs/<run-id> --category missed_harm
uv run turnkey audit outputs/<run-id>
```

An empty JSON list from `turnkey audit` means the bundle passed the current
artifact checks. It is not a claim that a detector is scientifically validated
or production-ready.

## Included Method Examples

Turnkey includes compact integrations that exercise different runtime
capabilities:

| Method | Signal or control pattern |
| --- | --- |
| `allow_all` | No-op baseline for pipeline and metric sanity |
| `keyword` | Local prompt rule for the first detector and CI smoke |
| `smoothllm_v3` | Perturbation and repeated target calls |
| `jailguard_v3` | Mutation and divergence |
| `gradsafe_v3` | Gradient provider |
| `rcs_toy_v3` | CPU-safe representation proxy |
| `rcs_paper_v3` | Hidden-state provider and calibration |

These integrations demonstrate the harness. They are not leaderboard results
or claims of strict paper reproduction; original model, dataset, calibration,
and license conditions still apply.

## Documentation

Choose the shortest path for what you are trying to do:

| Goal | Guide |
| --- | --- |
| Install and verify the source checkout | [Installation](docs/get-started/installation.md) |
| Complete the first run-to-audit loop | [Ten-minute quickstart](docs/get-started/quickstart.md) |
| Add a detector without changing Turnkey | [Build an external detector](docs/guides/build-a-detector.md) |
| Trace a missed or overblocked case | [Debug a run](docs/guides/debug-a-run.md) |
| Understand policies and model signals | [Runtime concepts](docs/concepts/runtime.md) |
| Learn the public artifact contract | [Artifact concepts](docs/concepts/artifacts.md) |
| Use the command line precisely | [CLI reference](docs/reference/cli.md) |
| Learn interactively | [Executable notebooks](docs/notebooks/) |

Build the complete local documentation site with:

```bash
uv sync --extra docs
uv run mkdocs serve
```

## Development

Development happens on `dev`; `main` is the stable branch and only receives
reviewed changes.

```bash
uv sync --extra dev
uv run ruff check .
uv run pytest -q
uv run make smoke
```

Optional model and judge dependencies are isolated from the CPU-first path:

```bash
uv sync --extra dev --extra hf --extra judges
```

## Community & Support

- Use [GitHub Issues](https://github.com/lijrjyan/turnkey/issues) for confirmed
  bugs, documentation gaps, and scoped feature proposals.
- After the public repository has been indexed, use
  [Ask DeepWiki](https://deepwiki.com/lijrjyan/turnkey) to explore the
  architecture and ask code-grounded questions.
- Never include credentials, private prompts, unpublished datasets, or personal
  data in issues, logs, examples, or generated artifacts.
- Do not open a public issue for a vulnerability or suspected data leak. Use
  GitHub's private vulnerability reporting channel when it is available.

## Project Status

Turnkey is an early source preview. Public interfaces, artifact schemas, and
method integrations can change before the first functional PyPI release. Pin a
Git commit for reproducible work and inspect the generated `run.json` before
comparing runs.

The README badge and `.devin/wiki.json` prepare DeepWiki integration, but the
free public index cannot read a private GitHub repository. Ask DeepWiki becomes
active after the formal repository is public and indexed; private indexing
requires a connected Devin account.

## License

Turnkey is licensed under the [Apache License 2.0](LICENSE). Third-party methods
and datasets retain their own terms; see
[third_party/README.md](third_party/README.md).
