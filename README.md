<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/brand/logo-on-dark.svg">
    <img src="docs/assets/brand/logo-on-light.svg" alt="Turnkey" width="420">
  </picture>

  <p><strong>Bring Your Own Detector.</strong></p>
  <p>A development and debugging harness for jailbreak detectors, guards, and model-side safety policies.</p>

  <p>
    <a href="https://github.com/lijrjyan/turnkey/stargazers"><img src="https://img.shields.io/github/stars/lijrjyan/turnkey?style=for-the-badge&logo=github&label=stars" alt="GitHub stars"></a>
    <a href="https://github.com/lijrjyan/turnkey/blob/main/LICENSE"><img src="https://img.shields.io/github/license/lijrjyan/turnkey?style=for-the-badge" alt="license"></a>
    <a href="https://pypi.org/project/turnkey/"><img src="https://img.shields.io/pypi/v/turnkey?style=for-the-badge&logo=pypi&label=PyPI" alt="PyPI version"></a>
    <a href="https://github.com/lijrjyan/turnkey/issues"><img src="https://img.shields.io/github/issues-raw/lijrjyan/turnkey?style=for-the-badge&label=open%20issues" alt="open issues"></a>
    <a href="https://deepwiki.com/lijrjyan/turnkey"><img src="https://img.shields.io/badge/Ask-DeepWiki-087fca?style=for-the-badge" alt="Ask DeepWiki"></a>
  </p>
</div>

---

<p align="center">
  <a href="#quick-start"><b>Quick Start</b></a> |
  <a href="https://lijrjyan.github.io/turnkey-site/"><b>Documentation</b></a> |
  <a href="https://lijrjyan.github.io/turnkey-site/learn/"><b>Notebooks</b></a> |
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

### Core Features

- **Bring your own detector:** scaffold with `turnkey init`, then load
  `path.py:build` without forking Turnkey or editing a central registry.
- **Compare paired paths:** evaluate the reference and intervention paths on
  the same selected and attacked input.
- **Ask for model state:** typed providers expose prompt logprobs, hidden
  states, gradients, and other method-owned signals with cache and lifecycle
  accounting.
- **Debug the failure:** connect a case to its policy, target, provider,
  timing, cache, and measured-forward events in the CLI or a standalone HTML
  report.
- **Audit the claim:** recompute metrics and verify redaction, source identity,
  event relationships, and public artifacts instead of trusting a summary.
- **Start without infrastructure:** the dummy backend and fixture dataset make
  the complete loop runnable without GPU access.

```text
dataset -> attack -> target -> judge -> detector policy -> metrics -> audit
```

## Quick Start

Install the packaged release from [PyPI](https://pypi.org/project/turnkey/)
with `pip install turnkey`, or install from source for development.

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
uv run turnkey report outputs/<run-id> --html report.html
```

An empty JSON list from `turnkey audit` means the bundle passed the current
artifact checks. It is not a claim that a detector is scientifically validated
or production-ready.

Start an external detector without copying boilerplate by hand:

```bash
uv run turnkey init my-detector
cd my-detector
uv run turnkey dev ./detector.py:build --max-samples 4
```

### Docker

Run the same CPU smoke with nothing installed but Docker:

```bash
docker build -t turnkey https://github.com/lijrjyan/turnkey.git
docker run --rm -v "$PWD/outputs:/app/outputs" turnkey
```

The default command runs `configs/runs/smoke.yaml` and writes the four core
artifacts under `outputs/`. Any CLI invocation works the same way, for example
`docker run --rm turnkey detector list`. For GPU-backed HF models, build the
large image with `--target full` and run with `--gpus all`.

## Included Method Examples

Turnkey includes compact integrations that exercise different runtime
capabilities:

| Method | Signal or control pattern |
| --- | --- |
| `allow_all` | No-op baseline for pipeline and metric sanity |
| `keyword` | Local prompt rule for the first detector and CI smoke |
| `smoothllm` | Perturbation and repeated target calls |
| `jailguard` | Mutation and divergence |
| `gradsafe` | Gradient provider |
| `perplexity` | Typed prompt-logprob filter |
| `self_exam` | Output-side deterministic self-screening |
| `llamaguard` | Separate Llama Guard input classifier |
| `rcs_toy` | CPU-safe representation proxy |
| `rcs` | Hidden-state provider and calibration |

These integrations demonstrate the harness. They are not leaderboard results
or claims of strict paper reproduction; original model, dataset, calibration,
and license conditions still apply.

## Documentation

Choose the shortest path for what you are trying to do:

| Goal | Guide |
| --- | --- |
| Install and verify the source checkout | [Installation](https://lijrjyan.github.io/turnkey-site/start/install/) |
| Complete the first run-to-audit loop | [Ten-minute quickstart](https://lijrjyan.github.io/turnkey-site/learn/quickstart/) |
| Add a detector without changing Turnkey | [Build an external detector](https://lijrjyan.github.io/turnkey-site/learn/build-your-own-detector/) |
| Trace a missed or overblocked case | [Debug a run](https://lijrjyan.github.io/turnkey-site/learn/debug-a-missed-case/) |
| Understand policies and model signals | [Runtime concepts](https://lijrjyan.github.io/turnkey-site/concepts/system-model/) |
| Compare model-based safety filters | [Model-based filters](https://lijrjyan.github.io/turnkey-site/concepts/detector-signals/) |
| Learn the public artifact contract | [Artifact concepts](https://lijrjyan.github.io/turnkey-site/concepts/artifacts-and-audit/) |
| Use the command line precisely | [CLI reference](https://lijrjyan.github.io/turnkey-site/reference/cli/) |
| Learn interactively | [Executable notebooks](https://lijrjyan.github.io/turnkey-site/learn/) |

The complete documentation site lives at
<https://lijrjyan.github.io/turnkey-site/> and is maintained in the
[turnkey-site](https://github.com/lijrjyan/turnkey-site) repository.

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
