# Contributing to Turnkey

Thank you for helping make detector development easier to run, inspect, and
audit.

Participation is governed by [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). Project
decision authority is summarized in [GOVERNANCE.md](GOVERNANCE.md).

## Before you start

Use GitHub Issues for confirmed bugs, documentation gaps, or a focused proposal
before investing in a large change. Security-sensitive reports follow
[SECURITY.md](SECURITY.md), not public issues.

Turnkey owns the Python runtime, component contracts, artifact schemas, tests,
canonical notebooks, and concise API documentation. Branded website content and
the ordered learning book belong in the separate
[`turnkey-site`](https://github.com/lijrjyan/turnkey-site) repository.

## Development workflow

1. Create work from `dev`; do not develop directly on stable `main`.
2. Install the development environment:

   ```bash
   uv sync --extra dev
   ```

3. Keep changes focused and add tests for behavior changes.
4. Run the verification appropriate to the change.

The minimum repository gate is:

```bash
uv run ruff check .
uv run pytest -q
uv run make smoke
```

When documentation or notebooks change, also run:

```bash
uv sync --extra docs
uv run python scripts/ci/run_notebooks.py
uv run mkdocs build --strict
```

Optional GPU, Hugging Face, and remote-judge paths must state their environment,
model, dataset, and credential requirements. A fake or CPU-safe smoke is useful
evidence, but must not be presented as proof of a real service or GPU path.

## Change boundaries

- Preserve the public extension boundary: external components load through
  documented entrypoints and should not require editing a central registry.
- Treat `run.json`, `cases.jsonl`, `events.jsonl`, and `metrics.json` as a public
  contract; schema changes need tests, migration notes, and a changelog entry.
- Distinguish a runnable integration from strict paper reproduction and from a
  production-readiness claim.
- Keep generated outputs, model caches, private prompts, internal datasets,
  credentials, and machine-specific state out of commits.
- Retain third-party attribution and license conditions when adding a method,
  model, dataset, or source-derived implementation.

## Pull requests

A reviewable pull request explains the user-visible outcome, lists exact
verification commands, calls out compatibility or research-fidelity limits,
and links its issue when one exists. Use Conventional Commit-style titles such
as `feat:`, `fix:`, `docs:`, `test:`, or `chore:`.

Maintainers may ask to split runtime, documentation, and mechanical refactoring
into separate changes when that makes the evidence easier to review.
