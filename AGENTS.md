# Turnkey contributor guide

Turnkey is a detector-development harness. Keep the public repository focused on runnable product behavior, user learning, and reproducible evidence; paper drafts, private experiment payloads, local planning ledgers, and secrets do not belong here.

All development and local commits must happen on `dev`. Run
`scripts/install-git-hooks.sh` after cloning. Direct pushes to `main` are
forbidden; `main` may advance only through an explicitly approved review/merge.

The runtime flow is:

```text
dataset -> attack -> target -> judge -> detector policy -> metrics -> audit
```

Public extension points live under `src/turnkey/components/`; runtime infrastructure lives in `runner/`, `runtime_providers/`, `audit/`, `analysis/`, and `matrix/`.

Before a local commit, run:

```bash
uv run ruff check .
uv run pytest -q
uv run make smoke
```

If docs or notebooks changed, also run:

```bash
uv sync --extra docs
uv run python scripts/ci/run_notebooks.py
uv run mkdocs build --strict
```

Use Conventional Commit messages. Never push, publish a package, change repository visibility, or deploy documentation unless the owner explicitly authorizes that external action.
