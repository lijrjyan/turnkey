# Installation

## Audience

This guide assumes basic Python and terminal familiarity. You do not need prior jailbreak-evaluation experience, a GPU, or an API key for the first run.

## From source

```bash
git clone https://github.com/lijrjyan/turnkey.git
cd turnkey
uv sync --extra dev
uv run turnkey --help
```

Python 3.10 or newer is required. `uv` is recommended because the repository commits `uv.lock`; `pip install -e '.[dev]'` is supported when `uv` is unavailable.

## Optional environments

Install heavy model and judge dependencies only when a guide needs them:

```bash
uv sync --extra dev --extra hf --extra judges
```

The `hf` extra includes PyTorch, Transformers, Datasets, Pillow, and PEFT. The zero-download quickstart intentionally avoids them.

## Verify the checkout

```bash
uv run ruff check .
uv run pytest -q
```

Next: [run the ten-minute quickstart](quickstart.md).
