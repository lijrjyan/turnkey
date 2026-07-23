# Turnkey — Bring Your Own Detector.
#
# One-click smoke (CPU, no model download, no API key):
#   docker build -t turnkey .
#   docker run --rm turnkey
#
# Keep run artifacts on the host:
#   docker run --rm -v "$PWD/outputs:/app/outputs" turnkey
#
# Any other CLI invocation:
#   docker run --rm turnkey detector list
#   docker run --rm -v "$PWD/outputs:/app/outputs" turnkey run --config configs/runs/smoke_perplexity.yaml
#
# GPU / HF-model image (large; includes torch + transformers):
#   docker build --target full -t turnkey:full .
#   docker run --rm --gpus all -v "$PWD/outputs:/app/outputs" turnkey:full run --config configs/runs/smoke_hf.yaml
# (on hosts whose daemon only exposes the nvidia runtime, use --runtime=nvidia
#  instead of --gpus all)

FROM python:3.12-slim AS core
COPY --from=ghcr.io/astral-sh/uv:0.9.9 /uv /uvx /usr/local/bin/

ENV UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependency layer first so source edits do not re-resolve the environment.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

COPY . .
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"

ENTRYPOINT ["turnkey"]
CMD ["run", "--config", "configs/runs/smoke.yaml"]

FROM core AS full
RUN uv sync --frozen --no-dev --extra hf --extra judges
