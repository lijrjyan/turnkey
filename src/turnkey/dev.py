from __future__ import annotations

from pathlib import Path

from turnkey.config import (
    Config,
    DatasetConfig,
    DetectorConfig,
    ModelConfig,
    NaturalnessConfig,
    RunConfig,
)
from turnkey.runner import run_eval


def run_method_dev(
    entrypoint: str,
    *,
    out_dir: str | Path = "outputs",
    max_samples: int = 4,
) -> Path:
    if max_samples <= 0:
        raise ValueError("dev max_samples must be positive")
    cfg = _dev_config(
        entrypoint=entrypoint,
        out_dir=out_dir,
        max_samples=max_samples,
    )
    return run_eval(
        cfg,
        command=[
            "turnkey",
            "dev",
            entrypoint,
            "--out-dir",
            str(out_dir),
            "--max-samples",
            str(max_samples),
        ],
    )


def _dev_config(
    *,
    entrypoint: str,
    out_dir: str | Path,
    max_samples: int,
) -> Config:
    return Config(
        run=RunConfig(
            name="dev",
            out_dir=str(out_dir),
            max_samples=max_samples,
            redact=True,
        ),
        dataset=DatasetConfig(
            name="fixtures_smoke@v1",
            params={"n_samples": max_samples},
        ),
        model=ModelConfig(
            backend="dummy",
            model_id="turnkey-dev-dummy",
            max_new_tokens=32,
            temperature=0.0,
        ),
        detector=DetectorConfig(name=entrypoint),
        naturalness=NaturalnessConfig(enabled=False),
    )
