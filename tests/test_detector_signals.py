from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

from turnkey.components.backends.base import LLMBackend
from turnkey.capabilities import BackendCapabilities
from turnkey.config import ModelConfig, load_config
from turnkey.registry import register_backend
from turnkey.runner import run_eval
from turnkey.schema import ModelOutput


@dataclass
class NoSignalBackend(LLMBackend):
    model_id: str = "no-signal"

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities()

    def generate(
        self,
        *,
        prompt: str,
        images=None,
        max_new_tokens: int,
        temperature: float,
    ) -> ModelOutput:  # noqa: ARG002
        return ModelOutput(
            executed=True,
            backend="test_no_signal",
            model_id=self.model_id,
            response_text="OK",
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
            latency_s=0.0,
        )


@register_backend("test_no_signal")
def _build_no_signal(cfg: ModelConfig) -> LLMBackend:
    return NoSignalBackend(model_id=cfg.model_id)


def _config_path(tmp_path: Path, *, backend: str = "dummy") -> Path:
    raw = yaml.safe_load(Path("configs/runs/smoke.yaml").read_text(encoding="utf-8"))
    raw["run"]["out_dir"] = str(tmp_path / "outputs")
    raw["model"]["backend"] = backend
    raw["model"]["model_id"] = "test-model"
    raw["model"]["return_prompt_logprobs"] = True
    raw["model"]["prefix_logprob_text"] = " signal"
    raw["detector"] = {"name": "allow_all", "params": {}}
    cfg_path = tmp_path / "signal_probe.yaml"
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return cfg_path


def test_model_signal_request_is_materialized_once_and_recorded(tmp_path: Path) -> None:
    cfg = load_config(_config_path(tmp_path))
    run_dir = run_eval(cfg)

    rows = [
        json.loads(line)
        for line in (run_dir / "cases.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows
    row = rows[0]
    assert row["intervention"]["detector"]["reason"] == "allow_all"
    assert row["intervention"]["model"]["prompt_logprobs"] is not None
    assert row["intervention"]["model"]["prefix_logprobs"] is not None


def test_model_signal_request_fails_when_backend_lacks_capability(tmp_path: Path) -> None:
    cfg = load_config(_config_path(tmp_path, backend="test_no_signal"))

    with pytest.raises(ValueError, match="prompt_logprobs"):
        run_eval(cfg)
