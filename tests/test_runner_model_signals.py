from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

from turnkey.capabilities import BackendCapabilities
from turnkey.components.backends.base import LLMBackend
from turnkey.config import ModelConfig, load_config
from turnkey.registry import register_backend
from turnkey.runner import run_eval
from turnkey.schema import ModelOutput, PrefixLogprobs, PromptLogprobs


@dataclass
class ModelSignalProbeBackend(LLMBackend):
    model_id: str = "model-signal-probe"

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(prompt_logprobs=True, prefix_logprobs=True)

    def get_prompt_logprobs(self, *, prompt: str, images=None) -> PromptLogprobs:  # noqa: ARG002
        return PromptLogprobs()

    def get_prefix_logprobs(
        self,
        *,
        prompt: str,
        prefix_text: str,
        images=None,
    ) -> PrefixLogprobs:  # noqa: ARG002
        return PrefixLogprobs(prefix_text=prefix_text)

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
            backend="test_model_signal_backend",
            model_id=self.model_id,
            response_text="OK",
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
            latency_s=0.0,
        )


@dataclass
class NoModelSignalBackend(ModelSignalProbeBackend):
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities()


@register_backend("test_model_signal_backend")
def _build_model_signal_backend(cfg: ModelConfig) -> LLMBackend:
    return ModelSignalProbeBackend(model_id=cfg.model_id)


@register_backend("test_no_model_signal_backend")
def _build_no_model_signal_backend(cfg: ModelConfig) -> LLMBackend:
    return NoModelSignalBackend(model_id=cfg.model_id)


def _config_path(tmp_path: Path, *, backend: str = "test_model_signal_backend") -> Path:
    raw = yaml.safe_load(Path("configs/runs/smoke.yaml").read_text(encoding="utf-8"))
    raw["run"]["out_dir"] = str(tmp_path / "outputs")
    raw["model"]["backend"] = backend
    raw["model"]["model_id"] = "model-signal-probe"
    raw["model"]["return_prompt_logprobs"] = True
    raw["model"]["prefix_logprob_text"] = " provider"
    raw["detector"] = {"name": "allow_all", "params": {}}
    cfg_path = tmp_path / "model_signal_probe.yaml"
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return cfg_path


def test_model_signal_provider_summary_is_visible_to_cases(tmp_path: Path) -> None:
    run_dir = run_eval(load_config(_config_path(tmp_path)))

    rows = [
        json.loads(line)
        for line in (run_dir / "cases.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows[0]["intervention"]["detector"]["reason"] == "allow_all"
    [provider] = rows[0]["intervention"]["signals"]["providers"]
    assert provider["name"] == "model_signals"
    assert provider["kind"] == "model_signals"
    assert provider["requested"] == ["prompt_logprobs", "prefix_logprobs"]
    assert provider["materialized"] == ["prompt_logprobs", "prefix_logprobs"]
    assert provider["capabilities"]["prompt_hidden_states"] == "none"
    assert provider["status"] == "ok"


def test_model_signal_request_fails_early_when_backend_lacks_capability(tmp_path: Path) -> None:
    cfg = load_config(_config_path(tmp_path, backend="test_no_model_signal_backend"))

    with pytest.raises(ValueError, match="signal request requires prompt_logprobs"):
        run_eval(cfg)
