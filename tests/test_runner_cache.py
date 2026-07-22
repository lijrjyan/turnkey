from __future__ import annotations

import json
from pathlib import Path

import pytest

from turnkey.audit import audit_run_dir
from turnkey.components.backends.base import LLMBackend
from turnkey.capabilities import BackendCapabilities
from turnkey.config import Config, DetectorConfig, ModelConfig, RunConfig
from turnkey.runtime_providers import (
    GradientScoreRequest,
    GradientScoreResult,
    LastTokenHiddenStateRequest,
    LastTokenHiddenStateResult,
    ProviderSummary,
)
from turnkey.registry import register_backend
from turnkey.runner import run_eval
from turnkey.runner.cache import RunResourceCache
from turnkey.runner.io import runtime_cache_key
from turnkey.schema import ModelOutput, PromptLogprobs, TokenLogprob


ANCHOR_LOSS_GRADIENT = "anchor_loss_gradient"
LAST_TOKEN_BY_LAYER = "last_token_by_layer"


class _CachedObject:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


_RELEASABLE_BACKEND_STATE = {
    "backend_active": False,
    "backend_builds": 0,
    "closed": 0,
    "events": [],
    "generations": [],
    "prompt_logprob_calls": [],
    "provider_active": False,
    "provider_builds": 0,
    "provider_overlaps": [],
    "signal_overlaps": [],
}


class _ReleasableBackend(LLMBackend):
    def __init__(self) -> None:
        _RELEASABLE_BACKEND_STATE["backend_builds"] += 1
        self.build_number = _RELEASABLE_BACKEND_STATE["backend_builds"]
        _RELEASABLE_BACKEND_STATE["backend_active"] = True
        _RELEASABLE_BACKEND_STATE["events"].append(f"backend:{self.build_number}:open")

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(prompt_logprobs=True)

    def get_prompt_logprobs(self, *, prompt: str, images=None) -> PromptLogprobs:  # noqa: ARG002
        if _RELEASABLE_BACKEND_STATE["provider_active"]:
            _RELEASABLE_BACKEND_STATE["signal_overlaps"].append(prompt)
        _RELEASABLE_BACKEND_STATE["prompt_logprob_calls"].append((self.build_number, prompt))
        _RELEASABLE_BACKEND_STATE["events"].append(
            f"backend:{self.build_number}:prompt_logprobs:{prompt}"
        )
        return PromptLogprobs(
            tokens=(TokenLogprob(token=prompt, token_id=0, logprob=-0.1),),
            logprob_sum=-0.1,
            logprob_avg=-0.1,
        )

    def generate(
        self,
        *,
        prompt: str,
        images=None,
        max_new_tokens: int,
        temperature: float,
    ) -> ModelOutput:  # noqa: ARG002
        assert not _RELEASABLE_BACKEND_STATE["provider_active"]
        _RELEASABLE_BACKEND_STATE["generations"].append((self.build_number, prompt))
        _RELEASABLE_BACKEND_STATE["events"].append(f"backend:{self.build_number}:generate:{prompt}")
        return ModelOutput(
            executed=True,
            backend="test_releasable_backend",
            model_id="fake-target-model",
            response_text="OK",
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
            latency_s=0.0,
        )

    def close(self) -> None:
        _RELEASABLE_BACKEND_STATE["closed"] += 1
        _RELEASABLE_BACKEND_STATE["backend_active"] = False
        _RELEASABLE_BACKEND_STATE["events"].append(f"backend:{self.build_number}:close")


class _StagedGradientRequestProvider:
    request_type = GradientScoreRequest
    model_forwards_per_call = 1
    requires_exclusive_target = True

    def provide(self, request: GradientScoreRequest) -> GradientScoreResult:
        if not _RELEASABLE_BACKEND_STATE["provider_active"]:
            _RELEASABLE_BACKEND_STATE["provider_active"] = True
            _RELEASABLE_BACKEND_STATE["provider_builds"] += 1
            _RELEASABLE_BACKEND_STATE["events"].append("provider:open")
        if _RELEASABLE_BACKEND_STATE["backend_active"]:
            _RELEASABLE_BACKEND_STATE["provider_overlaps"].append(request.prompt)
        _RELEASABLE_BACKEND_STATE["events"].append(f"provider:score:{request.prompt}")
        return GradientScoreResult(
            score=0.1,
            target_tokens=1,
            n_tensors=1,
            provider=ProviderSummary(
                name="gradient_score",
                kind="gradients",
                requested=(ANCHOR_LOSS_GRADIENT,),
                materialized=("gradient_norm",),
                status="ok",
            ),
        )

    def close(self) -> None:
        _RELEASABLE_BACKEND_STATE["provider_active"] = False
        _RELEASABLE_BACKEND_STATE["events"].append("provider:close")


class _StagedHiddenStateRequestProvider:
    request_type = LastTokenHiddenStateRequest
    model_forwards_per_call = 1
    requires_exclusive_target = True

    def provide(self, request: LastTokenHiddenStateRequest) -> LastTokenHiddenStateResult:
        import torch

        if not _RELEASABLE_BACKEND_STATE["provider_active"]:
            _RELEASABLE_BACKEND_STATE["provider_active"] = True
            _RELEASABLE_BACKEND_STATE["provider_builds"] += 1
            _RELEASABLE_BACKEND_STATE["events"].append("provider:open")
        if _RELEASABLE_BACKEND_STATE["backend_active"]:
            _RELEASABLE_BACKEND_STATE["provider_overlaps"].append(request.prompt)
        _RELEASABLE_BACKEND_STATE["events"].append(f"provider:hidden:{request.prompt}")
        sign = 1.0 if request.is_benign else -1.0
        return LastTokenHiddenStateResult(
            last_token_by_layer=torch.tensor(
                [[0.0, sign, 0.0], [sign, 0.5, 1.0]],
                dtype=torch.float32,
            ),
            n_layers=2,
            hidden_size=3,
            device="cpu",
            provider=ProviderSummary(
                name="last_token_hidden_states",
                kind="hidden_states",
                requested=(LAST_TOKEN_BY_LAYER,),
                materialized=(LAST_TOKEN_BY_LAYER,),
                status="ok",
            ),
        )

    def close(self) -> None:
        _RELEASABLE_BACKEND_STATE["provider_active"] = False
        _RELEASABLE_BACKEND_STATE["events"].append("provider:close")


@register_backend("test_releasable_backend")
def _build_releasable_backend(_cfg: ModelConfig) -> LLMBackend:
    return _ReleasableBackend()


def test_run_resource_cache_manifest_records_hashed_entries_without_raw_keys() -> None:
    cache = RunResourceCache()
    cache.backends[runtime_cache_key(ModelConfig(backend="vllm", model_id="secret-backend-model"))] = (
        _CachedObject()
    )
    cache.judges["secret-judge"] = _CachedObject()

    manifest = cache.manifest()

    assert manifest["schema_version"] == "turnkey_runtime_cache_manifest/v2"
    assert manifest["counts"] == {"backends": 1, "judges": 1}
    entries = manifest["entries"]
    assert {entry["bucket"] for entry in entries} == {"backends", "judges"}
    assert all(len(entry["key_sha256"]) == 64 for entry in entries)
    assert all(entry["object_type"].endswith("._CachedObject") for entry in entries)
    assert all(entry["ready"] is True for entry in entries)
    assert all("cache_kind" not in entry for entry in entries)
    assert "secret-backend-model" not in str(manifest)
    assert "secret-judge" not in str(manifest)


def test_run_resource_cache_has_no_legacy_method_provider_buckets() -> None:
    cache = RunResourceCache()

    assert cache.summary() == {"backends": 0, "judges": 0}
    assert not hasattr(cache, "detectors")
    assert not hasattr(cache, "get_detector")
    assert not hasattr(cache, "gradient_providers")
    assert not hasattr(cache, "hidden_state_providers")
    assert not hasattr(cache, "runtime_provider_caches")


def test_release_backend_preserves_hashed_manifest_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = ModelConfig(backend="dummy", model_id="secret-released-model")
    key = runtime_cache_key(cfg)
    backend = _CachedObject()
    cache = RunResourceCache(backends={key: backend})
    monkeypatch.setattr(cache, "release_unused_accelerator_memory", lambda: None)

    cache.release_backend(cfg)

    assert backend.closed is True
    assert cache.summary() == {"backends": 0, "judges": 0}
    assert cache.backends == {}
    manifest = cache.manifest()
    assert len(manifest["entries"]) == 1
    assert manifest["entries"][0]["bucket"] == "backends"
    assert manifest["entries"][0]["ready"] is False
    assert len(manifest["entries"][0]["key_sha256"]) == 64
    assert "secret-released-model" not in str(manifest)


def test_policy_gradsafe_reuses_reference_outputs_without_reopening_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from turnkey.components.detectors import gradsafe

    _reset_releasable_backend_state()
    monkeypatch.setattr(
        gradsafe,
        "GradientScoreRequestProvider",
        _StagedGradientRequestProvider,
    )
    cfg = Config(
        run=RunConfig(name="policy-gradient-staged", out_dir=str(tmp_path), max_samples=2),
        model=ModelConfig(
            backend="test_releasable_backend",
            model_id="fake-target-model",
            return_prompt_logprobs=True,
        ),
        detector=DetectorConfig(
            name="gradsafe",
            params={
                "model_id": "fake-gradient-model",
                "device": "cpu",
                "threshold": 0.5,
            },
        ),
    )

    run_dir = run_eval(cfg)

    assert audit_run_dir(run_dir) == []
    assert _RELEASABLE_BACKEND_STATE["backend_builds"] == 1
    assert _RELEASABLE_BACKEND_STATE["closed"] == 1
    assert _RELEASABLE_BACKEND_STATE["provider_builds"] == 1
    assert _RELEASABLE_BACKEND_STATE["provider_overlaps"] == []
    assert _RELEASABLE_BACKEND_STATE["signal_overlaps"] == []
    assert _RELEASABLE_BACKEND_STATE["prompt_logprob_calls"] == [
        (1, "Say hello in one short sentence."),
        (1, "Explain the water cycle in one sentence."),
    ]
    events = _RELEASABLE_BACKEND_STATE["events"]
    assert events.index("backend:1:close") < events.index("provider:open")
    assert events[-1] == "provider:close"
    assert _RELEASABLE_BACKEND_STATE["generations"] == [
        (1, "Say hello in one short sentence."),
        (1, "Explain the water cycle in one sentence."),
    ]
    rows = [
        json.loads(line)
        for line in (run_dir / "cases.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert all(row["intervention"]["model"]["prompt_logprobs"] is not None for row in rows)


def test_policy_rcs_releases_target_before_hidden_state_fit_and_reuses_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("torch")
    from turnkey.components.detectors import rcs

    _reset_releasable_backend_state()
    monkeypatch.setattr(
        rcs,
        "LastTokenHiddenStateRequestProvider",
        _StagedHiddenStateRequestProvider,
    )
    cfg = Config(
        run=RunConfig(name="policy-rcs-staged", out_dir=str(tmp_path), max_samples=2),
        model=ModelConfig(backend="test_releasable_backend", model_id="fake-target-model"),
        detector=DetectorConfig(
            name="rcs",
            params={
                "mode": "paper",
                "method": "kcd",
                "k": 1,
                "threshold": 0.0,
                "calibrate_threshold": False,
                "val_ratio": 0.0,
                "model": {
                    "model_id": "fake-hidden-model",
                    "device": "cpu",
                    "local_files_only": True,
                },
                "layer": 1,
                "projection_dim": 2,
                "projection_epochs": 1,
                "projection_batch_size": 4,
                "projection_dropout": 0.0,
                "benign_prompts": ["benign 0", "benign 1"],
                "malicious_prompts": ["malicious 0", "malicious 1"],
            },
        ),
    )

    run_dir = run_eval(cfg)

    assert audit_run_dir(run_dir) == []
    assert _RELEASABLE_BACKEND_STATE["backend_builds"] == 1
    assert _RELEASABLE_BACKEND_STATE["closed"] == 1
    assert _RELEASABLE_BACKEND_STATE["provider_builds"] == 1
    assert _RELEASABLE_BACKEND_STATE["provider_overlaps"] == []
    events = _RELEASABLE_BACKEND_STATE["events"]
    assert events.index("backend:1:close") < events.index("provider:open")
    assert events[-1] == "provider:close"
    assert _RELEASABLE_BACKEND_STATE["generations"] == [
        (1, "Say hello in one short sentence."),
        (1, "Explain the water cycle in one sentence."),
    ]
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["cost"]["extra_forwards_avg"] == 3.0


def _reset_releasable_backend_state() -> None:
    _RELEASABLE_BACKEND_STATE.update(
        {
            "backend_active": False,
            "backend_builds": 0,
            "closed": 0,
            "events": [],
            "generations": [],
            "prompt_logprob_calls": [],
            "provider_active": False,
            "provider_builds": 0,
            "provider_overlaps": [],
            "signal_overlaps": [],
        }
    )
