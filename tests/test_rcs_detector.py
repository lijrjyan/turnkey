from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from turnkey.audit import audit_run_dir
from turnkey.config import DetectorConfig, load_config
from turnkey.components.detectors import load_detector
from turnkey.components.detectors.rcs import RCSDetector
from turnkey.runtime_providers import (
    LastTokenHiddenStateConfig,
    LastTokenHiddenStateRequest,
    LastTokenHiddenStateResult,
    ProviderSummary,
)
from turnkey.runner import run_eval
from turnkey.schema import ImageInput, Sample
from json_fixtures import load_json, load_jsonl


class FakeRCSHiddenStateProvider:
    def __init__(self, cfg: LastTokenHiddenStateConfig):
        self.cfg = cfg

    def last_token_by_layer(self, sample: Sample) -> LastTokenHiddenStateResult:
        import torch

        sign = 1.0 if sample.is_benign else -1.0
        offset = 0.2 if sample.sample_id.endswith("1") else 0.0
        tensor = torch.tensor(
            [
                [0.0, sign + offset, 0.0],
                [sign + offset, 0.5, 1.0],
            ],
            dtype=torch.float32,
        )
        return LastTokenHiddenStateResult(
            last_token_by_layer=tensor,
            n_layers=2,
            hidden_size=3,
            device="cpu",
            provider=ProviderSummary(
                name="last_token_hidden_states",
                kind="hidden_states",
                requested=("last_token_by_layer",),
                materialized=("last_token_by_layer",),
                capabilities={
                    "model_id": self.cfg.model_id,
                    "revision": self.cfg.revision,
                    "device": "cpu",
                    "torch_dtype": self.cfg.torch_dtype,
                    "n_layers": 2,
                    "hidden_size": 3,
                    "n_images": len(sample.images),
                    "image_token": self.cfg.image_token,
                    "auto_insert_image_tokens": self.cfg.auto_insert_image_tokens,
                    "token_strategy": self.cfg.token_strategy,
                    "include_embedding_layer": self.cfg.include_embedding_layer,
                    "model_family": self.cfg.model_family,
                    "max_length": self.cfg.max_length,
                },
                status="ok",
                message="fake_last_token_by_layer",
            ),
        )


def test_rcs_toy_mcd_blocks_placeholder_on_multimodal_sample() -> None:
    det = load_detector(
        DetectorConfig(name="rcs_toy_v3", params={"mode": "toy", "method": "mcd", "threshold": 0.0, "prototype_image_count": 1})
    )
    img = ImageInput(path="__t__")
    benign = Sample(sample_id="b", behavior_id="b", is_benign=True, prompt="Say hello.", images=(img,))
    unsafe = Sample(
        sample_id="u",
        behavior_id="u",
        is_benign=False,
        prompt="UNSAFE_PLACEHOLDER: request redacted (test).",
        images=(img,),
    )
    assert det.decide(benign).block is False
    assert det.decide(unsafe).block is True


def test_rcs_toy_kcd_runs() -> None:
    det = load_detector(DetectorConfig(name="rcs_toy_v3", params={"mode": "toy", "method": "kcd", "threshold": 0.0}))
    s = Sample(sample_id="s", behavior_id="s", is_benign=True, prompt="Hello.")
    out = det.decide(s)
    assert isinstance(out.block, bool)
    assert isinstance(out.reason, str)


def test_rcs_toy_manifest_documents_toy_scope() -> None:
    manifest = RCSDetector(mode="toy", method="mcd", threshold=0.0).manifest(name="rcs_toy_v3")
    data = manifest.to_dict()

    assert data["name"] == "rcs_toy_v3"
    assert data["required_inputs"] == ["sample", "prompt", "images"]
    assert data["reproducibility"]["mode"] == "toy"


def test_rcs_toy_registry_rejects_paper_mode() -> None:
    with pytest.raises(ValueError, match="only supports mode=toy"):
        load_detector(DetectorConfig(name="rcs_toy_v3", params={"mode": "paper"}))


def test_rcs_toy_policy_run_persists_component_and_audits(tmp_path: Path) -> None:
    raw = yaml.safe_load(Path("configs/runs/smoke_rcs.yaml").read_text(encoding="utf-8"))
    raw["run"]["out_dir"] = str(tmp_path / "outputs")
    raw["detector"] = {
        "name": "rcs_toy_v3",
        "params": {
            "mode": "toy",
            "method": "mcd",
            "threshold": 0.0,
            "prototype_image_count": 1,
        },
    }
    cfg_path = tmp_path / "rcs_toy_v3.yaml"
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    run_dir = run_eval(load_config(cfg_path), source_config_path=str(cfg_path))

    assert audit_run_dir(run_dir) == []
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    component = run["components"]["intervention"]
    assert component["name"] == "rcs_toy_v3"
    assert component["parameters"]["mode"] == "toy"
    assert component["parameters"]["method"] == "mcd"


def test_rcs_paper_manifest_documents_reproducibility_inputs() -> None:
    det = RCSDetector(
        mode="paper",
        method="kcd",
        model={"model_id": "fake-hidden-model", "device": "cpu", "local_files_only": True},
        layer=1,
        require_balanced_train=True,
        benign_prompts=["benign 0", "benign 1"],
        malicious_prompts=["malicious 0", "malicious 1"],
    )
    manifest = det.manifest(name="rcs_paper_v3")
    data = manifest.to_dict()

    assert data["name"] == "rcs_paper_v3"
    assert data["required_inputs"] == ["sample", "prompt", "images"]
    assert data["reproducibility"]["training_examples"]["count"] == 4
    assert data["reproducibility"]["training_examples"]["schema"]["version"] == "rcs_train_jsonl/v1"
    assert data["reproducibility"]["training_examples"]["balanced"] is True
    assert data["reproducibility"]["training_examples"]["require_balanced_train"] is True
    assert data["reproducibility"]["training_examples"]["dataset_counts"] == [
        {"dataset": "benign_proto", "count": 2, "benign_count": 2, "malicious_count": 0},
        {"dataset": "malicious_proto", "count": 2, "benign_count": 0, "malicious_count": 2},
    ]
    assert data["reproducibility"]["layer_selection_strategy"] == "auto_svm_silhouette_ratio"
    assert data["reproducibility"]["model"]["model_id"] == "fake-hidden-model"
    assert data["reproducibility"]["model"]["token_env"] == "HF_TOKEN"


def test_rcs_paper_accepts_balanced_train_jsonl_contract(tmp_path: Path) -> None:
    train_jsonl = tmp_path / "rcs_train.jsonl"
    train_jsonl.write_text(
        "\n".join(
            [
                json.dumps({"prompt": "benign a", "is_benign": True, "dataset": "alpaca"}),
                json.dumps({"prompt": "benign b", "is_benign": True, "dataset": "mm-vet"}),
                json.dumps({"prompt": "malicious a", "is_benign": False, "dataset": "advbench"}),
                json.dumps({"prompt": "malicious b", "is_benign": False, "dataset": "jailbreakv"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    det = RCSDetector(
        mode="paper",
        method="kcd",
        model={"model_id": "fake-hidden-model", "device": "cpu", "local_files_only": True},
        train_jsonl=str(train_jsonl),
        require_balanced_train=True,
    )
    manifest = det.manifest(name="rcs_paper_v3").to_dict()
    training = manifest["reproducibility"]["training_examples"]

    assert training["source"] == str(train_jsonl)
    assert training["source_type"] == "jsonl"
    assert training["count"] == 4
    assert training["benign_count"] == 2
    assert training["malicious_count"] == 2
    assert training["balanced"] is True
    assert training["dataset_counts"] == [
        {"dataset": "advbench", "count": 1, "benign_count": 0, "malicious_count": 1},
        {"dataset": "alpaca", "count": 1, "benign_count": 1, "malicious_count": 0},
        {"dataset": "jailbreakv", "count": 1, "benign_count": 0, "malicious_count": 1},
        {"dataset": "mm-vet", "count": 1, "benign_count": 1, "malicious_count": 0},
    ]


def test_rcs_paper_rejects_unbalanced_reference_train_set() -> None:
    with pytest.raises(ValueError, match="require_balanced_train=true"):
        RCSDetector(
            mode="paper",
            method="kcd",
            model={"model_id": "fake-hidden-model", "device": "cpu", "local_files_only": True},
            require_balanced_train=True,
            benign_prompts=["benign 0", "benign 1"],
            malicious_prompts=["malicious 0"],
        )


def test_rcs_paper_rejects_missing_reference_train_class(tmp_path: Path) -> None:
    train_jsonl = tmp_path / "only_benign.jsonl"
    train_jsonl.write_text(
        "\n".join(
            [
                json.dumps({"prompt": "benign a", "is_benign": True}),
                json.dumps({"prompt": "benign b", "is_benign": True}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="at least one benign and one malicious"):
        RCSDetector(
            mode="paper",
            method="kcd",
            model={"model_id": "fake-hidden-model", "device": "cpu", "local_files_only": True},
            train_jsonl=str(train_jsonl),
        )


def test_rcs_paper_policy_run_uses_typed_hidden_state_requests_and_audits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = _run_rcs_paper_v3_with_fake_provider(tmp_path, monkeypatch)

    assert audit_run_dir(run_dir) == []
    run = load_json(run_dir / "run.json")
    component = run["components"]["intervention"]
    assert component["name"] == "rcs_paper_v3"
    assert component["parameters"]["mode"] == "paper"
    assert component["parameters"]["method"] == "kcd"
    assert not any(
        entry["bucket"] == "hidden_state_providers"
        for entry in run["runtime_cache"]["entries"]
    )

    metrics = load_json(run_dir / "metrics.json")
    assert metrics["cost"]["extra_forwards_avg"] == 3.0

    rows = load_jsonl(run_dir / "cases.jsonl")
    assert rows[0]["intervention"]["detector"]["reason"].startswith("rcs(mode=paper, method=kcd, layer=1")
    assert rows[0]["intervention"]["detector"]["diagnostics"]["rcs_layer_selection"] == {
        "strategy": "fixed",
        "selected_layer": 1,
    }



def _run_rcs_paper_v3_with_fake_provider(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    pytest.importorskip("torch")
    from turnkey.components.detectors import rcs as rcs_module

    class FakeRCSHiddenStateRequestProvider:
        request_type = LastTokenHiddenStateRequest
        model_forwards_per_call = 1
        requires_exclusive_target = True

        def __init__(self) -> None:
            self._providers: dict[LastTokenHiddenStateConfig, FakeRCSHiddenStateProvider] = {}

        def provide(self, request: LastTokenHiddenStateRequest) -> LastTokenHiddenStateResult:
            provider = self._providers.get(request.config)
            if provider is None:
                provider = FakeRCSHiddenStateProvider(request.config)
                self._providers[request.config] = provider
            return provider.last_token_by_layer(request.to_sample())

        def close(self) -> None:
            self._providers.clear()

    monkeypatch.setattr(rcs_module, "LastTokenHiddenStateRequestProvider", FakeRCSHiddenStateRequestProvider)
    raw = yaml.safe_load(Path("configs/runs/smoke.yaml").read_text(encoding="utf-8"))
    raw["run"]["name"] = "rcs-paper-v3-provider"
    raw["run"]["out_dir"] = str(tmp_path / "outputs")
    raw["run"]["max_samples"] = 2
    raw["detector"] = {
        "name": "rcs_paper_v3",
        "params": {
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
                "hidden_state_token_strategy": "mean_pool",
                "include_embedding_layer": True,
                "model_family": "qwen",
                "max_length": 8192,
            },
            "layer": 1,
            "projection_dim": 2,
            "projection_epochs": 1,
            "projection_batch_size": 4,
            "projection_dropout": 0.0,
            "benign_prompts": ["benign 0", "benign 1"],
            "malicious_prompts": ["malicious 0", "malicious 1"],
        },
    }
    cfg_path = tmp_path / "rcs_paper_v3.yaml"
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    return run_eval(load_config(cfg_path), source_config_path=str(cfg_path))
