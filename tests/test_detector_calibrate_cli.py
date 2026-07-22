from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from turnkey.audit import audit_run_dir
from turnkey.calibration import CALIBRATION_REPORT_SCHEMA, load_calibration_artifact
from turnkey.cache_manifest import cache_entry_ready, load_cache_manifest
from turnkey.cli import main
from turnkey.config import load_config
from turnkey.components.detectors import gradsafe as gradsafe_module
from turnkey.components.detectors.gradsafe import GRADSAFE_REFERENCE_COSINE_THRESHOLD, GradSafeDetector
from turnkey.runtime_providers import GradientScoreResult, ProviderSummary, load_gradsafe_reference_artifact
from turnkey.schema import Sample
from turnkey.runner import run_eval


class FakeCalibrationHiddenStateProvider:
    def __init__(self, cfg):  # noqa: ANN001
        self.cfg = cfg

    def last_token_by_layer(self, sample: Sample):
        import torch

        sign = 1.0 if sample.is_benign else -1.0
        offset = 0.1 if sample.sample_id.endswith(("1", "3", "5")) else 0.0
        return torch.tensor(
            [
                [0.0, sign + offset, 0.0],
                [sign + offset, 0.5, 1.0],
            ],
            dtype=torch.float32,
        )


def _smoke_config(tmp_path: Path) -> Path:
    raw = yaml.safe_load(Path("configs/runs/smoke.yaml").read_text(encoding="utf-8"))
    raw["run"]["out_dir"] = str(tmp_path / "outputs")
    cfg_path = tmp_path / "smoke.yaml"
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return cfg_path


def test_detector_calibrate_cli_writes_keyword_artifact_and_report(tmp_path: Path) -> None:
    cfg_path = _smoke_config(tmp_path)
    artifact_path = tmp_path / "keyword-calibration.json"
    report_path = tmp_path / "keyword-calibration-report.json"

    rc = main(
        [
            "detector",
            "calibrate",
            "--config",
            str(cfg_path),
            "--out",
            str(artifact_path),
            "--report",
            str(report_path),
        ]
    )

    assert rc == 0
    artifact = load_calibration_artifact(artifact_path)
    assert artifact.detector_name == "keyword"
    assert artifact.artifact_kind == "fixture_contract"
    assert artifact.target_model.model_id == "dummy-smoke"
    assert artifact.method["reproduction_scope"] == "bounded_fixture"
    assert artifact.calibration_data is not None
    assert artifact.calibration_data.count == 4

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == CALIBRATION_REPORT_SCHEMA
    assert report["artifact"]["identity"]["path"] == str(artifact_path)
    assert report["score_summary"]["all"]["count"] == 4

    cfg = load_config(cfg_path)
    cfg = replace(cfg, detector=replace(cfg.detector, calibration_artifact=str(artifact_path)))
    run_dir = run_eval(cfg, source_config_path=str(cfg_path))
    assert audit_run_dir(run_dir) == []
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert run["calibration_artifacts"]["intervention"]["mode"] == "loaded"


def test_detector_calibrate_cli_records_lofo_holdout_family(tmp_path: Path) -> None:
    cfg_path = _smoke_config(tmp_path)
    artifact_path = tmp_path / "keyword-lofo-calibration.json"
    report_path = tmp_path / "keyword-lofo-calibration-report.json"

    rc = main(
        [
            "detector",
            "calibrate",
            "--config",
            str(cfg_path),
            "--out",
            str(artifact_path),
            "--report",
            str(report_path),
            "--holdout-family",
            "persona",
        ]
    )

    assert rc == 0
    artifact = load_calibration_artifact(artifact_path)
    assert artifact.method["lofo_holdout_family"] == "persona"
    assert artifact.metadata["lofo"] == {"holdout_family": "persona"}
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["lofo"] == {"holdout_family": "persona"}


def test_detector_calibrate_cli_writes_rcs_scoring_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("torch")
    from turnkey.components.detectors import rcs as rcs_module

    monkeypatch.setattr(rcs_module, "HFLastTokenHiddenStateProvider", FakeCalibrationHiddenStateProvider)
    raw = yaml.safe_load(Path("configs/runs/smoke.yaml").read_text(encoding="utf-8"))
    raw["model"]["backend"] = "hf"
    raw["model"]["model_id"] = "fake-hidden-model"
    raw["detector"] = {
        "name": "rcs_paper_v3",
        "params": {
            "mode": "paper",
            "method": "kcd",
            "k": 1,
            "threshold": 0.0,
            "calibrate_threshold": True,
            "val_ratio": 0.25,
            "objective_bal_acc_weight": 0.8,
            "objective_f1_weight": 0.2,
            "require_balanced_train": True,
            "model": {"model_id": "fake-hidden-model", "device": "cpu", "local_files_only": True},
            "layer": 1,
            "projection_dim": 2,
            "projection_epochs": 1,
            "projection_batch_size": 4,
            "projection_dropout": 0.0,
            "benign_prompts": ["benign 0", "benign 1", "benign 2"],
            "malicious_prompts": ["malicious 0", "malicious 1", "malicious 2"],
        },
    }
    cfg_path = tmp_path / "rcs.yaml"
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    artifact_path = tmp_path / "rcs-calibration.json"
    report_path = tmp_path / "rcs-calibration-report.json"

    rc = main(
        [
            "detector",
            "calibrate",
            "--config",
            str(cfg_path),
            "--out",
            str(artifact_path),
            "--report",
            str(report_path),
        ]
    )

    assert rc == 0
    artifact = load_calibration_artifact(artifact_path)
    assert artifact.detector_name == "rcs_paper_v3"
    assert artifact.artifact_kind == "rcs_paper_scoring_state"
    assert artifact.method["calibration_rule"] == "held_out_training_split_weighted_balanced_accuracy_f1"
    assert artifact.operating_point["threshold_objective"]["balanced_accuracy_weight"] == 0.8
    state_file = Path(artifact.files["scoring_state"]["path"])
    assert state_file.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == CALIBRATION_REPORT_SCHEMA
    assert report["files"]["scoring_state"]["path"] == str(state_file)


def test_detector_calibrate_cli_writes_gradsafe_operating_point(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class FakeScorer:
        def __init__(self, config):  # noqa: ANN001
            events.append(f"open:{config.model_id}:{config.score_mode}")

        def score(self, *, prompt: str, anchor_response: str) -> GradientScoreResult:
            events.append(f"score:{prompt}")
            score = 0.9 if prompt.startswith("UNSAFE_PLACEHOLDER") else 0.1
            return GradientScoreResult(
                score=score,
                target_tokens=1,
                n_tensors=1,
                provider=ProviderSummary(
                    name="gradient_score",
                    kind="gradients",
                    requested=("anchor_loss_gradient",),
                    materialized=("gradient_norm",),
                ),
            )

        def close(self) -> None:
            events.append("close")

    monkeypatch.setattr(gradsafe_module, "HFGradientScoreProvider", FakeScorer, raising=False)
    cfg_path = _smoke_config(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["model"]["model_id"] = "fake-gradient-model"
    raw["detector"] = {
        "name": "gradsafe_v3",
        "params": {
            "model_id": "fake-gradient-model",
            "threshold": 0.2,
            "max_length": 64,
            "parameter_regex": "(lm_head|embed_tokens|wte)",
        },
    }
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    artifact_path = tmp_path / "gradsafe-calibration.json"
    report_path = tmp_path / "gradsafe-calibration-report.json"

    rc = main(
        [
            "detector",
            "calibrate",
            "--config",
            str(cfg_path),
            "--out",
            str(artifact_path),
            "--report",
            str(report_path),
        ]
    )

    assert rc == 0
    artifact = load_calibration_artifact(artifact_path)
    assert artifact.detector_name == "gradsafe_v3"
    assert artifact.artifact_kind == "gradsafe_operating_point"
    assert artifact.method["procedure_id"] == "gradsafe_gradient_norm_threshold"
    assert artifact.operating_point["score_mode"] == "gradient_norm"
    assert 0.1 <= float(artifact.threshold or 0.0) <= 0.9
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == CALIBRATION_REPORT_SCHEMA
    assert report["score_summary"]["all"]["count"] == 4
    assert events[0] == "open:fake-gradient-model:gradient_norm"
    assert len([event for event in events if event.startswith("score:")]) == 4
    assert events[-1] == "close"


def test_detector_calibrate_cli_writes_gradsafe_reference_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    events: list[str] = []

    class FakeScorer:
        def __init__(self, config):  # noqa: ANN001
            self.config = config
            events.append(f"open:{config.score_mode}:{config.reference_artifact}")

        def gradient_tensors(self, *, prompt: str, anchor_response: str):
            events.append(f"gradients:{prompt}")
            sign = 1.0 if prompt.startswith("UNSAFE_PLACEHOLDER") else -1.0
            return {
                "layers.0.mlp.weight": torch.tensor(
                    [[sign, 0.0], [0.0, sign]]
                )
            }, 1

        def score(self, *, prompt: str, anchor_response: str) -> GradientScoreResult:
            events.append(f"score:{prompt}")
            return GradientScoreResult(
                score=0.9 if prompt.startswith("UNSAFE_PLACEHOLDER") else 0.1,
                target_tokens=1,
                n_tensors=1,
                provider=ProviderSummary(
                    name="gradient_score",
                    kind="gradients",
                    requested=("anchor_loss_gradient",),
                    materialized=("gradient_cosine_score",),
                ),
            )

        def close(self) -> None:
            events.append(f"close:{self.config.score_mode}")

    monkeypatch.setattr(gradsafe_module, "HFGradientScoreProvider", FakeScorer)
    cfg_path = _smoke_config(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["model"]["model_id"] = "fake-gradient-model"
    raw["detector"] = {
        "name": "gradsafe_v3",
        "params": {
            "model_id": "fake-gradient-model",
            "score_mode": "reference_cosine",
            "max_length": 64,
            "parameter_regex": "(mlp|self)",
        },
    }
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    artifact_path = tmp_path / "gradsafe-calibration.json"
    report_path = tmp_path / "gradsafe-calibration-report.json"

    rc = main(
        [
            "detector",
            "calibrate",
            "--config",
            str(cfg_path),
            "--out",
            str(artifact_path),
            "--report",
            str(report_path),
        ]
    )

    assert rc == 0
    artifact = load_calibration_artifact(artifact_path)
    assert artifact.method["procedure_id"] == "gradsafe_reference_cosine_reference_artifact"
    assert artifact.threshold == GRADSAFE_REFERENCE_COSINE_THRESHOLD
    assert artifact.operating_point["score_mode"] == "reference_cosine"
    assert artifact.operating_point["reference_artifact_source"] == "generated_from_calibration_samples"
    reference_path = Path(artifact.files["reference"]["path"])
    assert reference_path.exists()
    assert reference_path.name == "reference.pt"
    assert cache_entry_ready(reference_path.parent) is True
    reference_manifest = load_cache_manifest(reference_path.parent)
    assert reference_manifest["capability"] == "gradsafe_reference_artifact"
    assert reference_manifest["files"]["reference.pt"]["path"] == "reference.pt"
    reference = load_gradsafe_reference_artifact(str(reference_path), gap_threshold=1.0)
    assert reference["summary"]["selected_row_features"] == 2
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["files"]["reference"]["path"] == str(reference_path)
    assert events[0] == "open:gradient_norm:None"
    assert events.count("close:gradient_norm") == 1
    assert any(event.startswith("open:reference_cosine:") for event in events)
    assert events[-1] == "close:reference_cosine"


def test_detector_calibrate_cli_places_generated_gradsafe_reference_in_ready_cache_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    def fake_build_reference_artifact(
        _detector: GradSafeDetector,
        _provider,
        _samples: list[Sample],
        path: Path,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake-reference-artifact")
        return {"summary": {"selected_row_features": 1}}

    class FakeScorer:
        def __init__(self, config):  # noqa: ANN001
            events.append(f"open:{config.score_mode}")

        def score(self, *, prompt: str, anchor_response: str) -> GradientScoreResult:
            return GradientScoreResult(
                score=0.9 if prompt.startswith("UNSAFE_PLACEHOLDER") else 0.1,
                target_tokens=1,
                n_tensors=1,
                provider=ProviderSummary(
                    name="gradient_score",
                    kind="gradients",
                    requested=("anchor_loss_gradient",),
                    materialized=("gradient_cosine_score",),
                ),
            )

        def close(self) -> None:
            events.append("close")

    monkeypatch.setattr(gradsafe_module, "_build_gradsafe_reference_artifact", fake_build_reference_artifact)
    monkeypatch.setattr(gradsafe_module, "HFGradientScoreProvider", FakeScorer)
    cfg_path = _smoke_config(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["model"]["model_id"] = "fake-gradient-model"
    raw["detector"] = {
        "name": "gradsafe_v3",
        "params": {
            "model_id": "fake-gradient-model",
            "score_mode": "reference_cosine",
            "max_length": 64,
        },
    }
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    artifact_path = tmp_path / "gradsafe-calibration.json"

    rc = main(["detector", "calibrate", "--config", str(cfg_path), "--out", str(artifact_path)])

    assert rc == 0
    artifact = load_calibration_artifact(artifact_path)
    reference_path = Path(artifact.files["reference"]["path"])
    assert reference_path.name == "reference.pt"
    assert cache_entry_ready(reference_path.parent) is True
    reference_manifest = load_cache_manifest(reference_path.parent)
    assert reference_manifest["capability"] == "gradsafe_reference_artifact"
    assert reference_manifest["metadata"]["detector"] == "gradsafe_v3"
    assert reference_manifest["files"]["reference.pt"]["bytes"] == len(b"fake-reference-artifact")
    assert events == ["open:gradient_norm", "close", "open:reference_cosine", "close"]


def test_detector_calibrate_cli_writes_jailguard_operating_point(tmp_path: Path) -> None:
    cfg_path = _smoke_config(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["detector"] = {
        "name": "jailguard_v3",
        "params": {
            "n_variants": 4,
            "mutator": "PI",
            "char_rate": 0.1,
            "threshold": 2.0,
            "calibration_mode": "bounded_sweep",
            "similarity": "bow",
            "response_mode": "echo_prompt",
            "seed": 7,
        },
    }
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    artifact_path = tmp_path / "jailguard-calibration.json"
    report_path = tmp_path / "jailguard-calibration-report.json"

    rc = main(
        [
            "detector",
            "calibrate",
            "--config",
            str(cfg_path),
            "--out",
            str(artifact_path),
            "--report",
            str(report_path),
        ]
    )

    assert rc == 0
    artifact = load_calibration_artifact(artifact_path)
    assert artifact.detector_name == "jailguard_v3"
    assert artifact.artifact_kind == "jailguard_operating_point"
    assert artifact.method["procedure_id"] == "jailguard_operating_point_sweep"
    assert artifact.operating_point["response_mode"] == "echo_prompt"
    assert artifact.operating_point["paper_default_text_threshold"] == 0.02
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == CALIBRATION_REPORT_SCHEMA
    assert report["score_summary"]["all"]["count"] == 4

    cfg = load_config(cfg_path)
    cfg = replace(cfg, detector=replace(cfg.detector, calibration_artifact=str(artifact_path)))
    run_dir = run_eval(cfg, source_config_path=str(cfg_path))
    assert audit_run_dir(run_dir) == []
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert run["calibration_artifacts"]["intervention"]["detector_name"] == "jailguard_v3"


def test_detector_calibrate_cli_uses_jailguard_paper_default_threshold(tmp_path: Path) -> None:
    cfg_path = _smoke_config(tmp_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["detector"] = {
        "name": "jailguard_v3",
        "params": {
            "n_variants": 4,
            "mutator": "PI",
            "char_rate": 0.1,
            "similarity": "bow",
            "response_mode": "echo_prompt",
            "seed": 7,
        },
    }
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    artifact_path = tmp_path / "jailguard-calibration.json"

    rc = main(["detector", "calibrate", "--config", str(cfg_path), "--out", str(artifact_path)])

    assert rc == 0
    artifact = load_calibration_artifact(artifact_path)
    assert artifact.method["procedure_id"] == "jailguard_paper_default_threshold"
    assert artifact.method["threshold_rule"] == "paper_default_text_threshold_or_configured_fixed_threshold"
    assert artifact.threshold == 0.02
    assert artifact.operating_point["calibration_mode"] == "paper_default"
    assert "threshold_objective" not in artifact.operating_point


def test_detector_calibrate_cli_rejects_detector_without_handler(tmp_path: Path) -> None:
    raw = yaml.safe_load(Path("configs/runs/smoke.yaml").read_text(encoding="utf-8"))
    raw["detector"] = {"name": "allow_all_v3", "params": {}}
    cfg_path = tmp_path / "unsupported.yaml"
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="no calibration handler"):
        main(
            [
                "detector",
                "calibrate",
                "--config",
                str(cfg_path),
                "--out",
                str(tmp_path / "artifact.json"),
            ]
        )
