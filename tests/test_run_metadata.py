from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest
import yaml

from turnkey.audit import audit_run_dir
from turnkey.calibration import (
    CalibrationArtifact,
    CalibrationDataManifest,
    CalibrationTargetModel,
    ScoreDistributionSummary,
    detector_config_hash,
    write_calibration_artifact,
)
from turnkey.components.backends.dummy import DummyBackend
from turnkey.config import load_config
from turnkey.runner import run_eval


REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


class _ResolvedRevisionBackend(DummyBackend):
    def resolved_revision(self) -> str:
        return REVISION


class _MismatchedRevisionBackend(DummyBackend):
    def resolved_revision(self) -> str:
        return "0" * 40


def _config(tmp_path: Path, *, backend: str = "dummy") -> Path:
    raw = yaml.safe_load(Path("configs/runs/smoke.yaml").read_text(encoding="utf-8"))
    raw["run"]["out_dir"] = str(tmp_path / "outputs")
    raw["model"]["backend"] = backend
    if backend == "hf":
        raw["model"].update({"model_id": "Qwen/Qwen3-0.6B", "revision": REVISION})
    path = tmp_path / "smoke.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def test_run_metadata_owns_config_components_command_and_inputs(tmp_path: Path) -> None:
    config_path = _config(tmp_path)
    run_dir = run_eval(
        load_config(config_path),
        source_config_path=str(config_path),
        command=["turnkey", "run", "--config", str(config_path)],
    )
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))

    assert run["status"] == "complete"
    assert run["command"]["argv"] == ["turnkey", "run", "--config", str(config_path)]
    assert run["model"]["model_id"] == "dummy-smoke"
    assert run["components"]["dataset"]["versioned_name"] == "fixtures_smoke@v1"
    assert run["components"]["reference"]["name"] == "allow_all"
    assert run["components"]["intervention"]["name"] == "keyword"
    assert run["input"]["case_ids"] == run["input"]["manifest"]["sample_ids"]
    assert audit_run_dir(run_dir) == []


def test_run_metadata_records_and_audits_resolved_model_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _config(tmp_path, backend="hf")
    monkeypatch.setattr(
        "turnkey.runner.load_backend",
        lambda config: _ResolvedRevisionBackend(model_id=config.model_id),
    )
    run_dir = run_eval(load_config(config_path), source_config_path=str(config_path))
    run_path = run_dir / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    assert run["model"]["requested_revision"] == REVISION
    assert run["model"]["resolved_revision"] == REVISION

    run["model"]["resolved_revision"] = None
    run_path.write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    assert any("model.resolved_revision" in error for error in audit_run_dir(run_dir))


def test_run_rejects_unresolved_or_mismatched_remote_hf_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _config(tmp_path, backend="hf")
    monkeypatch.setattr(
        "turnkey.runner.load_backend",
        lambda config: DummyBackend(model_id=config.model_id),
    )
    with pytest.raises(RuntimeError, match="did not report a resolved revision"):
        run_eval(load_config(config_path), source_config_path=str(config_path))

    monkeypatch.setattr(
        "turnkey.runner.load_backend",
        lambda config: _MismatchedRevisionBackend(model_id=config.model_id),
    )
    with pytest.raises(RuntimeError, match="does not match requested revision"):
        run_eval(
            replace(load_config(config_path), run=replace(load_config(config_path).run, name="mismatch")),
            source_config_path=str(config_path),
        )


def test_run_metadata_records_loaded_calibration(tmp_path: Path) -> None:
    config_path = _config(tmp_path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    artifact_path = tmp_path / "keyword-calibration.json"
    write_calibration_artifact(
        artifact_path,
        CalibrationArtifact(
            detector_name="keyword",
            detector_version="v1",
            artifact_kind="threshold",
            target_model=CalibrationTargetModel(model_id="dummy-smoke", backend="dummy"),
            detector_config_hash=detector_config_hash(raw["detector"]["params"]),
            method={"reproduction_scope": "bounded_fixture", "procedure_id": "fixture"},
            calibration_data=CalibrationDataManifest(
                source="fixture",
                count=4,
                benign_count=2,
                harmful_count=2,
            ),
            threshold=0.5,
            score_summary={"all": ScoreDistributionSummary(count=4, mean=0.5)},
        ),
    )
    raw["detector"]["calibration_artifact"] = str(artifact_path)
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    run_dir = run_eval(load_config(config_path), source_config_path=str(config_path))
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    calibration = run["calibration_artifacts"]["intervention"]

    assert calibration["mode"] == "loaded"
    assert calibration["identity"]["path"] == str(artifact_path)
    assert calibration["detector_name"] == "keyword"
    assert calibration["threshold"] == 0.5
    assert run["calibration_artifacts"]["reference"]["mode"] == "not_configured"
    assert audit_run_dir(run_dir) == []
