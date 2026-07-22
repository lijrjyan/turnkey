from __future__ import annotations

import json
from pathlib import Path

import pytest

from turnkey.calibration import (
    CALIBRATION_ARTIFACT_SCHEMA,
    CalibrationArtifact,
    CalibrationDataManifest,
    CalibrationTargetModel,
    ScoreDistributionSummary,
    calibration_artifact_identity,
    detector_config_hash,
    load_calibration_artifact,
    validate_calibration_artifact,
    write_calibration_artifact,
)


def _artifact() -> CalibrationArtifact:
    params = {"threshold": 0.5, "keywords": ["UNSAFE_PLACEHOLDER"]}
    return CalibrationArtifact(
        detector_name="keyword",
        detector_version="v1",
        artifact_kind="threshold",
        target_model=CalibrationTargetModel(model_id="dummy-smoke", backend="dummy"),
        detector_config_hash=detector_config_hash(params),
        method={
            "reproduction_scope": "bounded_fixture",
            "reference_sources": ["fixture"],
            "procedure_id": "keyword_threshold_fixture",
        },
        calibration_data=CalibrationDataManifest(
            source="fixture",
            count=4,
            benign_count=2,
            harmful_count=2,
            split="smoke",
            seed=7,
        ),
        threshold=0.5,
        operating_point={"metric": "balanced_accuracy", "value": 1.0},
        score_summary={
            "benign": ScoreDistributionSummary(count=2, mean=0.1, max=0.2),
            "harmful": ScoreDistributionSummary(count=2, mean=0.9, min=0.8),
        },
    )


def test_calibration_artifact_round_trip_and_identity(tmp_path: Path) -> None:
    path = tmp_path / "keyword-calibration.json"
    artifact = _artifact()

    write_calibration_artifact(path, artifact)
    loaded = load_calibration_artifact(path)

    assert loaded.to_dict() == artifact.to_dict()
    identity = calibration_artifact_identity(path)
    assert identity["path"] == str(path)
    assert identity["bytes"] == path.stat().st_size
    assert len(identity["sha256"]) == 64
    assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] == CALIBRATION_ARTIFACT_SCHEMA


def test_validate_calibration_artifact_rejects_detector_model_and_hash_mismatches() -> None:
    artifact = _artifact()
    params = {"threshold": 0.5, "keywords": ["UNSAFE_PLACEHOLDER"]}

    validate_calibration_artifact(
        artifact,
        detector_name="keyword",
        model_id="dummy-smoke",
        detector_params=params,
    )
    with pytest.raises(ValueError, match="detector mismatch"):
        validate_calibration_artifact(artifact, detector_name="other", model_id="dummy-smoke", detector_params=params)
    with pytest.raises(ValueError, match="model mismatch"):
        validate_calibration_artifact(artifact, detector_name="keyword", model_id="other", detector_params=params)
    with pytest.raises(ValueError, match="config hash mismatch"):
        validate_calibration_artifact(
            artifact,
            detector_name="keyword",
            model_id="dummy-smoke",
            detector_params={"threshold": 0.7},
        )


def test_calibration_artifact_requires_schema_version() -> None:
    raw = _artifact().to_dict()
    raw["schema_version"] = "other/v1"

    with pytest.raises(ValueError, match="unsupported calibration artifact schema"):
        CalibrationArtifact.from_dict(raw)


def test_detector_specific_calibration_artifact_examples_round_trip(tmp_path: Path) -> None:
    examples = [
        CalibrationArtifact(
            detector_name="rcs_paper_v3",
            detector_version="v3-standardized",
            artifact_kind="projection_threshold",
            target_model=CalibrationTargetModel(model_id="Qwen/Qwen3.5-2B", backend="hf"),
            method={
                "reproduction_scope": "bounded_reproduction",
                "procedure_id": "rcs_paper_fit_load",
                "reference_sources": ["paper", "official_repo"],
            },
            calibration_data=CalibrationDataManifest(source="redacted-jsonl", count=20, benign_count=10, harmful_count=10),
            threshold=0.0,
            score_summary={"validation": ScoreDistributionSummary(count=4, mean=0.0)},
            files={"projection": {"path": "projection.pt", "sha256": "0" * 64}},
        ),
        CalibrationArtifact(
            detector_name="gradsafe_v3",
            detector_version="v3-standardized",
            artifact_kind="reference_threshold",
            target_model=CalibrationTargetModel(model_id="Qwen/Qwen3.5-2B", backend="hf"),
            method={
                "reproduction_scope": "bounded_reproduction",
                "procedure_id": "gradsafe_reference_calibration",
                "reference_sources": ["paper", "official_repo"],
            },
            calibration_data=CalibrationDataManifest(source="redacted-jsonl", count=20, benign_count=10, harmful_count=10),
            threshold=1.0,
            score_summary={"validation": ScoreDistributionSummary(count=20, mean=0.5)},
            files={"reference": {"path": "gradsafe-reference.pt", "sha256": "1" * 64}},
        ),
        CalibrationArtifact(
            detector_name="jailguard_v3",
            detector_version="v3-standardized",
            artifact_kind="operating_point",
            target_model=CalibrationTargetModel(model_id="Qwen/Qwen3.5-2B", backend="hf"),
            method={
                "reproduction_scope": "bounded_reproduction",
                "procedure_id": "jailguard_operating_point_sweep",
                "reference_sources": ["paper", "official_repo"],
            },
            calibration_data=CalibrationDataManifest(source="redacted-jsonl", count=20, benign_count=10, harmful_count=10),
            threshold=0.02,
            operating_point={"target": "balanced_accuracy"},
            score_summary={"validation": ScoreDistributionSummary(count=20, mean=0.02)},
        ),
    ]

    for artifact in examples:
        path = tmp_path / f"{artifact.detector_name}.json"
        write_calibration_artifact(path, artifact)
        loaded = load_calibration_artifact(path)
        assert loaded.detector_name == artifact.detector_name
        assert loaded.method["reference_sources"] == ["paper", "official_repo"]
        assert loaded.method["reproduction_scope"] == "bounded_reproduction"
