from __future__ import annotations

import hashlib
import json
import math
import subprocess
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from turnkey.config import Config
from turnkey.components.detectors import load_detector
from turnkey.inputs.provider import materialize_input_provider


CALIBRATION_ARTIFACT_SCHEMA = "turnkey_detector_calibration/v1"
CALIBRATION_REPORT_SCHEMA = "turnkey_detector_calibration_report/v1"


@dataclass(frozen=True)
class CalibrationTargetModel:
    model_id: str
    backend: str | None = None
    revision: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CalibrationDataManifest:
    source: str
    count: int
    benign_count: int | None = None
    harmful_count: int | None = None
    split: str | None = None
    seed: int | None = None
    identity: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source.strip():
            raise ValueError("calibration data source must be non-empty")
        if int(self.count) < 0:
            raise ValueError("calibration data count must be >= 0")
        for field_name, value in (("benign_count", self.benign_count), ("harmful_count", self.harmful_count)):
            if value is not None and int(value) < 0:
                raise ValueError(f"calibration data {field_name} must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "count": int(self.count),
            "benign_count": self.benign_count,
            "harmful_count": self.harmful_count,
            "split": self.split,
            "seed": self.seed,
            "identity": dict(self.identity),
        }


@dataclass(frozen=True)
class ScoreDistributionSummary:
    count: int
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    p05: float | None = None
    p50: float | None = None
    p95: float | None = None
    max: float | None = None

    def __post_init__(self) -> None:
        if int(self.count) < 0:
            raise ValueError("score summary count must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CalibrationArtifact:
    detector_name: str
    target_model: CalibrationTargetModel
    threshold: float | None = None
    detector_version: str | None = None
    artifact_kind: str = "threshold"
    detector_config_hash: str | None = None
    method: dict[str, Any] = field(default_factory=dict)
    calibration_data: CalibrationDataManifest | None = None
    operating_point: dict[str, Any] = field(default_factory=dict)
    score_summary: dict[str, ScoreDistributionSummary] = field(default_factory=dict)
    files: dict[str, dict[str, Any]] = field(default_factory=dict)
    code: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = CALIBRATION_ARTIFACT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != CALIBRATION_ARTIFACT_SCHEMA:
            raise ValueError(f"unsupported calibration artifact schema: {self.schema_version}")
        if not self.detector_name.strip():
            raise ValueError("calibration artifact detector_name must be non-empty")
        if not self.artifact_kind.strip():
            raise ValueError("calibration artifact artifact_kind must be non-empty")
        if not self.target_model.model_id.strip():
            raise ValueError("calibration artifact target_model.model_id must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "detector_name": self.detector_name,
            "detector_version": self.detector_version,
            "artifact_kind": self.artifact_kind,
            "target_model": self.target_model.to_dict(),
            "detector_config_hash": self.detector_config_hash,
            "method": dict(self.method),
            "calibration_data": self.calibration_data.to_dict() if self.calibration_data is not None else None,
            "threshold": self.threshold,
            "operating_point": dict(self.operating_point),
            "score_summary": {name: summary.to_dict() for name, summary in self.score_summary.items()},
            "files": {name: dict(info) for name, info in self.files.items()},
            "code": dict(self.code),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CalibrationArtifact":
        if not isinstance(value, dict):
            raise TypeError("calibration artifact root must be an object")
        target_raw = value.get("target_model")
        if not isinstance(target_raw, dict):
            raise ValueError("calibration artifact target_model must be an object")
        data_raw = value.get("calibration_data")
        score_raw = value.get("score_summary") or {}
        if data_raw is not None and not isinstance(data_raw, dict):
            raise ValueError("calibration artifact calibration_data must be an object or null")
        if not isinstance(score_raw, dict):
            raise ValueError("calibration artifact score_summary must be an object")
        return cls(
            schema_version=str(value.get("schema_version", "")),
            detector_name=str(value.get("detector_name", "")),
            detector_version=_optional_str(value.get("detector_version")),
            artifact_kind=str(value.get("artifact_kind", "threshold")),
            target_model=CalibrationTargetModel(
                model_id=str(target_raw.get("model_id", "")),
                backend=_optional_str(target_raw.get("backend")),
                revision=_optional_str(target_raw.get("revision")),
            ),
            detector_config_hash=_optional_str(value.get("detector_config_hash")),
            method=_coerce_mapping(value.get("method"), "method"),
            calibration_data=CalibrationDataManifest(**data_raw) if data_raw is not None else None,
            threshold=_optional_float(value.get("threshold")),
            operating_point=_coerce_mapping(value.get("operating_point"), "operating_point"),
            score_summary={
                str(name): ScoreDistributionSummary(**summary)
                for name, summary in score_raw.items()
                if isinstance(summary, dict)
            },
            files=_coerce_mapping(value.get("files"), "files"),
            code=_coerce_mapping(value.get("code"), "code"),
            metadata=_coerce_mapping(value.get("metadata"), "metadata"),
        )


def detector_config_hash(params: dict[str, Any]) -> str:
    payload = json.dumps(params, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_calibration_artifact(path: str | Path) -> CalibrationArtifact:
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid calibration artifact json: {exc}") from exc
    return CalibrationArtifact.from_dict(raw)


def write_calibration_artifact(path: str | Path, artifact: CalibrationArtifact) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def calibration_artifact_identity(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    data = path.read_bytes()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }


def validate_calibration_artifact(
    artifact: CalibrationArtifact,
    *,
    detector_name: str,
    model_id: str,
    revision: str | None = None,
    detector_params: dict[str, Any] | None = None,
    require_config_hash: bool = False,
) -> None:
    if artifact.detector_name != detector_name:
        raise ValueError(
            f"calibration artifact detector mismatch: expected {detector_name!r}, got {artifact.detector_name!r}"
        )
    if artifact.target_model.model_id != model_id:
        raise ValueError(
            f"calibration artifact model mismatch: expected {model_id!r}, got {artifact.target_model.model_id!r}"
        )
    if artifact.target_model.revision is not None and revision is not None and artifact.target_model.revision != revision:
        raise ValueError(
            "calibration artifact revision mismatch: "
            f"expected {revision!r}, got {artifact.target_model.revision!r}"
        )
    expected_hash = detector_config_hash(detector_params or {})
    if require_config_hash and artifact.detector_config_hash is None:
        raise ValueError("calibration artifact is missing detector_config_hash")
    if artifact.detector_config_hash is not None and artifact.detector_config_hash != expected_hash:
        raise ValueError(
            "calibration artifact detector config hash mismatch: "
            f"expected {expected_hash}, got {artifact.detector_config_hash}"
        )


def calibration_artifact_receipt(path: str | Path, artifact: CalibrationArtifact) -> dict[str, Any]:
    return {
        "mode": "loaded",
        "identity": calibration_artifact_identity(path),
        "schema_version": artifact.schema_version,
        "detector_name": artifact.detector_name,
        "detector_version": artifact.detector_version,
        "artifact_kind": artifact.artifact_kind,
        "target_model": artifact.target_model.to_dict(),
        "detector_config_hash": artifact.detector_config_hash,
        "method": dict(artifact.method),
        "threshold": artifact.threshold,
        "operating_point": dict(artifact.operating_point),
        "score_summary": {name: summary.to_dict() for name, summary in artifact.score_summary.items()},
        "calibration_data": artifact.calibration_data.to_dict() if artifact.calibration_data is not None else None,
        "files": {name: dict(info) for name, info in artifact.files.items()},
        "code": dict(artifact.code),
        "metadata": dict(artifact.metadata),
    }


def no_calibration_receipt() -> dict[str, Any]:
    return {
        "mode": "not_configured",
        "identity": None,
        "schema_version": CALIBRATION_ARTIFACT_SCHEMA,
    }


def calibrate_detector_from_config(
    cfg: Config,
    *,
    artifact_path: str | Path,
    report_path: str | Path | None = None,
    source_config_path: str | None = None,
    command: list[str] | None = None,
    holdout_family: str | None = None,
) -> Path:
    handler = _CALIBRATORS.get(cfg.detector.name)
    if handler is None:
        raise ValueError(
            f"detector calibrate: no calibration handler for {cfg.detector.name!r}. "
            "Add a detector-specific paper calibration handler before using this detector."
        )

    artifact_path = Path(artifact_path)
    artifact, report = handler(
        cfg,
        artifact_path=artifact_path,
        source_config_path=source_config_path,
        command=command,
    )
    if holdout_family is not None:
        artifact = _with_lofo_holdout(artifact, holdout_family=holdout_family)
        report["lofo"] = {"holdout_family": holdout_family}
        report["method"] = dict(artifact.method)
    write_calibration_artifact(artifact_path, artifact)
    report["artifact"] = calibration_artifact_receipt(artifact_path, artifact)
    if report_path is not None:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return artifact_path


def _with_lofo_holdout(artifact: CalibrationArtifact, *, holdout_family: str) -> CalibrationArtifact:
    holdout_family = holdout_family.strip()
    if not holdout_family:
        raise ValueError("holdout_family must be non-empty")
    return replace(
        artifact,
        method={**artifact.method, "lofo_holdout_family": holdout_family},
        metadata={**artifact.metadata, "lofo": {"holdout_family": holdout_family}},
    )


def _calibrate_keyword_fixture(
    cfg: Config,
    *,
    artifact_path: Path,  # noqa: ARG001
    source_config_path: str | None,
    command: list[str] | None,
) -> tuple[CalibrationArtifact, dict[str, Any]]:
    input_bundle = materialize_input_provider(cfg)
    samples = input_bundle.attacked_samples
    detector = load_detector(cfg.detector)
    detector_manifest = detector.manifest(name=cfg.detector.name)
    decisions = [detector.decide(sample) for sample in samples]
    scores = [float(decision.score or 0.0) for decision in decisions]
    benign_scores = [score for score, sample in zip(scores, samples, strict=True) if sample.is_benign]
    harmful_scores = [score for score, sample in zip(scores, samples, strict=True) if sample.is_benign is False]
    blocked = sum(1 for decision in decisions if decision.block)
    benign_count = sum(1 for sample in samples if sample.is_benign)
    harmful_count = len(samples) - benign_count
    method = {
        "reproduction_scope": "bounded_fixture",
        "procedure_id": "keyword_fixture_contract",
        "reference_sources": ["fixture"],
    }
    artifact = CalibrationArtifact(
        detector_name=cfg.detector.name,
        detector_version=detector_manifest.version,
        artifact_kind="fixture_contract",
        target_model=CalibrationTargetModel(
            model_id=cfg.model.model_id,
            backend=cfg.model.backend,
            revision=cfg.model.revision,
        ),
        detector_config_hash=detector_config_hash(cfg.detector.params),
        method=method,
        calibration_data=CalibrationDataManifest(
            source=f"{cfg.dataset.name}/{cfg.attack.name}",
            count=len(samples),
            benign_count=benign_count,
            harmful_count=harmful_count,
            split="selected_attacked_samples",
            seed=cfg.content.seed,
        ),
        threshold=None,
        operating_point={
            "blocked": blocked,
            "allowed": len(samples) - blocked,
        },
        score_summary={
            "all": score_distribution_summary(scores),
            "benign": score_distribution_summary(benign_scores),
            "harmful": score_distribution_summary(harmful_scores),
        },
        code=_read_git_revision(),
        metadata={
            "detector_manifest": detector_manifest.to_dict(),
            "content": input_bundle.content_report.to_dict(),
        },
    )
    report = {
        "schema_version": CALIBRATION_REPORT_SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": {
            "argv": list(command) if command is not None else None,
            "source_config": _source_config_identity(source_config_path),
        },
        "detector": {
            "name": cfg.detector.name,
            "params": dict(cfg.detector.params),
            "manifest": detector_manifest.to_dict(),
        },
        "target_model": artifact.target_model.to_dict(),
        "method": method,
        "calibration_data": artifact.calibration_data.to_dict() if artifact.calibration_data is not None else None,
        "operating_point": dict(artifact.operating_point),
        "score_summary": {name: summary.to_dict() for name, summary in artifact.score_summary.items()},
    }
    return artifact, report


def _calibrate_rcs_paper(
    cfg: Config,
    *,
    artifact_path: Path,
    source_config_path: str | None,
    command: list[str] | None,
) -> tuple[CalibrationArtifact, dict[str, Any]]:
    from turnkey.components.detectors.rcs import calibrate_rcs_paper_from_config

    return calibrate_rcs_paper_from_config(
        cfg,
        artifact_path=artifact_path,
        source_config_path=source_config_path,
        command=command,
    )


def _calibrate_gradsafe(
    cfg: Config,
    *,
    artifact_path: Path,
    source_config_path: str | None,
    command: list[str] | None,
) -> tuple[CalibrationArtifact, dict[str, Any]]:
    from turnkey.components.detectors.gradsafe import calibrate_gradsafe_from_config

    return calibrate_gradsafe_from_config(
        cfg,
        artifact_path=artifact_path,
        source_config_path=source_config_path,
        command=command,
    )


def _calibrate_jailguard(
    cfg: Config,
    *,
    artifact_path: Path,
    source_config_path: str | None,
    command: list[str] | None,
) -> tuple[CalibrationArtifact, dict[str, Any]]:
    from turnkey.components.detectors.jailguard import calibrate_jailguard_from_config

    return calibrate_jailguard_from_config(
        cfg,
        artifact_path=artifact_path,
        source_config_path=source_config_path,
        command=command,
    )


def score_distribution_summary(scores: list[float]) -> ScoreDistributionSummary:
    finite = sorted(float(score) for score in scores if math.isfinite(float(score)))
    if not finite:
        return ScoreDistributionSummary(count=0)
    mean = sum(finite) / float(len(finite))
    variance = sum((score - mean) ** 2 for score in finite) / float(len(finite))
    return ScoreDistributionSummary(
        count=len(finite),
        mean=mean,
        std=math.sqrt(variance),
        min=finite[0],
        p05=_quantile(finite, 0.05),
        p50=_quantile(finite, 0.50),
        p95=_quantile(finite, 0.95),
        max=finite[-1],
    )


def _quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("cannot compute quantile of empty values")
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = max(0.0, min(1.0, float(q))) * float(len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    weight = pos - float(lo)
    return sorted_values[lo] * (1.0 - weight) + sorted_values[hi] * weight


def _read_git_revision() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        commit = None
    try:
        status = subprocess.check_output(["git", "status", "--short"], text=True, stderr=subprocess.DEVNULL)
        dirty = bool(status.strip())
    except Exception:  # noqa: BLE001
        dirty = None
    return {"commit": commit, "dirty": dirty}


def _source_config_identity(path: str | None) -> dict[str, Any] | None:
    if path is None:
        return None
    p = Path(path)
    identity: dict[str, Any] = {"path": str(path)}
    if p.exists() and p.is_file():
        data = p.read_bytes()
        identity.update({"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
    return identity


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _coerce_mapping(value: Any, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"calibration artifact {field_name} must be an object")
    return dict(value)


_CALIBRATORS = {
    "keyword": _calibrate_keyword_fixture,
    "gradsafe": _calibrate_gradsafe,
    "jailguard": _calibrate_jailguard,
    "rcs": _calibrate_rcs_paper,
}
