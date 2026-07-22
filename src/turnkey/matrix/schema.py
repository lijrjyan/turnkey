from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import yaml

from turnkey._internal.io import load_json_object
from turnkey.config import split_dataset_name_version


MATRIX_SPEC_SCHEMA = "turnkey_matrix_spec/v1"
MATRIX_PLAN_SCHEMA = "turnkey_matrix_plan/v1"
MATRIX_RESULTS_SCHEMA = "turnkey_matrix_results/v1"
MATRIX_SUMMARY_SCHEMA = "turnkey_matrix_summary/v1"
IMMUTABLE_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
REMOTE_DATASETS = {
    "jbb_behaviors",
    "sorrybench_202406",
    "sorrybench_public",
    "xstest",
}
JUDGE_REVISION_KEYS = {
    "guardreasoner": ("revision",),
    "llamaguard": ("revision",),
    "qwen3guard": ("revision",),
    "strongreject": ("adapter_revision", "base_model_revision"),
}


def _require_immutable_revision(value: object, *, location: str) -> None:
    if not isinstance(value, str) or IMMUTABLE_REVISION_RE.fullmatch(value) is None:
        raise ValueError(f"{location}: must be a full immutable commit SHA, got {value!r}")


def _validate_model_revision(value: object, *, location: str) -> None:
    if not isinstance(value, dict) or not isinstance(value.get("model_id"), str):
        return
    model_id = value["model_id"]
    if Path(model_id).is_absolute() or model_id.startswith(("./", "../", "~")):
        return
    _require_immutable_revision(value.get("revision"), location=f"{location}.revision")


def _validate_external_revisions(raw: dict[str, Any], *, path: Path) -> None:
    defaults = raw.get("defaults")
    if isinstance(defaults, dict):
        model = defaults.get("model")
        if isinstance(model, dict) and model.get("backend") == "hf":
            _validate_model_revision(model, location=f"{path}: defaults.model")
        judge = defaults.get("judge")
        if isinstance(judge, dict) and isinstance(judge.get("name"), str):
            params = judge.get("params")
            if isinstance(params, dict):
                for key in JUDGE_REVISION_KEYS.get(judge["name"], ()):
                    _require_immutable_revision(
                        params.get(key),
                        location=f"{path}: defaults.judge.params.{key}",
                    )

    datasets = raw.get("datasets")
    if isinstance(datasets, list):
        for index, dataset in enumerate(datasets):
            if not isinstance(dataset, dict) or not isinstance(dataset.get("name"), str):
                continue
            dataset_name, _version = split_dataset_name_version(dataset["name"])
            if dataset_name not in REMOTE_DATASETS:
                continue
            params = dataset.get("params")
            revision = params.get("revision") if isinstance(params, dict) else None
            _require_immutable_revision(
                revision,
                location=f"{path}: datasets[{index}].params.revision",
            )

    detectors = raw.get("detectors")
    if isinstance(detectors, list):
        for index, detector in enumerate(detectors):
            if not isinstance(detector, dict):
                continue
            params = detector.get("params")
            if not isinstance(params, dict):
                continue
            _validate_model_revision(params, location=f"{path}: detectors[{index}].params")
            _validate_model_revision(
                params.get("model"),
                location=f"{path}: detectors[{index}].params.model",
            )


def load_matrix_spec(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"{path}: matrix spec root must be a mapping")
    schema_version = raw.get("schema_version")
    if schema_version != MATRIX_SPEC_SCHEMA:
        raise ValueError(f"{path}: schema_version must be {MATRIX_SPEC_SCHEMA!r}, got {schema_version!r}")
    _validate_external_revisions(raw, path=path)
    return raw


def load_matrix_plan(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    raw = load_json_object(path, root_error="matrix plan root must be object")
    schema_version = raw.get("schema_version")
    if schema_version != MATRIX_PLAN_SCHEMA:
        raise ValueError(f"{path}: schema_version must be {MATRIX_PLAN_SCHEMA!r}, got {schema_version!r}")
    entries = raw.get("entries")
    if not isinstance(entries, list):
        raise TypeError(f"{path}: entries must be list")
    return raw


def load_matrix_results(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    raw = load_json_object(path, root_error="matrix results root must be object")
    schema_version = raw.get("schema_version")
    if schema_version != MATRIX_RESULTS_SCHEMA:
        raise ValueError(f"{path}: schema_version must be {MATRIX_RESULTS_SCHEMA!r}, got {schema_version!r}")
    results = raw.get("results")
    if not isinstance(results, list):
        raise TypeError(f"{path}: results must be list")
    return raw
