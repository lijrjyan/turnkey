from __future__ import annotations

import hashlib
from pathlib import Path
import re
from typing import Any

from turnkey._internal.data import json_sha256


RUN_SCHEMA = "turnkey_run/v1"
IMMUTABLE_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")


def check_run(
    *,
    path: Path,
    run: dict[str, Any],
    run_dir: Path,
    cases: list[dict[str, Any]],
    events: list[dict[str, Any]],
    errors: list[str],
) -> None:
    if run.get("schema_version") != RUN_SCHEMA:
        errors.append(f"{path}: schema_version must be {RUN_SCHEMA!r}")
    if run.get("status") != "complete":
        errors.append(f"{path}: status must be 'complete'")
    if run.get("run_id") != run_dir.name:
        errors.append(f"{path}: run_id must match run directory name")

    config = run.get("config")
    if not isinstance(config, dict):
        errors.append(f"{path}: config must be object")
        config = {}
    _check_model(path=path, run=run, config=config, errors=errors)
    _check_components(path=path, run=run, config=config, errors=errors)
    _check_calibration(path=path, run=run, config=config, errors=errors)
    _check_runtime_cache(path=path, run=run, errors=errors)

    artifacts = run.get("artifacts")
    if not isinstance(artifacts, dict):
        errors.append(f"{path}: artifacts must be object")
    else:
        for name, expected_count in (
            ("cases", len(cases)),
            ("events", len(events)),
            ("metrics", 1),
        ):
            _check_artifact(
                path=path,
                run_dir=run_dir,
                name=name,
                value=artifacts.get(name),
                expected_count=expected_count,
                errors=errors,
            )

    input_value = run.get("input")
    if not isinstance(input_value, dict):
        errors.append(f"{path}: input must be object")
    else:
        case_ids = [case.get("case_id") for case in cases]
        if input_value.get("case_ids") != case_ids:
            errors.append(f"{path}: input.case_ids do not match cases.jsonl order")
        _check_input_manifest(path=path, input_value=input_value, cases=cases, errors=errors)

    redaction = run.get("redaction")
    if not isinstance(redaction, dict) or redaction.get("public_redacted") is not True:
        errors.append(f"{path}: redaction.public_redacted must be true")
    _check_private(path=path, run_dir=run_dir, redaction=redaction, errors=errors)


def _check_model(
    *,
    path: Path,
    run: dict[str, Any],
    config: dict[str, Any],
    errors: list[str],
) -> None:
    model = run.get("model")
    configured = config.get("model")
    if not isinstance(model, dict) or not isinstance(configured, dict):
        errors.append(f"{path}: model and config.model must be objects")
        return
    for metadata_key, config_key in (
        ("backend", "backend"),
        ("model_id", "model_id"),
        ("requested_revision", "revision"),
        ("device", "device"),
        ("trust_remote_code", "trust_remote_code"),
    ):
        if model.get(metadata_key) != configured.get(config_key):
            errors.append(f"{path}: model.{metadata_key} does not match config.model.{config_key}")
    requested = model.get("requested_revision")
    resolved = model.get("resolved_revision")
    if isinstance(requested, str) and IMMUTABLE_REVISION_RE.fullmatch(requested):
        if resolved is not None and resolved != requested:
            errors.append(f"{path}: model.resolved_revision does not match requested revision")
        model_id = model.get("model_id")
        is_remote_hf = (
            model.get("backend") == "hf"
            and isinstance(model_id, str)
            and not Path(model_id).is_absolute()
            and not model_id.startswith(("./", "../", "~"))
        )
        if is_remote_hf and resolved != requested:
            errors.append(f"{path}: model.resolved_revision must resolve immutable HF revision")


def _check_components(
    *,
    path: Path,
    run: dict[str, Any],
    config: dict[str, Any],
    errors: list[str],
) -> None:
    components = run.get("components")
    if not isinstance(components, dict):
        errors.append(f"{path}: components must be object")
        return
    for pass_name, configured in (
        ("reference", _nested_dict(config, "nsg", "baseline_detector")),
        ("intervention", _nested_dict(config, "detector")),
    ):
        component = components.get(pass_name)
        loc = f"{path}: components.{pass_name}"
        if not isinstance(component, dict) or not isinstance(configured, dict):
            errors.append(f"{loc}: component and matching config must be objects")
            continue
        expected_name = configured.get("resolved_name", configured.get("name"))
        if component.get("name") != expected_name:
            errors.append(f"{loc}.name does not match effective config")
        parameters = component.get("parameters")
        if not isinstance(parameters, dict):
            errors.append(f"{loc}.parameters must be object")
        else:
            expected_hash = configured.get("component_parameters_sha256")
            if expected_hash != json_sha256(parameters):
                errors.append(f"{loc}.parameters do not match effective config hash")
        source = component.get("source")
        configured_source = configured.get("component_source")
        if source != configured_source:
            errors.append(f"{loc}.source does not match effective config")
        _check_source_identity(loc=loc, source=source, errors=errors)

    dataset = components.get("dataset")
    configured_dataset = config.get("dataset")
    if isinstance(dataset, dict) and isinstance(configured_dataset, dict):
        params = configured_dataset.get("params")
        requested = params.get("revision") if isinstance(params, dict) else None
        if dataset.get("requested_revision") != requested:
            errors.append(f"{path}: components.dataset.requested_revision does not match config")
        if isinstance(requested, str) and IMMUTABLE_REVISION_RE.fullmatch(requested):
            if dataset.get("resolved_revision") != requested:
                errors.append(
                    f"{path}: components.dataset.resolved_revision does not match requested revision"
                )


def _check_source_identity(*, loc: str, source: Any, errors: list[str]) -> None:
    if not isinstance(source, dict):
        errors.append(f"{loc}.source must be object")
        return
    if "path" not in source:
        return
    if not isinstance(source.get("path"), str) or not source.get("path"):
        errors.append(f"{loc}.source.path must be non-empty string")
    sha256 = source.get("sha256")
    if (
        not isinstance(sha256, str)
        or len(sha256) != 64
        or any(char not in "0123456789abcdef" for char in sha256)
    ):
        errors.append(f"{loc}.source.sha256 must be hex sha256")
    size = source.get("bytes")
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        errors.append(f"{loc}.source.bytes must be non-negative integer")


def _check_calibration(
    *,
    path: Path,
    run: dict[str, Any],
    config: dict[str, Any],
    errors: list[str],
) -> None:
    artifacts = run.get("calibration_artifacts")
    components = run.get("components")
    if not isinstance(artifacts, dict) or not isinstance(components, dict):
        errors.append(f"{path}: calibration_artifacts must be object")
        return
    for pass_name, configured in (
        ("reference", _nested_dict(config, "nsg", "baseline_detector")),
        ("intervention", _nested_dict(config, "detector")),
    ):
        artifact = artifacts.get(pass_name)
        component = components.get(pass_name)
        if not isinstance(artifact, dict) or not isinstance(configured, dict):
            errors.append(f"{path}: calibration_artifacts.{pass_name} must be object")
            continue
        configured_path = configured.get("calibration_artifact")
        mode = artifact.get("mode")
        if configured_path is None and mode != "not_configured":
            errors.append(f"{path}: calibration_artifacts.{pass_name}.mode must be not_configured")
        if configured_path is not None:
            identity = artifact.get("identity")
            if mode != "loaded" or not isinstance(identity, dict):
                errors.append(f"{path}: calibration_artifacts.{pass_name} must record loaded identity")
                continue
            if identity.get("path") != configured_path:
                errors.append(f"{path}: calibration_artifacts.{pass_name}.identity.path mismatch")
            if isinstance(component, dict) and artifact.get("detector_name") != component.get("name"):
                errors.append(f"{path}: calibration_artifacts.{pass_name}.detector_name mismatch")


def _check_runtime_cache(*, path: Path, run: dict[str, Any], errors: list[str]) -> None:
    cache = run.get("runtime_cache")
    if not isinstance(cache, dict):
        errors.append(f"{path}: runtime_cache must be object")
        return
    if cache.get("schema_version") != "turnkey_runtime_cache_manifest/v2":
        errors.append(f"{path}: runtime_cache.schema_version must be v2")
    counts = cache.get("counts")
    if not isinstance(counts, dict) or set(counts) != {"backends", "judges"}:
        errors.append(f"{path}: runtime_cache.counts must contain only backends and judges")
    entries = cache.get("entries")
    if not isinstance(entries, list):
        errors.append(f"{path}: runtime_cache.entries must be list")
        return
    for index, entry in enumerate(entries):
        loc = f"{path}: runtime_cache.entries[{index}]"
        if not isinstance(entry, dict):
            errors.append(f"{loc}: must be object")
            continue
        if entry.get("bucket") not in {"backends", "judges"}:
            errors.append(f"{loc}.bucket must be backends or judges")
        key_hash = entry.get("key_sha256")
        if not isinstance(key_hash, str) or len(key_hash) != 64:
            errors.append(f"{loc}.key_sha256 must be sha256")
        if type(entry.get("ready")) is not bool:
            errors.append(f"{loc}.ready must be bool")


def _nested_dict(value: dict[str, Any], *keys: str) -> dict[str, Any] | None:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current if isinstance(current, dict) else None


def _check_artifact(
    *,
    path: Path,
    run_dir: Path,
    name: str,
    value: Any,
    expected_count: int,
    errors: list[str],
) -> None:
    loc = f"{path}: artifacts.{name}"
    if not isinstance(value, dict):
        errors.append(f"{loc}: must be object")
        return
    relative = value.get("path")
    expected_relative = f"{name}.jsonl" if name != "metrics" else "metrics.json"
    if relative != expected_relative:
        errors.append(f"{loc}.path: expected {expected_relative!r}, got {relative!r}")
        return
    artifact_path = run_dir / expected_relative
    if not artifact_path.is_file():
        errors.append(f"{loc}.path: file does not exist")
        return
    data = artifact_path.read_bytes()
    expected = {
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "count": expected_count,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            errors.append(
                f"{loc}.{key}: expected {expected_value!r}, got {value.get(key)!r}"
            )


def _check_input_manifest(
    *,
    path: Path,
    input_value: dict[str, Any],
    cases: list[dict[str, Any]],
    errors: list[str],
) -> None:
    manifest = input_value.get("manifest")
    if not isinstance(manifest, dict):
        errors.append(f"{path}: input.manifest must be object")
        return
    redaction = manifest.get("redaction")
    if not isinstance(redaction, dict) or redaction.get("contains_plaintext") is not False:
        errors.append(f"{path}: input.manifest must not contain plaintext")
    samples = manifest.get("samples")
    if not isinstance(samples, list):
        errors.append(f"{path}: input.manifest.samples must be list")
        return
    if len(samples) != len(cases):
        errors.append(f"{path}: input.manifest.samples count does not match cases")
        return
    for index, (sample, case) in enumerate(zip(samples, cases, strict=True)):
        loc = f"{path}: input.manifest.samples[{index}]"
        if not isinstance(sample, dict):
            errors.append(f"{loc}: must be object")
            continue
        if sample.get("sample_id") != case.get("case_id"):
            errors.append(f"{loc}.sample_id does not match case_id")
        attacked = sample.get("attacked")
        prompt = attacked.get("prompt") if isinstance(attacked, dict) else None
        if not isinstance(prompt, dict) or prompt.get("sha256") != case.get("prompt", {}).get("sha256"):
            errors.append(f"{loc}.attacked.prompt does not match public case prompt")
        if isinstance(prompt, dict) and "text" in prompt:
            errors.append(f"{loc}.attacked.prompt must not contain text")


def _check_private(
    *,
    path: Path,
    run_dir: Path,
    redaction: Any,
    errors: list[str],
) -> None:
    private_path = run_dir / "private" / "cases.jsonl"
    enabled = isinstance(redaction, dict) and redaction.get("private_enabled") is True
    if private_path.exists() and not enabled:
        errors.append(f"{path}: unrecorded private/cases.jsonl sidecar")
    if enabled and not private_path.is_file():
        errors.append(f"{path}: recorded private artifact does not exist")
    if not enabled:
        return
    identity = redaction.get("private_artifact") if isinstance(redaction, dict) else None
    _check_artifact_identity(
        path=path,
        artifact_path=private_path,
        identity=identity,
        expected_relative="private/cases.jsonl",
        errors=errors,
    )


def _check_artifact_identity(
    *,
    path: Path,
    artifact_path: Path,
    identity: Any,
    expected_relative: str,
    errors: list[str],
) -> None:
    loc = f"{path}: redaction.private_artifact"
    if not isinstance(identity, dict):
        errors.append(f"{loc}: must be object")
        return
    if identity.get("path") != expected_relative:
        errors.append(f"{loc}.path: expected {expected_relative!r}")
    if not artifact_path.is_file():
        return
    data = artifact_path.read_bytes()
    for key, expected in (
        ("sha256", hashlib.sha256(data).hexdigest()),
        ("bytes", len(data)),
    ):
        if identity.get(key) != expected:
            errors.append(f"{loc}.{key}: expected {expected!r}, got {identity.get(key)!r}")
