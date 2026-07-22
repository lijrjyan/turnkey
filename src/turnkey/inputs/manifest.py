from __future__ import annotations

from dataclasses import asdict
import hashlib
from pathlib import Path
from typing import Any

from turnkey._internal.io import load_json_object
from turnkey.config import Config, dataset_identity
from turnkey.content import ContentSelectionReport
from turnkey._internal.redact import sha256_hex
from turnkey.schema import Sample


INPUT_MANIFEST_SCHEMA = "turnkey_input_manifest/v1"


def build_input_manifest(
    *,
    cfg: Config,
    content_report: ContentSelectionReport,
    selected_samples: list[Sample],
    attacked_samples: list[Sample],
    include_plaintext: bool = False,
) -> dict[str, Any]:
    if len(selected_samples) != len(attacked_samples):
        raise ValueError("input manifest requires selected and attacked sample lists with equal length")

    entries: list[dict[str, Any]] = []
    for source, attacked in zip(selected_samples, attacked_samples, strict=True):
        if source.sample_id != attacked.sample_id:
            raise ValueError(
                "input manifest requires attacks to preserve sample_id: "
                f"{source.sample_id!r} != {attacked.sample_id!r}"
            )
        entries.append(_sample_entry(source=source, attacked=attacked, include_plaintext=include_plaintext))

    dataset_meta = {"params": dict(cfg.dataset.params), **dataset_identity(cfg.dataset.name).to_metadata()}

    return {
        "schema_version": INPUT_MANIFEST_SCHEMA,
        "redaction": {"contains_plaintext": bool(include_plaintext)},
        "dataset": dataset_meta,
        "content": content_report.to_dict(),
        "attack": asdict(cfg.attack),
        "naturalness": asdict(cfg.naturalness),
        "run": {"max_samples": cfg.run.max_samples},
        "count": len(entries),
        "sample_ids": [entry["sample_id"] for entry in entries],
        "behavior_ids": [entry["behavior_id"] for entry in entries],
        "samples": entries,
    }


def load_input_manifest(path: str | Path) -> dict[str, Any]:
    try:
        value = load_json_object(path, root_error="input manifest root must be object")
    except TypeError as exc:
        raise ValueError(str(exc)) from exc
    if value.get("schema_version") == "turnkey_run/v1":
        input_value = value.get("input")
        manifest = input_value.get("manifest") if isinstance(input_value, dict) else None
        if not isinstance(manifest, dict):
            raise ValueError(f"{path}: run.json input.manifest must be an object")
        return manifest
    return value


def compare_input_manifests(*, expected: dict[str, Any], actual: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in (
        "schema_version",
        "dataset",
        "content",
        "attack",
        "naturalness",
        "run",
        "count",
        "sample_ids",
        "behavior_ids",
    ):
        if _identity_value(expected.get(key)) != _identity_value(actual.get(key)):
            errors.append(f"input manifest {key} mismatch")

    expected_samples = expected.get("samples")
    actual_samples = actual.get("samples")
    if not isinstance(expected_samples, list) or not isinstance(actual_samples, list):
        errors.append("input manifest samples must be lists")
        return errors
    if len(expected_samples) != len(actual_samples):
        errors.append(f"input manifest samples length mismatch: {len(expected_samples)} != {len(actual_samples)}")
        return errors

    for index, (expected_sample, actual_sample) in enumerate(zip(expected_samples, actual_samples, strict=True)):
        if _sample_identity(expected_sample) != _sample_identity(actual_sample):
            sample_id = None
            if isinstance(expected_sample, dict):
                sample_id = expected_sample.get("sample_id")
            errors.append(f"input manifest samples[{index}] mismatch for sample_id={sample_id!r}")
    return errors


def input_manifest_file_identity(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    text = data.decode("utf-8")
    return {
        "path": str(path),
        "sha256": sha256_hex(text),
        "bytes": len(data),
    }


def _sample_entry(*, source: Sample, attacked: Sample, include_plaintext: bool) -> dict[str, Any]:
    entry = {
        "sample_id": source.sample_id,
        "behavior_id": source.behavior_id,
        "is_benign": source.is_benign,
        "source": _sample_state(source, include_plaintext=include_plaintext),
        "attacked": _sample_state(attacked, include_plaintext=include_plaintext),
    }
    return entry


def _sample_state(sample: Sample, *, include_plaintext: bool) -> dict[str, Any]:
    prompt: dict[str, Any] = {"sha256": sha256_hex(sample.prompt), "chars": len(sample.prompt)}
    if include_plaintext:
        prompt["text"] = sample.prompt
    return {
        "prompt": prompt,
        "images": [_image_entry(image) for image in sample.images],
        "attack_family": sample.attack_family,
        "attack_method": sample.attack_method,
        "attack_params": dict(sample.attack_params),
    }


def _image_entry(image: Any) -> dict[str, Any]:
    path = getattr(image, "path", None)
    entry = {
        "path": path,
        "mime_type": getattr(image, "mime_type", None),
    }
    if isinstance(path, str):
        image_path = Path(path)
        if image_path.exists() and image_path.is_file():
            data = image_path.read_bytes()
            entry["sha256"] = hashlib.sha256(data).hexdigest()
            entry["bytes"] = len(data)
    return entry


def _sample_identity(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    return {
        "sample_id": value.get("sample_id"),
        "behavior_id": value.get("behavior_id"),
        "is_benign": value.get("is_benign"),
        "source": _state_identity(value.get("source")),
        "attacked": _state_identity(value.get("attacked")),
    }


def _state_identity(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    prompt = value.get("prompt")
    if isinstance(prompt, dict):
        prompt_identity = {
            "sha256": prompt.get("sha256"),
            "chars": prompt.get("chars"),
        }
    else:
        prompt_identity = prompt
    return {
        "prompt": prompt_identity,
        "images": _images_identity(value.get("images", [])),
        "attack_family": value.get("attack_family"),
        "attack_method": value.get("attack_method"),
        "attack_params": value.get("attack_params"),
    }


def _images_identity(value: Any) -> Any:
    if not isinstance(value, list):
        return value
    return [_image_identity(item) for item in value]


def _image_identity(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    if isinstance(value.get("sha256"), str):
        return {
            "sha256": value.get("sha256"),
            "bytes": value.get("bytes"),
            "mime_type": value.get("mime_type"),
        }
    return {
        "path": value.get("path"),
        "mime_type": value.get("mime_type"),
    }


def _identity_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _identity_value(val) for key, val in value.items() if key != "text"}
    if isinstance(value, list):
        return [_identity_value(item) for item in value]
    return value
