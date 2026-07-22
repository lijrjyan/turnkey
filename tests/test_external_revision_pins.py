from __future__ import annotations

from pathlib import Path
import re

import pytest
import yaml

from turnkey.matrix.schema import load_matrix_spec


COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
REMOTE_DATASETS = {
    "jbb_behaviors",
    "sorrybench_202406",
    "sorrybench_public",
    "xstest",
}


def _assert_commit(value: object, *, location: str) -> None:
    assert isinstance(value, str) and COMMIT_RE.fullmatch(value), (
        f"{location} must be pinned to a full immutable commit SHA, got {value!r}"
    )


def _assert_model_ref_pinned(model: object, *, location: str) -> None:
    if not isinstance(model, dict) or not isinstance(model.get("model_id"), str):
        return
    model_id = model["model_id"]
    if Path(model_id).is_absolute() or model_id.startswith(("./", "../", "~")):
        return
    _assert_commit(model.get("revision"), location=f"{location}.revision")


def _assert_config_tree_pinned(path: Path) -> None:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    is_matrix = path.parent.name == "matrix"

    if is_matrix:
        defaults = raw.get("defaults") or {}
        model = defaults.get("model") or {}
        if model.get("backend") == "hf":
            _assert_model_ref_pinned(model, location=f"{path}: defaults.model")
        datasets = raw.get("datasets") or []
        detectors = raw.get("detectors") or []
    else:
        model = raw.get("model") or {}
        if model.get("backend") == "hf":
            _assert_model_ref_pinned(model, location=f"{path}: model")
        datasets = [raw.get("dataset") or {}]
        detectors = [raw.get("detector") or {}]

    for index, dataset in enumerate(datasets):
        dataset_name = dataset.get("name") if isinstance(dataset, dict) else None
        dataset_base = dataset_name.rsplit("@", 1)[0] if isinstance(dataset_name, str) else None
        if isinstance(dataset, dict) and dataset_base in REMOTE_DATASETS:
            params = dataset.get("params") or {}
            _assert_commit(
                params.get("revision"),
                location=f"{path}: datasets[{index}].params.revision",
            )

    for index, detector in enumerate(detectors):
        if not isinstance(detector, dict):
            continue
        params = detector.get("params") or {}
        _assert_model_ref_pinned(params, location=f"{path}: detectors[{index}].params")
        _assert_model_ref_pinned(
            params.get("model"),
            location=f"{path}: detectors[{index}].params.model",
        )


def test_committed_run_and_matrix_configs_pin_external_revisions() -> None:
    paths = sorted(Path("configs/runs").glob("*.yaml")) + sorted(
        Path("configs/matrix").glob("*.yaml")
    )
    assert paths
    for path in paths:
        _assert_config_tree_pinned(path)


def _write_matrix_spec(tmp_path: Path, raw: dict[str, object]) -> Path:
    path = tmp_path / "matrix.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def _valid_matrix_spec() -> dict[str, object]:
    return {
        "schema_version": "turnkey_matrix_spec/v1",
        "name": "revision-test",
        "defaults": {
            "model": {
                "backend": "hf",
                "model_id": "Qwen/Qwen3-0.6B",
                "revision": "c1899de289a04d12100db370d81485cdf75e47ca",
            }
        },
        "datasets": [
            {
                "name": "jbb_behaviors",
                "params": {"revision": "886acc352a31533ffbcf4ef22c744658688086fc"},
            }
        ],
        "attacks": [{"name": "none", "params": {}}],
        "detectors": [{"name": "allow_all", "params": {}}],
    }


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda raw: raw["defaults"]["model"].__setitem__("revision", None),
            "defaults.model.revision",
        ),
        (
            lambda raw: raw["datasets"][0]["params"].__setitem__("revision", "main"),
            "datasets\\[0\\].params.revision",
        ),
    ],
)
def test_matrix_spec_rejects_mutable_external_revisions(tmp_path: Path, mutate, match: str) -> None:
    raw = _valid_matrix_spec()
    mutate(raw)

    with pytest.raises(ValueError, match=match):
        load_matrix_spec(_write_matrix_spec(tmp_path, raw))


def test_matrix_spec_rejects_unpinned_detector_model(tmp_path: Path) -> None:
    raw = _valid_matrix_spec()
    raw["detectors"] = [
        {
            "name": "gradsafe",
            "params": {
                "model_id": "Qwen/Qwen3-0.6B",
                "revision": None,
            },
        }
    ]

    with pytest.raises(ValueError, match=r"detectors\[0\]\.params\.revision"):
        load_matrix_spec(_write_matrix_spec(tmp_path, raw))


def test_matrix_spec_rejects_unpinned_strongreject_models(tmp_path: Path) -> None:
    raw = _valid_matrix_spec()
    raw["defaults"]["judge"] = {
        "name": "strongreject",
        "params": {
            "evaluator": "strongreject_finetuned",
            "adapter_revision": "4bd893d32390d2cace4f067dc2e3ef5294fd78a2",
        },
    }

    with pytest.raises(ValueError, match=r"defaults\.judge\.params\.base_model_revision"):
        load_matrix_spec(_write_matrix_spec(tmp_path, raw))


def test_matrix_spec_rejects_unpinned_versioned_remote_dataset(tmp_path: Path) -> None:
    raw = _valid_matrix_spec()
    raw["datasets"] = [{"name": "jbb_behaviors@v1", "params": {}}]

    with pytest.raises(ValueError, match=r"datasets\[0\]\.params\.revision"):
        load_matrix_spec(_write_matrix_spec(tmp_path, raw))


def test_matrix_spec_allows_explicit_local_model_path_without_hub_revision(tmp_path: Path) -> None:
    raw = _valid_matrix_spec()
    raw["defaults"]["model"] = {
        "backend": "hf",
        "model_id": "/models/local-checkpoint",
    }

    assert load_matrix_spec(_write_matrix_spec(tmp_path, raw)) == raw


def test_matrix_spec_accepts_full_commit_revisions(tmp_path: Path) -> None:
    raw = _valid_matrix_spec()
    assert load_matrix_spec(_write_matrix_spec(tmp_path, raw)) == raw
