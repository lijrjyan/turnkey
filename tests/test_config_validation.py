from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from turnkey.config import load_config


def _write_config(tmp_path: Path, raw: dict) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def test_load_config_rejects_non_mapping_empty_root(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("[]\n", encoding="utf-8")

    with pytest.raises(TypeError, match="Config root must be a mapping"):
        load_config(path)


@pytest.mark.parametrize(
    ("raw", "error_path"),
    [
        pytest.param({"detetor": {"name": "allow_all"}}, "detetor", id="top-level-typo"),
        pytest.param({"detector": {"nam": "allow_all"}}, "detector.nam", id="section-typo"),
        pytest.param(
            {"nsg": {"baseline_detector": {"nam": "allow_all"}}},
            "nsg.baseline_detector.nam",
            id="nested-section-typo",
        ),
    ],
)
def test_load_config_rejects_unknown_keys(
    tmp_path: Path,
    raw: dict,
    error_path: str,
) -> None:
    with pytest.raises(ValueError, match=rf"{error_path}.*unknown"):
        load_config(_write_config(tmp_path, raw))


@pytest.mark.parametrize(
    ("raw", "error_path", "expected_type"),
    [
        pytest.param({"nsg": {"enabled": "false"}}, "nsg.enabled", "bool", id="bool-string"),
        pytest.param({"run": {"max_samples": "1"}}, "run.max_samples", "int", id="int-string"),
        pytest.param({"model": {"temperature": "0.0"}}, "model.temperature", "float", id="float-string"),
        pytest.param({"detector": {"name": ["allow_all"]}}, "detector.name", "str", id="string-list"),
        pytest.param({"content": {"sample_ids": "fx-0001"}}, "content.sample_ids", "list", id="list-string"),
        pytest.param({"judge": {"params": []}}, "judge.params", "dict", id="dict-list"),
    ],
)
def test_load_config_rejects_wrong_value_types(
    tmp_path: Path,
    raw: dict,
    error_path: str,
    expected_type: str,
) -> None:
    with pytest.raises(TypeError, match=rf"{error_path}.*{expected_type}"):
        load_config(_write_config(tmp_path, raw))


@pytest.mark.parametrize(
    ("raw", "error_path", "component_kind"),
    [
        pytest.param(
            {"dataset": {"name": "missing_dataset"}},
            "dataset.name",
            "dataset",
            id="dataset",
        ),
        pytest.param(
            {"attack": {"name": "missing_attack"}},
            "attack.name",
            "attack",
            id="attack",
        ),
        pytest.param(
            {"model": {"backend": "missing_backend"}},
            "model.backend",
            "backend",
            id="backend",
        ),
        pytest.param(
            {"detector": {"name": "missing_detector"}},
            "detector.name",
            "detector",
            id="detector",
        ),
        pytest.param(
            {"judge": {"name": "missing_judge"}},
            "judge.name",
            "judge",
            id="judge",
        ),
        pytest.param(
            {"nsg": {"baseline_detector": {"name": "missing_detector"}}},
            "nsg.baseline_detector.name",
            "detector",
            id="baseline-detector",
        ),
    ],
)
def test_load_config_rejects_unknown_component_names(
    tmp_path: Path,
    raw: dict,
    error_path: str,
    component_kind: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=rf"{error_path}.*unknown {component_kind}",
    ):
        load_config(_write_config(tmp_path, raw))


def test_load_config_accepts_explicit_detector_entrypoints(tmp_path: Path) -> None:
    method_file = tmp_path / "external_method.py"
    method_file.write_text("def build():\n    raise AssertionError('not loaded during validation')\n")

    config = load_config(
        _write_config(
            tmp_path,
            {"detector": {"name": f"{method_file}:build", "params": {"threshold": 0.5}}},
        )
    )

    assert config.detector.name == f"{method_file}:build"


@pytest.mark.parametrize("config_path", sorted(Path("configs/runs").glob("*.yaml")), ids=lambda path: path.name)
def test_all_repository_run_configs_pass_strict_validation(config_path: Path) -> None:
    load_config(config_path)
