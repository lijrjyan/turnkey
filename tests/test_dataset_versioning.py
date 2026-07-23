from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from turnkey.cli import main
from turnkey.config import DatasetConfig, dataset_identity, load_config
from turnkey.components.datasets import available_datasets, load_dataset
from turnkey.runner import run_eval
from turnkey.validation import validate_cases_jsonl


def _run_smoke(tmp_path: Path, *, dataset_name: str = "fixtures_smoke@v1") -> Path:
    raw = yaml.safe_load(Path("configs/runs/smoke.yaml").read_text(encoding="utf-8"))
    raw["run"]["out_dir"] = str(tmp_path / "outputs")
    raw["dataset"]["name"] = dataset_name
    cfg_path = tmp_path / "smoke.yaml"
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return run_eval(load_config(cfg_path), source_config_path=str(cfg_path))


def _load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_explicit_versioned_dataset_aliases_are_registered() -> None:
    names = available_datasets()

    assert "fixtures_smoke@v1" in names
    assert "jbb_behaviors@v1" in names
    assert "sorrybench_202406@v2" in names
    assert dataset_identity("sorrybench_202406@v2").versioned_name == "sorrybench_202406@v2"


def test_unversioned_dataset_defaults_to_v1_with_deprecation_warning() -> None:
    with pytest.warns(DeprecationWarning, match="fixtures_smoke@v1"):
        samples = load_dataset(DatasetConfig(name="fixtures_smoke", params={"n_samples": 1}))

    assert [sample.sample_id for sample in samples] == ["fx-0001"]


def test_cases_record_dataset_version_metadata(tmp_path: Path) -> None:
    run_dir = _run_smoke(tmp_path)
    cases_path = run_dir / "cases.jsonl"
    rows = _load_rows(cases_path)
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    manifest = run["input"]["manifest"]

    assert {row["dataset"]["name"] for row in rows} == {"fixtures_smoke"}
    assert {row["dataset"]["version"] for row in rows} == {"v1"}
    assert manifest["dataset"]["versioned_name"] == "fixtures_smoke@v1"
    assert run["components"]["dataset"]["versioned_name"] == "fixtures_smoke@v1"
    assert validate_cases_jsonl(cases_path) == []


def test_validate_rejects_mixed_dataset_versions(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run_dir = _run_smoke(tmp_path)
    cases_path = run_dir / "cases.jsonl"
    rows = _load_rows(cases_path)
    rows[-1]["dataset"]["version"] = "v2"
    _write_rows(cases_path, rows)

    errors = validate_cases_jsonl(cases_path)
    assert any("mixed dataset identities" in error for error in errors)

    assert main(["validate", str(cases_path)]) == 1
    captured = capsys.readouterr()
    assert "mixed dataset identities" in captured.err
