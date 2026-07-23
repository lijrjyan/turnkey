from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any

import pytest
import yaml

from turnkey.audit import audit_run_dir
from turnkey.config import load_config
from turnkey.runner import run_eval
from json_fixtures import load_json, load_jsonl, write_json


def _persona_config(
    tmp_path: Path,
    *,
    run_name: str,
    input_manifest_path: Path | None = None,
    persona: str = "Helpful Teacher",
    unsafe_log_plaintext: bool = False,
) -> Path:
    raw = yaml.safe_load(Path("configs/runs/smoke_persona.yaml").read_text(encoding="utf-8"))
    raw["run"]["name"] = run_name
    raw["run"]["out_dir"] = str(tmp_path / "outputs")
    if input_manifest_path is not None:
        raw["run"]["input_manifest_path"] = str(input_manifest_path)
    if unsafe_log_plaintext:
        raw["run"]["unsafe_log_plaintext"] = True
        raw["run"]["redact"] = False
    raw["attack"]["params"]["persona"] = persona
    cfg_path = tmp_path / f"{run_name}.yaml"
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return cfg_path


def _mm_config(tmp_path: Path, *, run_name: str) -> Path:
    raw = yaml.safe_load(Path("configs/runs/smoke_mm.yaml").read_text(encoding="utf-8"))
    raw["run"]["name"] = run_name
    raw["run"]["out_dir"] = str(tmp_path / "outputs")
    raw["run"]["max_samples"] = 1
    cfg_path = tmp_path / f"{run_name}.yaml"
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return cfg_path


def test_run_nests_one_redacted_input_manifest(tmp_path: Path) -> None:
    cfg_path = _persona_config(tmp_path, run_name="input-manifest")

    run_dir = run_eval(load_config(cfg_path), source_config_path=str(cfg_path))

    assert audit_run_dir(run_dir) == []
    run = load_json(run_dir / "run.json")
    manifest = run["input"]["manifest"]
    cases = load_jsonl(run_dir / "cases.jsonl")

    assert run["input"]["mode"] == "generated"
    assert manifest["redaction"]["contains_plaintext"] is False
    assert not _contains_prompt_text(manifest)
    assert manifest["sample_ids"] == run["input"]["case_ids"]
    assert manifest["count"] == len(cases)

    case_by_id = {case["case_id"]: case for case in cases}
    for sample in manifest["samples"]:
        case = case_by_id[sample["sample_id"]]
        assert sample["attacked"]["prompt"]["sha256"] == case["prompt"]["sha256"]
        assert sample["attacked"]["prompt"]["chars"] == case["prompt"]["chars"]


def test_run_can_verify_same_input_manifest_identity(tmp_path: Path) -> None:
    first_cfg = _persona_config(tmp_path, run_name="input-manifest-source")
    first_run_dir = run_eval(load_config(first_cfg), source_config_path=str(first_cfg))
    manifest_path = first_run_dir / "run.json"

    second_cfg = _persona_config(
        tmp_path,
        run_name="input-manifest-reuse",
        input_manifest_path=manifest_path,
    )
    second_run_dir = run_eval(load_config(second_cfg), source_config_path=str(second_cfg))

    assert audit_run_dir(second_run_dir) == []
    run = load_json(second_run_dir / "run.json")
    assert run["input"]["mode"] == "verified"
    assert run["input"]["expected_manifest"]["path"] == str(manifest_path)
    assert len(run["input"]["expected_manifest"]["sha256"]) == 64


def test_audit_does_not_depend_on_source_expected_manifest_path(tmp_path: Path) -> None:
    first_cfg = _persona_config(tmp_path, run_name="input-manifest-source-path")
    first_run_dir = run_eval(load_config(first_cfg), source_config_path=str(first_cfg))
    manifest_path = first_run_dir / "run.json"

    second_cfg = _persona_config(
        tmp_path,
        run_name="input-manifest-reuse-after-source-removal",
        input_manifest_path=manifest_path,
    )
    second_run_dir = run_eval(load_config(second_cfg), source_config_path=str(second_cfg))
    shutil.rmtree(first_run_dir)

    assert audit_run_dir(second_run_dir) == []


def test_run_rejects_mismatched_input_manifest_identity(tmp_path: Path) -> None:
    first_cfg = _persona_config(tmp_path, run_name="input-manifest-expected")
    first_run_dir = run_eval(load_config(first_cfg), source_config_path=str(first_cfg))

    second_cfg = _persona_config(
        tmp_path,
        run_name="input-manifest-mismatch",
        input_manifest_path=first_run_dir / "run.json",
        persona="Different Persona",
    )

    with pytest.raises(ValueError, match="input manifest verification failed"):
        run_eval(load_config(second_cfg), source_config_path=str(second_cfg))


def test_input_manifest_records_image_identity_for_input_provider(tmp_path: Path) -> None:
    cfg_path = _mm_config(tmp_path, run_name="input-manifest-mm")

    run_dir = run_eval(load_config(cfg_path), source_config_path=str(cfg_path))

    assert audit_run_dir(run_dir) == []
    manifest = load_json(run_dir / "run.json")["input"]["manifest"]
    sample = manifest["samples"][0]
    source_image = sample["source"]["images"][0]
    attacked_image = sample["attacked"]["images"][0]

    assert source_image["mime_type"] == "image/x-portable-pixmap"
    assert len(source_image["sha256"]) == 64
    assert source_image["bytes"] > 0
    assert attacked_image["sha256"] == source_image["sha256"]


def test_audit_fails_on_input_manifest_hash_or_prompt_drift(tmp_path: Path) -> None:
    cfg_path = _persona_config(tmp_path, run_name="input-manifest-drift")
    run_dir = run_eval(load_config(cfg_path), source_config_path=str(cfg_path))
    run_path = run_dir / "run.json"
    run = load_json(run_path)
    run["input"]["manifest"]["samples"][0]["attacked"]["prompt"]["sha256"] = "bad-hash"
    write_json(run_path, run)

    errors = audit_run_dir(run_dir)
    assert any("samples[0].attacked.prompt" in error for error in errors)


def test_audit_fails_on_input_case_id_drift(tmp_path: Path) -> None:
    cfg_path = _persona_config(tmp_path, run_name="input-provider-drift")
    run_dir = run_eval(load_config(cfg_path), source_config_path=str(cfg_path))
    run_path = run_dir / "run.json"
    run = load_json(run_path)
    run["input"]["case_ids"][0] = "missing-case"
    write_json(run_path, run)

    errors = audit_run_dir(run_dir)

    assert any("input.case_ids" in error for error in errors)


def _contains_prompt_text(value: Any) -> bool:
    if isinstance(value, dict):
        if "text" in value:
            return True
        return any(_contains_prompt_text(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_prompt_text(item) for item in value)
    return False
