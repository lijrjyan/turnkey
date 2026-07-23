from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from turnkey.cli import main
from turnkey.config import load_config
from turnkey.runner import run_eval


CORE_FILES = {"run.json", "cases.jsonl", "events.jsonl", "metrics.json"}


def _run_smoke(tmp_path: Path) -> Path:
    raw = yaml.safe_load(Path("configs/runs/smoke.yaml").read_text(encoding="utf-8"))
    raw["run"]["out_dir"] = str(tmp_path / "outputs")
    config_path = tmp_path / "smoke.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return run_eval(
        load_config(config_path),
        source_config_path=str(config_path),
        command=["turnkey", "run", "--config", str(config_path)],
    )


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    assert rows and all(isinstance(row, dict) for row in rows)
    return rows


def _identity(path: Path, *, count: int) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": path.name,
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "count": count,
    }


def test_smoke_writes_only_four_authoritative_public_files(tmp_path: Path) -> None:
    run_dir = _run_smoke(tmp_path)

    files = {
        path.relative_to(run_dir).as_posix()
        for path in run_dir.rglob("*")
        if path.is_file()
    }

    assert files == CORE_FILES


def test_run_metadata_links_generic_cases_events_and_metrics(tmp_path: Path) -> None:
    run_dir = _run_smoke(tmp_path)
    run = _load_json(run_dir / "run.json")
    cases = _load_jsonl(run_dir / "cases.jsonl")
    events = _load_jsonl(run_dir / "events.jsonl")
    metrics = _load_json(run_dir / "metrics.json")

    assert run["schema_version"] == "turnkey_run/v1"
    assert metrics["schema_version"] == "turnkey_metrics/v1"
    assert all(case["schema_version"] == "turnkey_case/v1" for case in cases)
    assert all(event["schema_version"] == "turnkey_event/v1" for event in events)
    assert all("reference" in case and "intervention" in case for case in cases)
    assert all("baseline" not in case and "with_detector" not in case for case in cases)
    assert {case["case_id"] for case in cases} == set(run["input"]["case_ids"])
    assert run["artifacts"] == {
        "cases": _identity(run_dir / "cases.jsonl", count=len(cases)),
        "events": _identity(run_dir / "events.jsonl", count=len(events)),
        "metrics": _identity(run_dir / "metrics.json", count=1),
    }


def test_private_cli_isolates_plaintext_from_public_bundle(
    tmp_path: Path,
    capsys,
) -> None:
    raw = yaml.safe_load(Path("configs/runs/smoke.yaml").read_text(encoding="utf-8"))
    raw["run"]["out_dir"] = str(tmp_path / "outputs")
    config_path = tmp_path / "smoke.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    assert main(["run", "--config", str(config_path), "--private"]) == 0
    run_dir = Path(capsys.readouterr().out.strip())
    run = _load_json(run_dir / "run.json")
    public_text = (run_dir / "cases.jsonl").read_text(encoding="utf-8")
    private_text = (run_dir / "private" / "cases.jsonl").read_text(encoding="utf-8")
    private_prompt = _load_jsonl(run_dir / "private" / "cases.jsonl")[0]["prompt"]

    assert run["redaction"]["private_enabled"] is True
    assert run["redaction"]["private_artifact"] == {
        **_identity(run_dir / "private" / "cases.jsonl", count=4),
        "path": "private/cases.jsonl",
    }
    assert private_prompt not in public_text
    assert private_prompt in private_text
    assert "<redacted sha256=" in public_text
