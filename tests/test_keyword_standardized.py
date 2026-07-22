from __future__ import annotations

import json
from pathlib import Path

import yaml

from turnkey.audit import audit_run_dir
from turnkey.config import load_config
from turnkey.components.detectors.keyword import KeywordDetector
from turnkey.runner import run_eval


def test_keyword_manifest_declares_prompt_input() -> None:
    manifest = KeywordDetector(("UNSAFE_PLACEHOLDER",)).manifest(name="keyword_v3")
    data = manifest.to_dict()

    assert data["name"] == "keyword_v3"
    assert data["required_inputs"] == ["sample", "prompt"]
    assert data["reproducibility"] == {"keywords": ["UNSAFE_PLACEHOLDER"]}


def test_keyword_policy_run_preserves_blocked_path_and_audits(tmp_path: Path) -> None:
    raw = yaml.safe_load(Path("configs/runs/smoke_persona.yaml").read_text(encoding="utf-8"))
    raw["run"]["out_dir"] = str(tmp_path / "outputs")
    raw["detector"] = {
        "name": "keyword_v3",
        "params": {"keywords": ["UNSAFE_PLACEHOLDER"]},
    }
    cfg_path = tmp_path / "keyword_v3.yaml"
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    run_dir = run_eval(load_config(cfg_path), source_config_path=str(cfg_path))
    assert audit_run_dir(run_dir) == []

    rows = [
        json.loads(line)
        for line in (run_dir / "cases.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    blocked = [row for row in rows if row["intervention"]["detector"]["block"] is True]
    assert {row["case_id"] for row in blocked} == {"fx-0003", "fx-0004"}
    assert all(row["intervention"]["model"]["executed"] is False for row in blocked)
    assert all(row["intervention"]["judge"]["details"]["blocked"] is True for row in blocked)

    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert run["components"]["intervention"] == {
        "name": "keyword_v3",
        "parameters": {"keywords": ["UNSAFE_PLACEHOLDER"]},
        "source": {"kind": "builtin", "name": "keyword_v3"},
    }
