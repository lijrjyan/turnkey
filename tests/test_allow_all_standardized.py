from __future__ import annotations

import json
from pathlib import Path

import yaml

from turnkey.audit import audit_run_dir
from turnkey.config import load_config
from turnkey.components.detectors.allow_all import AllowAllDetector
from turnkey.runner import run_eval


def test_allow_all_manifest_declares_sample_input() -> None:
    manifest = AllowAllDetector().manifest(name="allow_all")
    data = manifest.to_dict()

    assert data["name"] == "allow_all"
    assert data["required_inputs"] == ["sample"]


def test_allow_all_policy_run_persists_component_and_audits(tmp_path: Path) -> None:
    raw = yaml.safe_load(Path("configs/runs/smoke.yaml").read_text(encoding="utf-8"))
    raw["run"]["out_dir"] = str(tmp_path / "outputs")
    raw["detector"] = {"name": "allow_all", "params": {}}
    cfg_path = tmp_path / "allow_all.yaml"
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    run_dir = run_eval(load_config(cfg_path), source_config_path=str(cfg_path))
    assert audit_run_dir(run_dir) == []

    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert run["components"]["reference"]["name"] == "allow_all"
    assert run["components"]["intervention"] == {
        "name": "allow_all",
        "parameters": {},
        "source": {"kind": "builtin", "name": "allow_all"},
    }
