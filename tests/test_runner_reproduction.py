from __future__ import annotations

import json
from pathlib import Path

import yaml

from turnkey.audit import audit_run_dir
from turnkey.config import load_config
from turnkey.runner import run_eval


def test_run_receipt_records_reproduction_metadata(tmp_path: Path) -> None:
    cfg_path = tmp_path / "bounded-reproduction.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "run": {
                    "name": "bounded-reproduction-metadata",
                    "out_dir": str(tmp_path / "outputs"),
                    "redact": True,
                    "max_samples": 1,
                },
                "dataset": {"name": "fixtures_smoke", "params": {"n_samples": 2}},
                "content": {"sample_ids": ["fx-0001"], "shuffle": False, "limit": 1},
                "attack": {"name": "none", "params": {}},
                "model": {
                    "backend": "dummy",
                    "model_id": "dummy-smoke",
                    "max_new_tokens": 8,
                    "temperature": 0.0,
                },
                "nsg": {"baseline_detector": {"name": "allow_all", "params": {}}},
                "detector": {"name": "keyword", "params": {"keywords": ["UNSAFE_PLACEHOLDER"]}},
                "judge": {"name": "dummy_refusal", "params": {}},
                "reproduction": {
                    "claim": "bounded_qwen_reproduction",
                    "setting": "unit_test",
                    "main_model": "Qwen/Qwen3-0.6B",
                    "reference_target_model": "official target differs",
                    "paper_sources": ["paper:example"],
                    "reference_repos": ["https://example.invalid/reference"],
                    "setting_differences": ["uses dummy backend only for metadata test"],
                    "notes": ["receipt contract test"],
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    cfg = load_config(cfg_path)
    run_dir = run_eval(cfg, source_config_path=str(cfg_path))

    assert audit_run_dir(run_dir) == []
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert run["reproduction"] == {
        "schema_version": "turnkey_reproduction/v1",
        "claim": "bounded_qwen_reproduction",
        "setting": "unit_test",
        "main_model": "Qwen/Qwen3-0.6B",
        "reference_target_model": "official target differs",
        "paper_sources": ["paper:example"],
        "reference_repos": ["https://example.invalid/reference"],
        "setting_differences": ["uses dummy backend only for metadata test"],
        "notes": ["receipt contract test"],
    }
    assert run["config"]["reproduction"]["claim"] == "bounded_qwen_reproduction"
