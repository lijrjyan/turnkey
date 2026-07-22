from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from script_loader import load_script_module


TRACE_SMOKE: ModuleType = load_script_module(Path("scripts/ci/run_trace_smoke.py"))


def _case(*, attack_method: str = "none", tier: str = "T0") -> dict[str, Any]:
    return {
        "case_id": "fx-0001",
        "dataset": {"name": "fixtures_smoke", "version": "v1"},
        "threat": {"attack_method": attack_method, "tier": tier},
        "reference": {"model": {"executed": True}},
        "intervention": {
            "detector": {"block": False},
            "model": {"executed": True},
        },
    }


def _write_run(tmp_path: Path, *, case: dict[str, Any] | None = None) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run.json").write_text(
        json.dumps({"schema_version": "turnkey_run/v1"}),
        encoding="utf-8",
    )
    (run_dir / "cases.jsonl").write_text(
        json.dumps(case or _case()) + "\n",
        encoding="utf-8",
    )
    (run_dir / "events.jsonl").write_text(
        json.dumps(
            {
                "schema_version": "turnkey_event/v1",
                "case_id": "fx-0001",
                "kind": "policy",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(
        json.dumps({"ASR_strict": 0.0, "ORR_benign": 0.0, "NSG_abs": 0.0, "NSG_rel": 0.0}),
        encoding="utf-8",
    )
    return run_dir


def test_trace_smoke_helper_builds_validation_commands() -> None:
    assert TRACE_SMOKE.validation_commands(Path("outputs/run-1")) == [
        ["turnkey", "validate", "outputs/run-1/cases.jsonl"],
        ["turnkey", "audit", "outputs/run-1"],
    ]


def test_trace_smoke_helper_accepts_core_runtime_contract(tmp_path: Path) -> None:
    TRACE_SMOKE.assert_expected_contents(
        Path("configs/runs/smoke.yaml"),
        _write_run(tmp_path),
    )


def test_trace_smoke_helper_rejects_pair_run_without_pair_attack(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, case=_case(tier="T2"))
    with pytest.raises(AssertionError, match="attack_method=pair"):
        TRACE_SMOKE.assert_expected_contents(Path("configs/runs/smoke_pair.yaml"), run_dir)
