from __future__ import annotations

import json
from pathlib import Path

import yaml

from turnkey.config import load_config
from turnkey.runner import run_eval
from turnkey.schema import Sample
from turnkey.threat import budget_for_sample, normalize_threat_tier, threat_context_for_sample


def test_threat_and_budget_normalization() -> None:
    sample = Sample(
        sample_id="s1",
        behavior_id="b1",
        is_benign=False,
        prompt="placeholder",
        attack_family="t2",
        attack_method="manyshot",
        attack_params={"manyshot_n_shots": 8},
    )

    assert normalize_threat_tier(sample.attack_family) == "T2"
    assert threat_context_for_sample(sample).tier == "T2"
    budget = budget_for_sample(sample)
    assert budget.shots == 8
    assert budget.bucket == "medium"


def test_tiered_smoke_writes_threat_budget_and_group_metrics(tmp_path: Path) -> None:
    raw = yaml.safe_load(Path("configs/runs/tiered_smoke.yaml").read_text(encoding="utf-8"))
    raw["run"]["out_dir"] = str(tmp_path / "outputs")
    cfg_path = tmp_path / "tiered_smoke.yaml"
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    run_dir = run_eval(load_config(cfg_path))
    rows = [
        json.loads(line)
        for line in (run_dir / "cases.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows
    assert {row["threat"]["tier"] for row in rows} == {"T2"}
    assert {row["budget"]["shots"] for row in rows} == {8}
    assert {row["budget"]["bucket"] for row in rows} == {"medium"}

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["threat_tier_counts"] == {"T2": len(rows)}
    assert metrics["budget_bucket_counts"] == {"medium": len(rows)}
    assert metrics["attack_family_counts"] == {"T2": len(rows)}
    expected_split = {
        "benign": sum(1 for row in rows if row["is_benign"] is True),
        "harmful": sum(1 for row in rows if row["is_benign"] is False),
    }
    assert metrics["benign_harmful_counts"] == expected_split
    assert set(metrics["detector_decision_counts"]) <= {"allowed", "blocked"}
    assert "T2" in metrics["groups"]["threat_tier"]
    assert "medium" in metrics["groups"]["budget_bucket"]
    assert "T2" in metrics["groups"]["attack_family"]
    assert "harmful" in metrics["groups"]["benign_harmful"]
    assert set(metrics["groups"]["detector_decision"]) <= {"allowed", "blocked"}
