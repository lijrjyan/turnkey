from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from turnkey.config import ContentConfig, load_config
from turnkey.content import select_content
from turnkey.components.datasets.fixtures_smoke import load_fixtures_smoke_dataset
from turnkey.runner import run_eval


def test_select_content_by_sample_ids_preserves_requested_order() -> None:
    samples = load_fixtures_smoke_dataset()
    selected, report = select_content(
        samples,
        ContentConfig(sample_ids=["fx-0003", "fx-0001"]),
    )

    assert [sample.sample_id for sample in selected] == ["fx-0003", "fx-0001"]
    assert report.input_count == 4
    assert report.selected_count == 2
    assert report.sample_ids == ["fx-0003", "fx-0001"]
    assert report.selection_hash


def test_select_content_by_behavior_ids_shuffle_and_limit_is_deterministic() -> None:
    samples = load_fixtures_smoke_dataset()
    cfg = ContentConfig(
        behavior_ids=[
            "fixtures:benign:greeting",
            "fixtures:benign:science",
            "fixtures:unsafe_placeholder:1",
        ],
        shuffle=True,
        seed=7,
        limit=2,
    )

    selected_a, report_a = select_content(samples, cfg)
    selected_b, report_b = select_content(samples, cfg)

    assert [sample.sample_id for sample in selected_a] == [sample.sample_id for sample in selected_b]
    assert report_a.to_dict() == report_b.to_dict()
    assert len(selected_a) == 2


def test_select_content_invalid_ids_are_actionable() -> None:
    samples = load_fixtures_smoke_dataset()
    with pytest.raises(ValueError, match="content.sample_ids not found"):
        select_content(samples, ContentConfig(sample_ids=["missing"]))
    with pytest.raises(ValueError, match="content.behavior_ids not found"):
        select_content(samples, ContentConfig(behavior_ids=["missing"]))


def test_run_metadata_records_content_selection_before_attack(tmp_path: Path) -> None:
    raw = yaml.safe_load(Path("configs/runs/smoke_persona.yaml").read_text(encoding="utf-8"))
    raw["run"]["out_dir"] = str(tmp_path / "outputs")
    raw["run"].pop("max_samples", None)
    raw["content"] = {
        "sample_ids": ["fx-0003", "fx-0001"],
        "shuffle": False,
        "seed": 0,
        "limit": 2,
    }
    cfg_path = tmp_path / "content_persona.yaml"
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    run_dir = run_eval(load_config(cfg_path), source_config_path=str(cfg_path))

    rows = [
        json.loads(line)
        for line in (run_dir / "cases.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert {row["case_id"] for row in rows} == {"fx-0003", "fx-0001"}
    assert {row["attack_method"] for row in rows} == {"persona"}

    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    content = run["components"]["content"]
    assert content["input_count"] == 4
    assert content["selected_count"] == 2
    assert content["sample_ids"] == ["fx-0003", "fx-0001"]
    assert content["config"]["limit"] == 2
