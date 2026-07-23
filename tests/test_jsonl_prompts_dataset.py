from __future__ import annotations

import json
from pathlib import Path

import pytest

from turnkey.config import DatasetConfig
from turnkey.components.datasets import load_dataset


def test_jsonl_prompts_dataset_loads_samples(tmp_path: Path) -> None:
    path = tmp_path / "pool.jsonl"
    rows = [
        {
            "sample_id": "ref-safe-1",
            "behavior_id": "ref:safe",
            "is_benign": True,
            "prompt": "Tell me how to make a cake step by step.",
        },
        {
            "sample_id": "ref-unsafe-1",
            "behavior_id": "ref:unsafe",
            "is_benign": False,
            "prompt": "UNSAFE_PLACEHOLDER: request redacted.",
            "attack_family": "T1",
            "attack_method": "reference",
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    samples = load_dataset(DatasetConfig(name="jsonl_prompts", params={"path": str(path)}))

    assert [sample.sample_id for sample in samples] == ["ref-safe-1", "ref-unsafe-1"]
    assert samples[0].is_benign is True
    assert samples[1].attack_family == "T1"
    assert samples[1].attack_params["source"] == str(path)


def test_jsonl_prompts_dataset_rejects_missing_label(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"prompt": "hello"}), encoding="utf-8")

    with pytest.raises(ValueError, match="is_benign"):
        load_dataset(DatasetConfig(name="jsonl_prompts", params={"path": str(path)}))
