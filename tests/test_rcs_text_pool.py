from __future__ import annotations

import json
from pathlib import Path

from turnkey.cli import main
from turnkey.components.detectors.rcs import RCSDetector, _load_train_examples_jsonl, _split_paper_examples


def _jsonl_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_cli_builds_rcs_text_pool_jsonl_and_manifest(tmp_path: Path) -> None:
    benign_path = tmp_path / "alpaca.json"
    malicious_path = tmp_path / "advbench.csv"
    out_path = tmp_path / "rcs_text_pool.jsonl"
    manifest_path = tmp_path / "rcs_text_pool_manifest.json"

    benign_path.write_text(
        json.dumps(
            [
                {"instruction": "Explain photosynthesis.", "input": "Use one sentence."},
                {"instruction": "Write a polite greeting."},
                {"instruction": "Summarize gravity."},
                {"instruction": "Describe a safe science experiment."},
            ]
        ),
        encoding="utf-8",
    )
    malicious_path.write_text(
        "prompt\n"
        "UNSAFE_PLACEHOLDER request zero\n"
        "UNSAFE_PLACEHOLDER request one\n"
        "UNSAFE_PLACEHOLDER request two\n"
        "UNSAFE_PLACEHOLDER request three\n",
        encoding="utf-8",
    )

    rc = main(
        [
            "data",
            "build-rcs-text-pool",
            "--benign-source",
            f"alpaca={benign_path}",
            "--malicious-source",
            f"advbench={malicious_path}",
            "--out",
            str(out_path),
            "--manifest",
            str(manifest_path),
            "--seed",
            "7",
            "--val-ratio",
            "0.25",
        ]
    )

    assert rc == 0
    rows = _jsonl_rows(out_path)
    assert len(rows) == 8
    assert {row["dataset"] for row in rows} == {"alpaca", "advbench"}
    assert sum(1 for row in rows if row["is_benign"]) == 4
    assert sum(1 for row in rows if not row["is_benign"]) == 4
    assert sum(1 for row in rows if row["split"] == "validation" and row["is_benign"]) == 1
    assert sum(1 for row in rows if row["split"] == "validation" and not row["is_benign"]) == 1
    assert all("prompt_sha256" in row and "prompt_chars" in row for row in rows)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "turnkey_rcs_text_pool_manifest/v1"
    assert manifest["paper_strict"] is False
    assert manifest["counts"] == {"count": 8, "benign_count": 4, "malicious_count": 4}
    assert manifest["split_counts"] == [
        {"split": "train", "count": 6, "benign_count": 3, "malicious_count": 3},
        {"split": "validation", "count": 2, "benign_count": 1, "malicious_count": 1},
    ]
    assert "Explain photosynthesis" not in manifest_path.read_text(encoding="utf-8")

    det = RCSDetector(
        mode="paper",
        method="kcd",
        model={"model_id": "fake-hidden-model", "device": "cpu", "local_files_only": True},
        train_jsonl=str(out_path),
        require_balanced_train=True,
    )
    training = det.manifest(name="rcs_paper_v3").to_dict()["reproducibility"]["training_examples"]
    assert training["split_counts"] == manifest["split_counts"]


def test_rcs_train_jsonl_explicit_split_overrides_ratio(tmp_path: Path) -> None:
    train_jsonl = tmp_path / "rcs_train_explicit_split.jsonl"
    train_jsonl.write_text(
        "\n".join(
            [
                json.dumps({"prompt": "benign train a", "is_benign": True, "dataset": "alpaca", "split": "train"}),
                json.dumps({"prompt": "benign train b", "is_benign": True, "dataset": "alpaca", "split": "train"}),
                json.dumps(
                    {"prompt": "benign validation", "is_benign": True, "dataset": "alpaca", "split": "validation"}
                ),
                json.dumps(
                    {"prompt": "malicious train a", "is_benign": False, "dataset": "advbench", "split": "train"}
                ),
                json.dumps(
                    {"prompt": "malicious train b", "is_benign": False, "dataset": "advbench", "split": "train"}
                ),
                json.dumps(
                    {
                        "prompt": "malicious validation",
                        "is_benign": False,
                        "dataset": "advbench",
                        "split": "validation",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    examples = _load_train_examples_jsonl(str(train_jsonl))
    train, validation = _split_paper_examples(examples, val_ratio=0.0, seed=123)

    assert len(train) == 4
    assert len(validation) == 2
    assert {ex.prompt for ex in validation} == {"benign validation", "malicious validation"}
