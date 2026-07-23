from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from turnkey.components.attacks import load_attack
from turnkey.config import AttackConfig
from turnkey.schema import Sample


def test_pair_attack_template_wraps_instruction_and_updates_metadata() -> None:
    attack = load_attack(
        AttackConfig(
            name="pair",
            params={
                "mode": "template",
                "max_queries": 20,
                "attacker_model": "attacker-small",
                "target_model": "target-small",
                "jailbreak_prefix": "PAIR candidate prompt",
                "attack_family": "T2",
            },
        )
    )
    sample = Sample(
        sample_id="s1",
        behavior_id="b1",
        is_benign=True,
        prompt="Explain the water cycle in one sentence.",
        attack_family="T0",
        attack_method="none",
    )

    out = attack.apply(sample)

    assert out.attack_method == "pair"
    assert out.attack_family == "T2"
    assert out.attack_params["base_attack_method"] == "none"
    assert out.attack_params["pair_mode"] == "template"
    assert out.attack_params["pair_max_queries"] == 20
    assert out.attack_params["pair_attacker_model"] == "attacker-small"
    assert out.attack_params["pair_reference_repo"] == "https://github.com/patrickrchao/JailbreakingLLMs"
    assert out.attack_params["pair_reference_commit"] is None
    assert "PAIR candidate prompt" in out.prompt
    assert "Explain the water cycle" in out.prompt


def test_pair_attack_tag_mode_does_not_modify_prompt() -> None:
    attack = load_attack(AttackConfig(name="pair", params={"mode": "tag", "max_queries": 3}))
    sample = Sample(sample_id="s1", behavior_id="b1", is_benign=True, prompt="Hello.")

    out = attack.apply(sample)

    assert out.attack_method == "pair"
    assert out.prompt == "Hello."
    assert out.attack_params["pair_mode"] == "tag"


def test_pair_attack_replay_uses_prompt_map_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "pair_map.jsonl"
    path.write_text(json.dumps({"sample_id": "s1", "adv_prompt": "REPLAYED PAIR PROMPT"}) + "\n", encoding="utf-8")

    attack = load_attack(
        AttackConfig(
            name="pair",
            params={
                "mode": "replay",
                "prompt_map_path": str(path),
                "key_field": "sample_id",
                "reference_commit": "6379ef705a0fc745530f7d895963510c021b496a",
            },
        )
    )
    sample = Sample(sample_id="s1", behavior_id="b1", is_benign=True, prompt="Hello.")

    out = attack.apply(sample)

    assert out.prompt == "REPLAYED PAIR PROMPT"
    assert out.attack_method == "pair"
    assert out.attack_params["pair_reference_commit"] == "6379ef705a0fc745530f7d895963510c021b496a"


def test_pair_attack_replay_strict_missing_raises(tmp_path: Path) -> None:
    path = tmp_path / "pair_map.json"
    path.write_text(json.dumps({"other": "x"}), encoding="utf-8")
    attack = load_attack(AttackConfig(name="pair", params={"mode": "replay", "prompt_map_path": str(path)}))
    sample = Sample(sample_id="s1", behavior_id="b1", is_benign=True, prompt="Hello.")

    with pytest.raises(KeyError):
        attack.apply(sample)


def test_pair_attack_replay_rejects_duplicate_jsonl_keys(tmp_path: Path) -> None:
    path = tmp_path / "pair_map.jsonl"
    path.write_text(
        json.dumps({"sample_id": "s1", "prompt": "first"}) + "\n"
        + json.dumps({"sample_id": "s1", "prompt": "second"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate sample_id='s1'"):
        load_attack(AttackConfig(name="pair", params={"mode": "replay", "prompt_map_path": str(path)}))


def test_pair_attack_replay_rejects_empty_json_prompt(tmp_path: Path) -> None:
    path = tmp_path / "pair_map.json"
    path.write_text(json.dumps({"s1": ""}), encoding="utf-8")

    with pytest.raises(ValueError, match="empty prompt value"):
        load_attack(AttackConfig(name="pair", params={"mode": "replay", "prompt_map_path": str(path)}))


def test_pair_attack_replay_accepts_adversarial_prompt_field(tmp_path: Path) -> None:
    path = tmp_path / "pair_map.jsonl"
    path.write_text(
        json.dumps({"sample_id": "s1", "adversarial_prompt": "ADVERSARIAL FIELD PROMPT"}) + "\n",
        encoding="utf-8",
    )
    attack = load_attack(AttackConfig(name="pair", params={"mode": "replay", "prompt_map_path": str(path)}))
    sample = Sample(sample_id="s1", behavior_id="b1", is_benign=True, prompt="Hello.")

    out = attack.apply(sample)

    assert out.prompt == "ADVERSARIAL FIELD PROMPT"


def test_reproduce_pair_imports_wandb_table_export(tmp_path: Path) -> None:
    export_path = tmp_path / "pair_wandb_table.json"
    export_path.write_text(
        json.dumps(
            {
                "columns": ["prompt", "judge_scores", "iter"],
                "data": [
                    ["candidate one", 1, 1],
                    ["candidate two", 10, 2],
                ],
            }
        ),
        encoding="utf-8",
    )
    out_path = tmp_path / "pair_map.jsonl"

    subprocess.run(
        [
            sys.executable,
            "scripts/reproduce_pair.py",
            "--mode",
            "import-output",
            "--input",
            str(export_path),
            "--out",
            str(out_path),
            "--sample-id",
            "s1",
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    [row] = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
    manifest = json.loads(out_path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    assert row["sample_id"] == "s1"
    assert row["prompt"] == "candidate two"
    assert row["score"] == 10.0
    assert row["reference_commit"] == "6379ef705a0fc745530f7d895963510c021b496a"
    assert manifest["schema_version"] == "turnkey_pair_prompt_map_manifest/v1"
    assert manifest["count"] == 1
    assert manifest["unique_key_count"] == 1


def test_reproduce_pair_imports_all_keys_from_table_export(tmp_path: Path) -> None:
    export_path = tmp_path / "pair_wandb_table.json"
    export_path.write_text(
        json.dumps(
            {
                "columns": ["sample_id", "prompt", "judge_scores", "iter"],
                "data": [
                    ["s1", "s1 low", 1, 1],
                    ["s2", "s2 only", 7, 1],
                    ["s1", "s1 best", 10, 2],
                ],
            }
        ),
        encoding="utf-8",
    )
    out_path = tmp_path / "pair_map.jsonl"

    subprocess.run(
        [
            sys.executable,
            "scripts/reproduce_pair.py",
            "--mode",
            "import-output",
            "--input",
            str(export_path),
            "--out",
            str(out_path),
            "--import-all",
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
    manifest = json.loads(out_path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    assert [(row["sample_id"], row["prompt"], row["score"]) for row in rows] == [
        ("s1", "s1 best", 10.0),
        ("s2", "s2 only", 7.0),
    ]
    assert manifest["count"] == 2
    assert manifest["source_count"] == 1


def test_reproduce_pair_imports_all_keys_against_goal_export(tmp_path: Path) -> None:
    export_path = tmp_path / "pair_wandb_table.json"
    export_path.write_text(
        json.dumps(
            {
                "columns": ["sample_id", "prompt", "judge_scores", "iter"],
                "data": [
                    ["s1", "s1 low", 1, 1],
                    ["s2", "s2 only", 7, 1],
                    ["s1", "s1 best", 10, 2],
                ],
            }
        ),
        encoding="utf-8",
    )
    goal_export = tmp_path / "pair_goals.jsonl"
    goal_export.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "sample_id": "s2",
                        "behavior_id": "b2",
                        "dataset": "fixtures_smoke",
                        "selected_index": 1,
                        "is_benign": False,
                        "goal": "raw goal 2",
                        "goal_sha256": "hash-2",
                    }
                ),
                json.dumps(
                    {
                        "sample_id": "s1",
                        "behavior_id": "b1",
                        "dataset": "fixtures_smoke",
                        "selected_index": 0,
                        "is_benign": True,
                        "goal": "raw goal 1",
                        "goal_sha256": "hash-1",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    out_path = tmp_path / "pair_map.jsonl"

    subprocess.run(
        [
            sys.executable,
            "scripts/reproduce_pair.py",
            "--mode",
            "import-output",
            "--input",
            str(export_path),
            "--out",
            str(out_path),
            "--import-all",
            "--goal-export",
            str(goal_export),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
    manifest = json.loads(out_path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    assert [(row["sample_id"], row["prompt"], row["score"]) for row in rows] == [
        ("s2", "s2 only", 7.0),
        ("s1", "s1 best", 10.0),
    ]
    assert rows[0]["behavior_id"] == "b2"
    assert rows[0]["goal_sha256"] == "hash-2"
    assert "goal" not in rows[0]
    assert rows[0]["goal_export_path"] == str(goal_export)
    assert manifest["goal_export"] == str(goal_export)
    assert manifest["unique_key_count"] == 2


def test_reproduce_pair_import_goal_export_requires_coverage(tmp_path: Path) -> None:
    export_path = tmp_path / "pair_wandb_table.jsonl"
    export_path.write_text(json.dumps({"sample_id": "s1", "prompt": "s1 prompt", "judge_scores": 10}) + "\n", encoding="utf-8")
    goal_export = tmp_path / "pair_goals.jsonl"
    goal_export.write_text(
        json.dumps({"sample_id": "s1", "goal": "raw goal 1"}) + "\n"
        + json.dumps({"sample_id": "s2", "goal": "raw goal 2"}) + "\n",
        encoding="utf-8",
    )
    out_path = tmp_path / "pair_map.jsonl"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/reproduce_pair.py",
            "--mode",
            "import-output",
            "--input",
            str(export_path),
            "--out",
            str(out_path),
            "--import-all",
            "--goal-export",
            str(goal_export),
        ],
        check=False,
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "imported prompt map missing 1/2 exported goals" in completed.stderr


def test_reproduce_pair_imports_batch_directory_against_goal_export(tmp_path: Path) -> None:
    goal_export = tmp_path / "pair_goals.jsonl"
    goal_export.write_text(
        json.dumps({"sample_id": "s2", "behavior_id": "b2", "dataset": "fixtures_smoke", "selected_index": 1, "goal": "raw goal 2", "goal_sha256": "hash-2"})
        + "\n"
        + json.dumps({"sample_id": "s1", "behavior_id": "b1", "dataset": "fixtures_smoke", "selected_index": 0, "goal": "raw goal 1", "goal_sha256": "hash-1"})
        + "\n",
        encoding="utf-8",
    )
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "s1.json").write_text(
        json.dumps(
            {
                "columns": ["prompt", "judge_scores", "iter"],
                "data": [["s1 weak", 1, 1], ["s1 best", 10, 2]],
            }
        ),
        encoding="utf-8",
    )
    (raw_dir / "s2.jsonl").write_text(
        json.dumps({"prompt": "s2 only", "judge_scores": 7, "iter": 1}) + "\n",
        encoding="utf-8",
    )
    out_path = tmp_path / "pair_map.jsonl"

    subprocess.run(
        [
            sys.executable,
            "scripts/reproduce_pair.py",
            "--mode",
            "import-batch-dir",
            "--input-dir",
            str(raw_dir),
            "--goal-export",
            str(goal_export),
            "--out",
            str(out_path),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
    manifest = json.loads(out_path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    assert [(row["sample_id"], row["prompt"], row["score"]) for row in rows] == [
        ("s2", "s2 only", 7.0),
        ("s1", "s1 best", 10.0),
    ]
    assert rows[0]["goal_export_path"] == str(goal_export)
    assert rows[1]["source_path"] == str(raw_dir / "s1.json")
    assert manifest["mode"] == "import-batch-dir"
    assert manifest["source_count"] == 2


def test_reproduce_pair_import_batch_directory_requires_all_keys(tmp_path: Path) -> None:
    goal_export = tmp_path / "pair_goals.jsonl"
    goal_export.write_text(
        json.dumps({"sample_id": "s1", "goal": "raw goal 1"}) + "\n"
        + json.dumps({"sample_id": "s2", "goal": "raw goal 2"}) + "\n",
        encoding="utf-8",
    )
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "s1.jsonl").write_text(json.dumps({"prompt": "s1 prompt", "judge_scores": 10}) + "\n", encoding="utf-8")
    out_path = tmp_path / "pair_map.jsonl"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/reproduce_pair.py",
            "--mode",
            "import-batch-dir",
            "--input-dir",
            str(raw_dir),
            "--goal-export",
            str(goal_export),
            "--out",
            str(out_path),
        ],
        check=False,
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "missing PAIR output files for 1/2 exported goals" in completed.stderr


def test_reproduce_pair_exports_matrix_goals_for_official_pair(tmp_path: Path) -> None:
    spec_path = tmp_path / "pair_matrix.yaml"
    spec_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "turnkey_matrix_spec/v1",
                "name": "pair-export-fixture",
                "defaults": {"content": {"shuffle": False, "limit": 2}},
                "datasets": [{"name": "fixtures_smoke", "params": {"n_samples": 3}}],
                "attacks": [
                    {
                        "name": "pair",
                        "params": {
                            "mode": "replay",
                            "key_field": "sample_id",
                            "target_model": "target-fixture",
                            "attacker_model": "attacker-fixture",
                            "max_queries": 7,
                            "target_str": "Sure, fixture:",
                            "reference_commit": "6379ef705a0fc745530f7d895963510c021b496a",
                        },
                    }
                ],
                "detectors": [{"name": "allow_all", "params": {}}],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    out_path = tmp_path / "pair_goals.jsonl"

    subprocess.run(
        [
            sys.executable,
            "scripts/reproduce_pair.py",
            "--mode",
            "export-goals",
            "--spec",
            str(spec_path),
            "--out",
            str(out_path),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
    manifest = json.loads(out_path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    assert len(rows) == 2
    assert manifest["schema_version"] == "turnkey_pair_goal_export/v1"
    assert manifest["count"] == 2
    assert manifest["target_model"] == "target-fixture"
    assert [row["selected_index"] for row in rows] == [0, 1]
    assert all(row["dataset"] == "fixtures_smoke" for row in rows)
    assert all(row["goal"] and row["goal_sha256"] for row in rows)
    assert {row["target_str"] for row in rows} == {"Sure, fixture:"}


def test_reproduce_pair_writes_reference_batch_from_goal_export(tmp_path: Path) -> None:
    goal_export = tmp_path / "pair_goals.jsonl"
    goal_export.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "sample_id": "s1",
                        "behavior_id": "b1",
                        "dataset": "fixtures_smoke",
                        "selected_index": 0,
                        "is_benign": True,
                        "goal": "Explain the water cycle.",
                        "goal_sha256": "hash-1",
                        "goal_chars": 24,
                        "target_str": "Sure, here is a fixture answer:",
                    }
                ),
                json.dumps(
                    {
                        "sample_id": "s2",
                        "behavior_id": "b2",
                        "dataset": "fixtures_smoke",
                        "selected_index": 1,
                        "is_benign": False,
                        "goal": "Describe a chemistry safety rule.",
                        "goal_sha256": "hash-2",
                        "goal_chars": 31,
                        "target_str": "Sure, here is another fixture answer:",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    out_path = tmp_path / "pair_reference_commands.jsonl"
    script_path = tmp_path / "pair_reference_commands.sh"

    subprocess.run(
        [
            sys.executable,
            "scripts/reproduce_pair.py",
            "--mode",
            "reference-batch",
            "--goal-export",
            str(goal_export),
            "--out",
            str(out_path),
            "--script-out",
            str(script_path),
            "--attacker-model",
            "gpt-4",
            "--target-model",
            "gpt-3.5-turbo",
            "--judge-model",
            "gpt-4",
            "--max-queries",
            "20",
            "--not-jailbreakbench",
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
    manifest = json.loads(out_path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "turnkey_pair_reference_batch/v1"
    assert manifest["count"] == 2
    assert manifest["attacker_model"] == "gpt-4-0125-preview"
    assert manifest["target_model"] == "gpt-3.5-turbo-1106"
    assert rows[0]["sample_id"] == "s1"
    assert rows[0]["goal_sha256"] == "hash-1"
    assert rows[0]["argv"][:6] == [
        "python",
        "main.py",
        "--attack-model",
        "gpt-4-0125-preview",
        "--target-model",
        "gpt-3.5-turbo-1106",
    ]
    assert "--not-jailbreakbench" in rows[0]["argv"]
    assert rows[0]["argv"][rows[0]["argv"].index("--goal") + 1] == "Explain the water cycle."
    script = script_path.read_text(encoding="utf-8")
    assert "goal_sha256=hash-1" in script
    assert "--not-jailbreakbench" in script
