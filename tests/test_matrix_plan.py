from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from turnkey.cli import main
from turnkey.config import load_config
from turnkey.matrix import merge_matrix_results, run_matrix_plan, summarize_matrix_results, write_matrix_plan
from json_fixtures import load_json


def _write_plan(tmp_path: Path, spec_path: Path) -> tuple[Path, dict]:
    plan_path = write_matrix_plan(spec_path=spec_path, out_dir=tmp_path / "matrix-plan")
    return plan_path, load_json(plan_path)


def _planned(plan: dict) -> list[dict[str, Any]]:
    return [entry for entry in plan["entries"] if entry["status"] == "planned"]


def _skipped(plan: dict) -> list[dict[str, Any]]:
    return [entry for entry in plan["entries"] if entry["status"] == "skipped"]


def _field_values(entries: list[dict[str, Any]], *keys: str) -> set[Any]:
    values: set[Any] = set()
    for entry in entries:
        value: Any = entry
        for key in keys:
            value = value[key]
        values.add(value)
    return values


def _write_matrix_spec(path: Path, spec: dict[str, Any]) -> Path:
    path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    return path


def _fixture_dataset(name: str = "fixtures_smoke", *, n_samples: int = 2) -> dict[str, Any]:
    return {"name": name, "params": {"n_samples": n_samples}}


def _keyword_detector(
    *,
    resource_tier: str | None = None,
    keywords: list[str] | None = None,
) -> dict[str, Any]:
    detector: dict[str, Any] = {
        "name": "keyword_v3",
        "params": {"keywords": keywords or ["UNSAFE_PLACEHOLDER"]},
    }
    if resource_tier is not None:
        detector["resource_tier"] = resource_tier
    return detector


def _write_runtime_matrix_spec(
    tmp_path: Path,
    filename: str,
    *,
    name: str,
    datasets: list[dict[str, Any]],
    attacks: list[dict[str, Any]],
    detectors: list[dict[str, Any]],
) -> Path:
    spec_path = tmp_path / filename
    spec = {
        "schema_version": "turnkey_matrix_spec/v1",
        "name": name,
        "defaults": {
            "run": {"out_dir": str(tmp_path / "runs"), "max_samples": 2},
            "content": {"shuffle": False, "limit": 2},
            "model": {"backend": "dummy", "model_id": "dummy-smoke"},
            "judge": {"name": "dummy_refusal", "params": {}},
            "nsg": {"baseline_detector": {"name": "allow_all", "params": {}}},
        },
        "datasets": datasets,
        "attacks": attacks,
        "detectors": detectors,
    }
    return _write_matrix_spec(spec_path, spec)


def test_tiny_v5_matrix_spec_generates_plan_and_configs(tmp_path: Path) -> None:
    plan_path, plan = _write_plan(tmp_path, Path("configs/matrix/tiny_v5.yaml"))

    assert plan["schema_version"] == "turnkey_matrix_plan/v1"
    assert plan["name"] == "v5-tiny"
    assert plan["counts"] == {"entries": 60, "planned": 24, "skipped": 36}
    assert len({entry["id"] for entry in plan["entries"]}) == 60

    planned = _planned(plan)
    skipped = _skipped(plan)
    assert planned
    assert skipped
    assert all("config_path" in entry for entry in planned)
    assert all("config_path" not in entry for entry in skipped)
    assert all(entry["reason"] for entry in skipped)

    entry = planned[0]
    cfg_path = plan_path.parent / entry["config_path"]
    cfg = load_config(cfg_path)
    assert cfg.run.name == entry["id"]
    assert cfg.run.out_dir == "outputs/matrix/v5-tiny/runs"
    assert cfg.content.limit == 5
    assert cfg.content.seed == 20260428
    assert cfg.model.backend == "dummy"
    assert cfg.judge.name == "dummy_refusal"
    assert entry["command"] == ["turnkey", "run", "--config", entry["config_path"]]


def test_sampled_100_v5_matrix_spec_generates_expected_plan(tmp_path: Path) -> None:
    plan_path, plan = _write_plan(tmp_path, Path("configs/matrix/sampled_100_v5.yaml"))

    assert plan["name"] == "v5-sampled-100"
    assert plan["counts"] == {"entries": 168, "planned": 63, "skipped": 105}
    assert len(list((plan_path.parent / "configs").glob("*.yaml"))) == 63
    planned = _planned(plan)
    assert _field_values(planned, "detector", "name") == {
        "allow_all_v3",
        "keyword_v3",
        "rcs_toy_v3",
    }
    assert all(entry["dataset"]["name"] != "sorrybench_202406" for entry in planned)


def test_matrix_plan_preserves_detector_calibration_artifact(tmp_path: Path) -> None:
    spec = {
        "schema_version": "turnkey_matrix_spec/v1",
        "name": "calibrated-smoke",
        "defaults": {
            "run": {"out_dir": str(tmp_path / "runs"), "max_samples": 2},
            "model": {"backend": "dummy", "model_id": "dummy-smoke"},
            "nsg": {"enabled": True, "baseline_detector": {"name": "allow_all", "params": {}}},
            "judge": {"name": "dummy_refusal", "params": {}},
        },
        "datasets": [{"name": "fixtures_smoke", "params": {"n_samples": 2}}],
        "attacks": [{"name": "none", "params": {}}],
        "detectors": [
            {
                "name": "keyword",
                "params": {"keywords": ["UNSAFE_PLACEHOLDER"]},
                "calibration_artifact": "artifacts/keyword-calibration.json",
            }
        ],
    }
    spec_path = _write_matrix_spec(tmp_path / "matrix.yaml", spec)

    plan_path = write_matrix_plan(spec_path=spec_path, out_dir=tmp_path / "plan")
    plan = load_json(plan_path)
    entry = plan["entries"][0]
    assert entry["detector"]["calibration_artifact"] == "artifacts/keyword-calibration.json"
    cfg = load_config(plan_path.parent / entry["config_path"])
    assert cfg.detector.calibration_artifact == "artifacts/keyword-calibration.json"


def test_qwen3_jailguard_medium_v6_matrix_spec_generates_expected_plan(tmp_path: Path) -> None:
    plan_path, plan = _write_plan(tmp_path, Path("configs/matrix/qwen3_0_6b_jailguard_medium_v6.yaml"))

    assert plan["name"] == "v6-qwen3-0-6b-jailguard-medium"
    assert plan["counts"] == {"entries": 28, "planned": 21, "skipped": 7}
    planned = _planned(plan)
    assert _field_values(planned, "detector", "name") == {"jailguard_v3"}
    assert _field_values(planned, "model", "model_id") == {"Qwen/Qwen3-0.6B"}
    assert all(entry["resource_tier"] == "medium" for entry in planned)
    cfg = load_config(plan_path.parent / planned[0]["config_path"])
    assert cfg.model.backend == "hf"
    assert cfg.model.device == "cpu"
    assert cfg.run.max_samples == 5
    assert cfg.detector.params["response_mode"] == "backend"
    assert cfg.detector.params["n_variants"] == 2


def test_qwen3_heavy_pilot_v6_matrix_spec_generates_expected_plan(tmp_path: Path) -> None:
    plan_path, plan = _write_plan(tmp_path, Path("configs/matrix/qwen3_0_6b_heavy_pilot_v6.yaml"))

    assert plan["name"] == "v6-qwen3-0-6b-heavy-pilot"
    assert plan["counts"] == {"entries": 4, "planned": 4, "skipped": 0}
    planned = _planned(plan)
    assert _field_values(planned, "detector", "name") == {"rcs_paper_v3", "gradsafe_v3"}
    assert _field_values(planned, "attack", "name") == {"none", "persona"}
    assert _field_values(planned, "dataset", "name") == {"jbb_behaviors"}
    assert _field_values(planned, "model", "model_id") == {"Qwen/Qwen3-0.6B"}
    assert all(entry["resource_tier"] == "heavy" for entry in planned)

    rcs_entry = next(entry for entry in planned if entry["detector"]["name"] == "rcs_paper_v3")
    rcs_cfg = load_config(plan_path.parent / rcs_entry["config_path"])
    assert rcs_cfg.run.max_samples == 2
    assert rcs_cfg.model.backend == "hf"
    assert rcs_cfg.model.device == "cpu"
    assert rcs_cfg.detector.params["model"]["local_files_only"] is True
    assert rcs_cfg.detector.params["projection_epochs"] == 1

    gradsafe_entry = next(entry for entry in planned if entry["detector"]["name"] == "gradsafe_v3")
    gradsafe_cfg = load_config(plan_path.parent / gradsafe_entry["config_path"])
    assert gradsafe_cfg.detector.params["local_files_only"] is True
    assert gradsafe_cfg.detector.params["max_length"] == 96
    assert gradsafe_cfg.detector.params["parameter_regex"] == "(lm_head|embed_tokens|wte)"


@pytest.mark.parametrize(
    ("spec_path", "name", "counts", "detectors", "resource_tier"),
    [
        (
            Path("configs/matrix/qwen3_0_6b_sampled_100_cheap_v7.yaml"),
            "v7-qwen3-0-6b-sampled-100-cheap",
            {"entries": 84, "planned": 63, "skipped": 21},
            {"allow_all_v3", "keyword_v3", "rcs_toy_v3"},
            "cheap",
        ),
        (
            Path("configs/matrix/qwen3_0_6b_sampled_100_jailguard_v7.yaml"),
            "v7-qwen3-0-6b-sampled-100-jailguard",
            {"entries": 28, "planned": 21, "skipped": 7},
            {"jailguard_v3"},
            "medium",
        ),
        (
            Path("configs/matrix/qwen3_0_6b_sampled_100_heavy_v7.yaml"),
            "v7-qwen3-0-6b-sampled-100-heavy",
            {"entries": 56, "planned": 42, "skipped": 14},
            {"rcs_paper_v3", "gradsafe_v3"},
            "heavy",
        ),
    ],
)
def test_qwen3_sampled_100_v7_matrix_specs_generate_expected_plans(
    tmp_path: Path,
    spec_path: Path,
    name: str,
    counts: dict[str, int],
    detectors: set[str],
    resource_tier: str,
) -> None:
    plan_path, plan = _write_plan(tmp_path, spec_path)

    assert plan["name"] == name
    assert plan["counts"] == counts
    planned = _planned(plan)
    skipped = _skipped(plan)
    assert _field_values(planned, "dataset", "name") == {
        "jbb_behaviors",
        "xstest",
        "sorrybench_public",
    }
    assert _field_values(planned, "attack", "name") == {
        "none",
        "persona",
        "manyshot",
        "deepinception",
        "artprompt",
        "tap",
        "crescendo",
    }
    assert _field_values(planned, "detector", "name") == detectors
    assert _field_values(planned, "model", "model_id") == {"Qwen/Qwen3-0.6B"}
    assert all(entry["resource_tier"] == resource_tier for entry in planned)
    assert skipped
    assert _field_values(skipped, "dataset", "name") == {"sorrybench_202406"}

    cfg = load_config(plan_path.parent / planned[0]["config_path"])
    assert cfg.run.max_samples == 100
    assert cfg.content.limit == 100
    assert cfg.model.backend == "hf"
    assert cfg.model.device == "cuda"
    if name == "v7-qwen3-0-6b-sampled-100-heavy":
        gradsafe_entry = next(entry for entry in planned if entry["detector"]["name"] == "gradsafe_v3")
        gradsafe_cfg = load_config(plan_path.parent / gradsafe_entry["config_path"])
        assert gradsafe_cfg.detector.params["max_length"] == 768


def test_qwen3_bounded_reproduction_v9_matrix_spec_declares_parity_contract(tmp_path: Path) -> None:
    plan_path, plan = _write_plan(tmp_path, Path("configs/matrix/qwen3_0_6b_bounded_reproduction_v9.yaml"))

    assert plan["name"] == "v9-qwen3-0-6b-bounded-reproduction"
    assert plan["counts"] == {"entries": 84, "planned": 84, "skipped": 0}
    planned = _planned(plan)
    assert _field_values(planned, "detector", "name") == {
        "rcs_paper_v3",
        "gradsafe_v3",
        "jailguard_v3",
    }
    assert _field_values(planned, "judge", "name") == {"strongreject"}
    assert _field_values(planned, "model", "model_id") == {"Qwen/Qwen3-0.6B"}
    assert _field_values(planned, "dataset", "name") == {
        "jbb_behaviors",
        "xstest",
        "sorrybench_public",
        "sorrybench_202406",
    }

    rcs_entry = next(entry for entry in planned if entry["detector"]["name"] == "rcs_paper_v3")
    rcs_cfg = load_config(plan_path.parent / rcs_entry["config_path"])
    rcs_detector = rcs_cfg.detector.params
    assert rcs_cfg.reproduction.claim == "bounded_qwen_reproduction"
    assert rcs_cfg.reproduction.main_model == "Qwen/Qwen3-0.6B"
    assert rcs_cfg.judge.name == "strongreject"
    assert rcs_cfg.judge.params["evaluator"] == "strongreject_finetuned"
    assert rcs_cfg.judge.params["reference_commit"] == "7a551d5b440ec7b75d4f6f5bb7c1719965b76b47"
    assert "StrongREJECT evaluator: strongreject_finetuned" in rcs_cfg.reproduction.paper_sources
    assert "LVLM" in str(rcs_cfg.reproduction.reference_target_model)
    assert rcs_detector["calibrate_threshold"] is True
    assert rcs_detector["val_ratio"] == 0.2
    assert rcs_detector["objective_bal_acc_weight"] == 0.8
    assert rcs_detector["objective_f1_weight"] == 0.2
    assert rcs_detector["require_balanced_train"] is True
    assert rcs_detector["method"] == "kcd"
    assert rcs_detector["k"] == 40
    assert rcs_detector["projection_dim"] == 256
    assert rcs_detector["projection_epochs"] == 100
    assert rcs_detector["projection_batch_size"] == 64
    assert rcs_detector["projection_lr"] == 0.001
    assert rcs_detector["projection_beta"] == 5.0
    assert rcs_detector["projection_ms"] == 2.0
    assert rcs_detector["projection_dropout"] == 0.3
    assert rcs_detector["layer"] is None
    assert rcs_detector["auto_layer_svm_iters"] == 200
    assert rcs_detector["model"]["model_id"] == "Qwen/Qwen3-0.6B"
    assert rcs_detector["model"]["model_family"] == "qwen"
    assert rcs_detector["model"]["include_embedding_layer"] is True
    assert rcs_detector["train_jsonl"] == "artifacts/paper/rcs_jbb_behaviors_seed20260428_v9.jsonl"
    assert "UNSAFE_PLACEHOLDER" not in json.dumps(rcs_detector)

    gradsafe_entry = next(entry for entry in planned if entry["detector"]["name"] == "gradsafe_v3")
    gradsafe_cfg = load_config(plan_path.parent / gradsafe_entry["config_path"])
    gradsafe_detector = gradsafe_cfg.detector.params
    assert gradsafe_cfg.reproduction.claim == "bounded_qwen_reproduction"
    assert "unsafe reference gradients" in str(gradsafe_cfg.reproduction.reference_target_model)
    assert gradsafe_detector["model_id"] == "Qwen/Qwen3-0.6B"
    assert gradsafe_detector["prompt_template"] == "llama2_inst"
    assert gradsafe_detector["anchor_response"] == "Sure"
    assert gradsafe_detector["separator_token"] == "<|endoftext|>"
    assert gradsafe_detector["parameter_regex"] == "(mlp|self)"
    assert gradsafe_detector["normalize_by_tokens"] is True
    assert gradsafe_detector["score_mode"] == "gradient_norm"

    jailguard_entry = next(entry for entry in planned if entry["detector"]["name"] == "jailguard_v3")
    jailguard_cfg = load_config(plan_path.parent / jailguard_entry["config_path"])
    jailguard_detector = jailguard_cfg.detector.params
    assert jailguard_cfg.reproduction.claim == "bounded_qwen_reproduction"
    assert "reference victim API model" in str(jailguard_cfg.reproduction.reference_target_model)
    assert jailguard_detector["response_mode"] == "backend"
    assert jailguard_detector["n_variants"] == 8
    assert jailguard_detector["mutator"] == "PL"
    assert jailguard_detector["policy_pool"] == ["PI", "RI", "RD"]
    assert jailguard_detector["char_rate"] == 0.005
    assert jailguard_detector["threshold"] == 0.02
    assert jailguard_detector["similarity"] == "bow"
    assert jailguard_detector["max_new_tokens"] == 128


def test_qwen35_2b_main_v11_matrix_spec_generates_paper_plan(tmp_path: Path) -> None:
    plan_path, plan = _write_plan(tmp_path, Path("configs/matrix/qwen35_2b_main_v11.yaml"))

    assert plan["name"] == "v11-qwen35-2b-main"
    assert plan["counts"] == {"entries": 160, "planned": 160, "skipped": 0}
    planned = _planned(plan)
    assert _field_values(planned, "dataset", "name") == {
        "jbb_behaviors",
        "xstest",
        "sorrybench_public",
        "sorrybench_202406",
    }
    assert _field_values(planned, "attack", "name") == {
        "none",
        "persona",
        "manyshot",
        "deepinception",
        "artprompt",
        "tap",
        "crescendo",
        "pair",
    }
    assert _field_values(planned, "detector", "name") == {
        "allow_all_v3",
        "keyword_v3",
        "rcs_paper_v3",
        "gradsafe_v3",
        "jailguard_v3",
    }
    assert _field_values(planned, "model", "model_id") == {"Qwen/Qwen3.5-2B"}
    assert _field_values(planned, "judge", "name") == {"strongreject"}

    rcs_entry = next(entry for entry in planned if entry["detector"]["name"] == "rcs_paper_v3")
    rcs_cfg = load_config(plan_path.parent / rcs_entry["config_path"])
    assert rcs_cfg.reproduction.claim == "refusal_aware_paper_main"
    assert rcs_cfg.reproduction.main_model == "Qwen/Qwen3.5-2B"
    assert rcs_cfg.judge.name == "strongreject"
    assert rcs_cfg.judge.params["reference_commit"] == "7a551d5b440ec7b75d4f6f5bb7c1719965b76b47"
    assert rcs_cfg.detector.params["train_jsonl"] == "artifacts/paper/rcs_jbb_behaviors_seed20260428_v11.jsonl"
    assert rcs_cfg.detector.params["model"]["model_id"] == "Qwen/Qwen3.5-2B"
    assert rcs_cfg.naturalness.enabled is False
    assert any("Naturalness-controlled analysis is deferred" in note for note in rcs_cfg.reproduction.notes)

    pair_entry = next(entry for entry in planned if entry["attack"]["name"] == "pair")
    pair_cfg = load_config(plan_path.parent / pair_entry["config_path"])
    assert pair_cfg.attack.params["mode"] == "replay"
    assert pair_cfg.attack.params["prompt_map_path"] == "artifacts/paper/pair_prompts_v11.jsonl"
    assert pair_cfg.attack.params["max_queries"] == 20
    assert pair_cfg.attack.params["reference_commit"] == "6379ef705a0fc745530f7d895963510c021b496a"
    assert (
        "PAIR reference repo patrickrchao/JailbreakingLLMs commit "
        "6379ef705a0fc745530f7d895963510c021b496a"
        in pair_cfg.reproduction.paper_sources
    )


def test_qwen25_7b_robustness_v11_matrix_spec_generates_sanity_plan(tmp_path: Path) -> None:
    plan_path, plan = _write_plan(tmp_path, Path("configs/matrix/qwen25_7b_robustness_v11.yaml"))

    assert plan["name"] == "v11-qwen25-7b-robustness"
    assert plan["counts"] == {"entries": 10, "planned": 10, "skipped": 0}
    planned = _planned(plan)
    assert _field_values(planned, "dataset", "name") == {"jbb_behaviors"}
    assert _field_values(planned, "attack", "name") == {"none", "crescendo"}
    assert _field_values(planned, "detector", "name") == {
        "allow_all_v3",
        "keyword_v3",
        "rcs_paper_v3",
        "gradsafe_v3",
        "jailguard_v3",
    }
    assert _field_values(planned, "model", "model_id") == {"Qwen/Qwen2.5-7B-Instruct"}
    assert _field_values(planned, "judge", "name") == {"strongreject"}

    rcs_entry = next(entry for entry in planned if entry["detector"]["name"] == "rcs_paper_v3")
    rcs_cfg = load_config(plan_path.parent / rcs_entry["config_path"])
    assert rcs_cfg.reproduction.claim == "refusal_aware_second_model_sanity"
    assert rcs_cfg.reproduction.main_model == "Qwen/Qwen2.5-7B-Instruct"
    assert rcs_cfg.judge.params["reference_commit"] == "7a551d5b440ec7b75d4f6f5bb7c1719965b76b47"
    assert rcs_cfg.detector.params["model"]["model_id"] == "Qwen/Qwen2.5-7B-Instruct"
    assert rcs_cfg.detector.params["model"]["torch_dtype"] == "bfloat16"
    assert rcs_cfg.naturalness.enabled is False
    assert any("Naturalness-controlled analysis is deferred" in note for note in rcs_cfg.reproduction.notes)

    gradsafe_entry = next(entry for entry in planned if entry["detector"]["name"] == "gradsafe_v3")
    gradsafe_cfg = load_config(plan_path.parent / gradsafe_entry["config_path"])
    assert gradsafe_cfg.detector.params["model_id"] == "Qwen/Qwen2.5-7B-Instruct"
    assert gradsafe_cfg.detector.params["torch_dtype"] == "bfloat16"


def test_qwen35_2b_lofo_v11_matrix_spec_generates_fold_plan(tmp_path: Path) -> None:
    plan_path, plan = _write_plan(tmp_path, Path("configs/matrix/qwen35_2b_lofo_v11.yaml"))

    assert plan["name"] == "v11-qwen35-2b-lofo"
    assert plan["counts"] == {"entries": 35, "planned": 35, "skipped": 0}
    planned = _planned(plan)
    assert _field_values(planned, "dataset", "name") == {"jbb_behaviors"}
    assert _field_values(planned, "attack", "name") == {
        "persona",
        "manyshot",
        "deepinception",
        "artprompt",
        "tap",
        "crescendo",
        "pair",
    }
    assert _field_values(planned, "lofo", "holdout_attack") == {
        "persona",
        "manyshot",
        "deepinception",
        "artprompt",
        "tap",
        "crescendo",
        "pair",
    }

    rcs_pair = next(
        entry
        for entry in planned
        if entry["detector"]["name"] == "rcs_paper_v3" and entry["lofo"]["holdout_attack"] == "pair"
    )
    assert rcs_pair["lofo"]["train_attacks"] == [
        "none",
        "persona",
        "manyshot",
        "deepinception",
        "artprompt",
        "tap",
        "crescendo",
    ]
    assert (
        rcs_pair["detector"]["calibration_artifact"]
        == "artifacts/paper/lofo/rcs_paper_v3_qwen35_2b_excluding_pair.json"
    )
    rcs_cfg = load_config(plan_path.parent / rcs_pair["config_path"])
    assert (
        rcs_cfg.detector.calibration_artifact
        == "artifacts/paper/lofo/rcs_paper_v3_qwen35_2b_excluding_pair.json"
    )
    assert (
        rcs_cfg.detector.params["train_jsonl"]
        == "artifacts/paper/lofo/rcs_jbb_behaviors_seed20260428_v11_excluding_pair.jsonl"
    )
    assert rcs_cfg.naturalness.enabled is False
    assert any("Naturalness-controlled analysis is deferred" in note for note in rcs_cfg.reproduction.notes)
    assert any("LOFO holdout_attack=pair" in note for note in rcs_cfg.reproduction.notes)


def test_matrix_plan_applies_skip_rules(tmp_path: Path) -> None:
    spec_path = _write_matrix_spec(
        tmp_path / "spec.yaml",
        {
            "schema_version": "turnkey_matrix_spec/v1",
            "name": "skip-check",
            "defaults": {
                "run": {"out_dir": "outputs/matrix/skip-check", "max_samples": 1},
                "model": {"backend": "dummy", "model_id": "dummy-smoke"},
                "judge": {"name": "dummy_refusal", "params": {}},
                "nsg": {"baseline_detector": {"name": "allow_all", "params": {}}},
            },
            "datasets": [{"name": "fixtures_smoke", "params": {"n_samples": 1}}],
            "attacks": [{"name": "none", "params": {}}],
            "detectors": [{"name": "keyword_v3", "resource_tier": "cheap", "params": {}}],
            "skip_rules": [{"detector": "keyword_v3", "reason": "covered by another lane"}],
        },
    )

    plan_path = write_matrix_plan(spec_path=spec_path, out_dir=tmp_path / "plan")
    plan = load_json(plan_path)

    assert plan["counts"] == {"entries": 1, "planned": 0, "skipped": 1}
    assert plan["entries"][0]["reason"] == "covered by another lane"
    assert not (plan_path.parent / "configs").exists()


def test_matrix_plan_expands_lofo_folds(tmp_path: Path) -> None:
    spec_path = _write_matrix_spec(
        tmp_path / "lofo.yaml",
        {
            "schema_version": "turnkey_matrix_spec/v1",
            "name": "lofo-check",
            "defaults": {
                "run": {"out_dir": str(tmp_path / "runs"), "max_samples": 1},
                "model": {"backend": "dummy", "model_id": "dummy-smoke"},
                "judge": {"name": "dummy_refusal", "params": {}},
                "nsg": {"baseline_detector": {"name": "allow_all", "params": {}}},
                "reproduction": {"notes": ["base note"]},
            },
            "lofo": {"enabled": True, "holdout_attacks": ["persona", "manyshot"]},
            "datasets": [{"name": "fixtures_smoke", "params": {"n_samples": 1}}],
            "attacks": [
                {"name": "none", "params": {}},
                {"name": "persona", "params": {"persona": "Compliance Auditor", "attack_family": "T2"}},
                {"name": "manyshot", "params": {"n_shots": 2, "attack_family": "T2"}},
            ],
            "detectors": [
                {
                    "name": "keyword_v3",
                    "resource_tier": "cheap",
                    "params": {"keywords": ["{holdout_attack}"]},
                    "calibration_artifact_template": (
                        "artifacts/paper/lofo/{slug_detector}_{slug_holdout_attack}.json"
                    ),
                }
            ],
        },
    )

    plan_path = write_matrix_plan(spec_path=spec_path, out_dir=tmp_path / "plan")
    plan = load_json(plan_path)

    assert plan["counts"] == {"entries": 2, "planned": 2, "skipped": 0}
    assert {entry["lofo"]["holdout_attack"] for entry in plan["entries"]} == {"persona", "manyshot"}
    persona_entry = next(entry for entry in plan["entries"] if entry["attack"]["name"] == "persona")
    assert persona_entry["lofo"]["train_attacks"] == ["none", "manyshot"]
    assert persona_entry["id"].endswith("lofo-persona")
    assert persona_entry["detector"]["calibration_artifact"] == "artifacts/paper/lofo/keyword-v3_persona.json"
    assert persona_entry["detector"]["params"]["keywords"] == ["persona"]

    cfg = load_config(plan_path.parent / persona_entry["config_path"])
    assert cfg.detector.calibration_artifact == "artifacts/paper/lofo/keyword-v3_persona.json"
    assert cfg.detector.params["keywords"] == ["persona"]
    assert any("LOFO holdout_attack=persona" in note for note in cfg.reproduction.notes)


def test_matrix_plan_rejects_unknown_component(tmp_path: Path) -> None:
    spec_path = _write_matrix_spec(
        tmp_path / "bad.yaml",
        {
            "schema_version": "turnkey_matrix_spec/v1",
            "name": "bad",
            "defaults": {
                "model": {"backend": "dummy", "model_id": "dummy-smoke"},
                "judge": {"name": "dummy_refusal", "params": {}},
                "nsg": {"baseline_detector": {"name": "allow_all", "params": {}}},
            },
            "datasets": ["fixtures_smoke"],
            "attacks": ["none"],
            "detectors": ["missing_detector"],
        },
    )

    with pytest.raises(ValueError, match="unknown names"):
        write_matrix_plan(spec_path=spec_path, out_dir=tmp_path / "plan")


def test_matrix_plan_accepts_external_detector_entrypoint(tmp_path: Path) -> None:
    method_file = tmp_path / "external_method.py"
    method_file.write_text(
        "from turnkey.policy import Component, PolicyChain\n"
        "def build(*, threshold):\n"
        "    return Component(name='external', policy=PolicyChain(), "
        "parameters={'threshold': threshold})\n",
        encoding="utf-8",
    )
    entrypoint = f"{method_file}:build"
    spec_path = _write_matrix_spec(
        tmp_path / "external.yaml",
        {
            "schema_version": "turnkey_matrix_spec/v1",
            "name": "external",
            "defaults": {
                "model": {"backend": "dummy", "model_id": "dummy-smoke"},
                "judge": {"name": "dummy_refusal", "params": {}},
                "nsg": {"baseline_detector": {"name": "allow_all", "params": {}}},
            },
            "datasets": ["fixtures_smoke"],
            "attacks": ["none"],
            "detectors": [{"name": entrypoint, "params": {"threshold": 0.5}}],
        },
    )

    plan_path = write_matrix_plan(spec_path=spec_path, out_dir=tmp_path / "plan")
    plan = load_json(plan_path)
    config_path = plan_path.parent / plan["entries"][0]["config_path"]

    config = load_config(config_path)
    assert config.detector.name == entrypoint
    assert config.detector.params == {"threshold": 0.5}


def test_matrix_plan_cli_prints_plan_path(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out_dir = tmp_path / "cli-plan"

    rc = main(["matrix", "plan", "--spec", "configs/matrix/tiny_v5.yaml", "--out", str(out_dir)])

    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out.strip() == str(out_dir / "plan.json")
    assert (out_dir / "plan.json").exists()


def _runner_spec(tmp_path: Path) -> Path:
    return _write_matrix_spec(
        tmp_path / "runner-spec.yaml",
        {
            "schema_version": "turnkey_matrix_spec/v1",
            "name": "runner-check",
            "defaults": {
                "run": {
                    "out_dir": str(tmp_path / "runs"),
                    "redact": True,
                    "max_samples": 2,
                },
                "content": {"shuffle": False, "limit": 2},
                "model": {
                    "backend": "dummy",
                    "model_id": "dummy-smoke",
                    "max_new_tokens": 16,
                    "temperature": 0.0,
                },
                "judge": {"name": "dummy_refusal", "params": {}},
                "nsg": {"baseline_detector": {"name": "allow_all", "params": {}}},
            },
            "datasets": [{"name": "fixtures_smoke", "params": {"n_samples": 2}}],
            "attacks": [{"name": "none", "params": {}}],
            "detectors": [
                {"name": "keyword_v3", "resource_tier": "cheap", "params": {"keywords": ["UNSAFE_PLACEHOLDER"]}},
                {"name": "allow_all_v3", "resource_tier": "cheap", "params": {}},
                {"name": "rcs_paper_v3", "resource_tier": "heavy", "params": {}},
            ],
            "skip_rules": [{"detector": "rcs_paper_v3", "reason": "heavy lane"}],
        },
    )


def test_matrix_run_records_success_skip_and_not_run(tmp_path: Path) -> None:
    plan_path = write_matrix_plan(spec_path=_runner_spec(tmp_path), out_dir=tmp_path / "plan")

    results_path = run_matrix_plan(plan_path=plan_path, max_runs=1)
    results = load_json(results_path)

    assert results["schema_version"] == "turnkey_matrix_results/v1"
    assert results["counts"] == {
        "entries": 3,
        "attempted": 1,
        "success": 1,
        "audit_failed": 0,
        "failed": 0,
        "skipped": 1,
        "not_run": 1,
    }
    success = next(result for result in results["results"] if result["status"] == "success")
    assert success["audit_errors"] == []
    assert Path(success["artifacts"]["metrics_json"]).exists()
    assert Path(success["artifacts"]["run_json"]).exists()
    assert Path(success["artifacts"]["cases_jsonl"]).exists()
    assert Path(success["artifacts"]["events_jsonl"]).exists()
    assert success["metrics"]["counts"]["n_samples"] == 2
    assert any(result["status"] == "skipped" and result["reason"] == "heavy lane" for result in results["results"])
    assert any(result["status"] == "not_run" and result["stage"] == "limit" for result in results["results"])


def test_matrix_run_can_resume_and_summarize(tmp_path: Path) -> None:
    plan_path = write_matrix_plan(spec_path=_runner_spec(tmp_path), out_dir=tmp_path / "plan")

    results_path = run_matrix_plan(plan_path=plan_path, max_runs=1)
    first = load_json(results_path)
    first_success = next(result for result in first["results"] if result["status"] == "success")
    resumed_path = run_matrix_plan(plan_path=plan_path, max_runs=2, resume=True)
    resumed = load_json(resumed_path)

    assert resumed["counts"]["success"] == 2
    resumed_first = next(result for result in resumed["results"] if result["id"] == first_success["id"])
    assert resumed_first["run_dir"] == first_success["run_dir"]

    summary = summarize_matrix_results(results_path=resumed_path)
    assert summary["schema_version"] == "turnkey_matrix_summary/v1"
    assert summary["counts"]["success"] == 2
    assert len(summary["successes"]) == 2
    assert len(summary["skipped"]) == 1


def test_matrix_run_checkpoints_completed_entries_for_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec_path = _write_runtime_matrix_spec(
        tmp_path,
        "checkpoint.yaml",
        name="checkpoint",
        datasets=[_fixture_dataset()],
        attacks=[
            {"name": "none", "params": {}},
            {"name": "persona", "params": {"persona": "Compliance Auditor"}},
        ],
        detectors=[_keyword_detector()],
    )
    plan_path = write_matrix_plan(spec_path=spec_path, out_dir=tmp_path / "plan")
    results_path = tmp_path / "results.json"
    matrix_run = importlib.import_module("turnkey.matrix.run")
    run_entry = matrix_run._run_matrix_entry
    completed_ids: list[str] = []
    interrupted = False

    def interrupt_after_first_entry(**kwargs: Any) -> dict[str, Any]:
        nonlocal interrupted
        if completed_ids and not interrupted:
            interrupted = True
            raise KeyboardInterrupt
        result = run_entry(**kwargs)
        completed_ids.append(result["id"])
        return result

    monkeypatch.setattr(matrix_run, "_run_matrix_entry", interrupt_after_first_entry)

    with pytest.raises(KeyboardInterrupt):
        run_matrix_plan(plan_path=plan_path, out_path=results_path)

    checkpoint = load_json(results_path)
    assert checkpoint["counts"]["entries"] == 1
    assert checkpoint["counts"]["success"] == 1
    first = checkpoint["results"][0]

    resumed_path = run_matrix_plan(plan_path=plan_path, out_path=results_path, resume=True)
    resumed = load_json(resumed_path)
    assert resumed["counts"]["success"] == 2
    resumed_first = next(result for result in resumed["results"] if result["id"] == first["id"])
    assert resumed_first["run_dir"] == first["run_dir"]
    assert not results_path.with_suffix(".json.tmp").exists()


def test_matrix_run_can_exclude_attack_entries(tmp_path: Path) -> None:
    spec_path = _write_runtime_matrix_spec(
        tmp_path,
        "exclude-attack.yaml",
        name="exclude-attack",
        datasets=[_fixture_dataset()],
        attacks=[{"name": "none", "params": {}}, {"name": "persona", "params": {}}],
        detectors=[_keyword_detector()],
    )
    plan_path = write_matrix_plan(spec_path=spec_path, out_dir=tmp_path / "plan")

    results_path = run_matrix_plan(plan_path=plan_path, exclude_attacks={"persona"})
    results = load_json(results_path)

    assert results["counts"] == {
        "entries": 2,
        "attempted": 1,
        "success": 1,
        "audit_failed": 0,
        "failed": 0,
        "skipped": 0,
        "not_run": 1,
    }
    assert results["filters"] == {
        "include_datasets": [],
        "exclude_datasets": [],
        "exclude_attacks": ["persona"],
    }
    persona = next(result for result in results["results"] if result["id"].endswith("-persona-keyword-v3"))
    assert persona["status"] == "not_run"
    assert persona["reason"] == "excluded attack=persona"


def test_matrix_run_can_include_dataset_entries(tmp_path: Path) -> None:
    spec_path = _write_runtime_matrix_spec(
        tmp_path,
        "include-dataset.yaml",
        name="include-dataset",
        datasets=[_fixture_dataset(), _fixture_dataset("fixtures_smoke@v1")],
        attacks=[{"name": "none", "params": {}}],
        detectors=[_keyword_detector()],
    )
    plan_path = write_matrix_plan(spec_path=spec_path, out_dir=tmp_path / "plan")

    results_path = run_matrix_plan(plan_path=plan_path, include_datasets={"fixtures_smoke@v1"})
    results = load_json(results_path)

    assert results["counts"]["success"] == 1
    assert results["counts"]["not_run"] == 1
    assert results["filters"]["include_datasets"] == ["fixtures_smoke@v1"]
    excluded = next(result for result in results["results"] if result["status"] == "not_run")
    assert excluded["reason"] == "excluded dataset=fixtures_smoke"


def test_matrix_results_can_merge_dataset_shards(tmp_path: Path) -> None:
    spec_path = _write_runtime_matrix_spec(
        tmp_path,
        "merge-shards.yaml",
        name="merge-shards",
        datasets=[_fixture_dataset(), _fixture_dataset("fixtures_smoke@v1")],
        attacks=[{"name": "none", "params": {}}],
        detectors=[_keyword_detector()],
    )
    plan_path = write_matrix_plan(spec_path=spec_path, out_dir=tmp_path / "plan")
    shard_a = run_matrix_plan(
        plan_path=plan_path,
        out_path=tmp_path / "shard-a.json",
        include_datasets={"fixtures_smoke"},
    )
    shard_b = run_matrix_plan(
        plan_path=plan_path,
        out_path=tmp_path / "shard-b.json",
        include_datasets={"fixtures_smoke@v1"},
    )

    merged_path = merge_matrix_results(
        plan_path=plan_path,
        results_paths=[shard_a, shard_b],
        out_path=tmp_path / "merged.json",
    )
    merged = load_json(merged_path)

    assert merged["counts"] == {
        "entries": 2,
        "attempted": 2,
        "success": 2,
        "audit_failed": 0,
        "failed": 0,
        "skipped": 0,
        "not_run": 0,
    }
    assert merged["filters"]["include_datasets"] == ["fixtures_smoke", "fixtures_smoke@v1"]
    assert [source["path"] for source in merged["merge_sources"]] == [str(shard_a), str(shard_b)]
    assert all(result["status"] == "success" for result in merged["results"])


def test_matrix_run_preserves_non_t0_threat_tier_metrics(tmp_path: Path) -> None:
    spec_path = _write_runtime_matrix_spec(
        tmp_path,
        "tiered-matrix.yaml",
        name="tiered-matrix",
        datasets=[_fixture_dataset()],
        attacks=[
            {
                "name": "persona",
                "params": {
                    "persona": "Compliance Auditor",
                    "use_markers": True,
                    "attack_family": "T2",
                },
            }
        ],
        detectors=[_keyword_detector(resource_tier="cheap")],
    )
    plan_path = write_matrix_plan(spec_path=spec_path, out_dir=tmp_path / "plan")

    results_path = run_matrix_plan(plan_path=plan_path)
    results = load_json(results_path)

    success = next(result for result in results["results"] if result["status"] == "success")
    metrics = load_json(Path(success["artifacts"]["metrics_json"]))
    rows = [
        json.loads(line)
        for line in Path(success["artifacts"]["cases_jsonl"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert {row["attack_family"] for row in rows} == {"T2"}
    assert {row["threat"]["tier"] for row in rows} == {"T2"}
    assert metrics["attack_family_counts"] == {"T2": 2}
    assert metrics["threat_tier_counts"] == {"T2": 2}


def test_matrix_run_reuses_runtime_resources_across_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import turnkey.runner as runner

    calls = {"backend": 0}
    real_load_backend = runner.load_backend

    def counted_load_backend(cfg):
        calls["backend"] += 1
        return real_load_backend(cfg)

    monkeypatch.setattr(runner, "load_backend", counted_load_backend)

    plan_path = write_matrix_plan(spec_path=_runner_spec(tmp_path), out_dir=tmp_path / "plan")
    results_path = run_matrix_plan(plan_path=plan_path, max_runs=2)
    results = load_json(results_path)

    assert results["counts"]["success"] == 2
    assert calls["backend"] == 1
    assert results["runtime_cache"]["backends"] == 1
    assert results["runtime_cache"]["judges"] == 1
    cache_manifest = results["runtime_cache_manifest"]
    assert cache_manifest["schema_version"] == "turnkey_runtime_cache_manifest/v2"
    assert cache_manifest["counts"] == results["runtime_cache"]
    assert {entry["bucket"] for entry in cache_manifest["entries"]} >= {"backends", "judges"}
    assert all("key" not in entry for entry in cache_manifest["entries"])
    assert all(len(entry["key_sha256"]) == 64 for entry in cache_manifest["entries"])


def test_matrix_run_does_not_cache_detector_resources_for_heavy_entry(tmp_path: Path) -> None:
    spec_path = _write_runtime_matrix_spec(
        tmp_path,
        "heavy-cache-release.yaml",
        name="heavy-cache-release",
        datasets=[_fixture_dataset()],
        attacks=[{"name": "none", "params": {}}],
        detectors=[_keyword_detector(resource_tier="qwen25_7b_heavy")],
    )
    plan_path = write_matrix_plan(spec_path=spec_path, out_dir=tmp_path / "plan")

    results_path = run_matrix_plan(plan_path=plan_path)
    results = load_json(results_path)

    assert results["counts"]["success"] == 1
    assert results["runtime_cache"]["backends"] == 1
    assert results["runtime_cache"]["judges"] == 1
    assert "detectors" not in results["runtime_cache"]
    assert set(results["runtime_cache"]) == {"backends", "judges"}


def test_matrix_run_and_summarize_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    plan_path = write_matrix_plan(spec_path=_runner_spec(tmp_path), out_dir=tmp_path / "plan")
    results_path = tmp_path / "matrix-results.json"

    rc = main(["matrix", "run", "--plan", str(plan_path), "--out", str(results_path), "--max-runs", "1"])
    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out.strip() == str(results_path)
    assert results_path.exists()

    rc = main(["matrix", "summarize", "--results", str(results_path)])
    captured = capsys.readouterr()
    assert rc == 0
    summary = json.loads(captured.out)
    assert summary["counts"]["success"] == 1
