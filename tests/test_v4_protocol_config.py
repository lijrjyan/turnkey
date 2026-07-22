from __future__ import annotations

from pathlib import Path

from turnkey.config import load_config
from turnkey.components.detectors import load_detector


def test_v4_qwen_rcs_paper_smoke_config_declares_fixed_protocol() -> None:
    cfg = load_config("configs/runs/v4_qwen3_0_6b_rcs_paper_v3_smoke.yaml")

    assert cfg.run.name == "v4-qwen3-0-6b-rcs-paper-v3-smoke"
    assert cfg.model.backend == "hf"
    assert cfg.model.model_id == "Qwen/Qwen3-0.6B"
    assert cfg.model.device == "cpu"
    assert cfg.dataset.name == "fixtures_smoke"
    assert cfg.content.sample_ids == ["fx-0001", "fx-0003"]
    assert cfg.attack.name == "persona"
    assert cfg.detector.name == "rcs_paper_v3"
    assert cfg.detector.params["model"]["local_files_only"] is True


def test_v4_qwen_rcs_paper_smoke_config_detector_manifest() -> None:
    cfg = load_config(Path("configs/runs/v4_qwen3_0_6b_rcs_paper_v3_smoke.yaml"))
    detector = load_detector(cfg.detector)
    manifest = detector.manifest(name=cfg.detector.name).to_dict()

    assert manifest["name"] == "rcs_paper_v3"
    assert manifest["required_inputs"] == ["sample", "prompt", "images"]
    assert manifest["reproducibility"]["model"]["model_id"] == "Qwen/Qwen3-0.6B"
    assert manifest["reproducibility"]["training_examples"]["count"] == 4


def test_v9_qwen_rcs_reference_profile_smoke_exercises_reference_controls() -> None:
    cfg = load_config(Path("configs/runs/v9_qwen3_0_6b_rcs_qwen_text_reference_profile_smoke.yaml"))
    detector = load_detector(cfg.detector)
    manifest = detector.manifest(name=cfg.detector.name).to_dict()

    assert cfg.run.name == "v9-qwen3-0-6b-rcs-qwen-text-reference-profile-smoke"
    assert cfg.detector.name == "rcs_paper_v3"
    assert cfg.detector.params["layer"] is None
    assert cfg.detector.params["calibrate_threshold"] is True
    assert cfg.detector.params["val_ratio"] == 0.3
    assert cfg.detector.params["require_balanced_train"] is True
    assert cfg.detector.params["model"]["model_family"] == "qwen"
    assert cfg.detector.params["model"]["include_embedding_layer"] is True
    assert manifest["reproducibility"]["model"]["hidden_state_model_family"] == "qwen"
    assert manifest["reproducibility"]["model"]["include_embedding_layer"] is True
    assert manifest["reproducibility"]["model"]["hidden_state_max_length"] == 8192
    assert manifest["reproducibility"]["training_examples"]["count"] == 10
    assert manifest["reproducibility"]["training_examples"]["balanced"] is True
    assert manifest["reproducibility"]["training_examples"]["require_balanced_train"] is True
    assert manifest["reproducibility"]["layer"] is None
    assert manifest["reproducibility"]["calibration"]["calibrate_threshold"] is True


def test_jailguard_paper_openai_config_uses_reference_text_controls() -> None:
    cfg = load_config(Path("configs/runs/jailguard_paper_openai.yaml"))

    assert cfg.run.name == "jailguard-paper-openai"
    assert cfg.dataset.name == "jbb_behaviors"
    assert cfg.dataset.params["harmful_limit"] == 4
    assert cfg.dataset.params["benign_limit"] == 4
    assert cfg.attack.name == "none"
    assert cfg.model.backend == "openai_compat"
    assert cfg.model.model_id == "gpt-3.5-turbo-1106"
    assert cfg.model.base_url == "https://api.openai.com"
    assert cfg.detector.name == "jailguard_v3"
    assert cfg.detector.params["response_mode"] == "backend"
    assert cfg.detector.params["n_variants"] == 8
    assert cfg.detector.params["mutator"] == "PL"
    assert cfg.detector.params["policy_pool"] == ["PI", "TI", "TL"]
    assert cfg.detector.params["policy_probs"] == [0.24, 0.52, 0.24]
    assert cfg.detector.params["char_rate"] == 0.005
    assert cfg.detector.params["threshold"] == 0.02
    assert cfg.detector.params["similarity"] == "spacy"
    assert cfg.detector.params["punctuation_backend"] == "textaugment"
    assert cfg.detector.params["translation_target_langs"] == ["ru", "fr", "de", "el", "id", "it", "ja", "ko", "la", "pl"]
