from pathlib import Path

from turnkey.components.detectors import available_detectors, load_detector
from turnkey.components.detectors.rcs import RCSPaperPolicy
from turnkey.config import DetectorConfig, load_config


def test_complex_builtins_expose_only_policy_runtime_names() -> None:
    names = set(available_detectors())

    assert {"jailguard", "gradsafe", "rcs"}.isdisjoint(names)
    assert {
        "jailguard_v3",
        "gradsafe_v3",
        "smoothllm_v3",
        "rcs_toy_v3",
        "rcs_paper_v3",
    } <= names


def test_gradsafe_smoke_uses_policy_runtime_name() -> None:
    cfg = load_config(Path("configs/runs/smoke_gradsafe.yaml"))

    assert cfg.detector.name == "gradsafe_v3"


def test_simple_detector_aliases_share_one_runtime_contract() -> None:
    pairs = (
        (DetectorConfig(name="allow_all"), DetectorConfig(name="allow_all_v3")),
        (
            DetectorConfig(name="keyword", params={"keywords": ["blocked"]}),
            DetectorConfig(name="keyword_v3", params={"keywords": ["blocked"]}),
        ),
    )

    for base_config, versioned_config in pairs:
        base_manifest = load_detector(base_config).manifest(name=base_config.name).to_dict()
        versioned_manifest = load_detector(versioned_config).manifest(name=versioned_config.name).to_dict()
        base_manifest.pop("name")
        versioned_manifest.pop("name")

        assert base_manifest == versioned_manifest


def test_rcs_registry_selects_mode_without_class_compatibility_state() -> None:
    toy = load_detector(DetectorConfig(name="rcs_toy_v3", params={"mode": "toy"}))
    paper = load_detector(
        DetectorConfig(
            name="rcs_paper_v3",
            params={
                "mode": "paper",
                "model": {"model_id": "hidden-model", "device": "cpu"},
                "benign_prompts": ["benign 0", "benign 1"],
                "malicious_prompts": ["malicious 0", "malicious 1"],
            },
        )
    )

    assert not isinstance(toy.policy(), RCSPaperPolicy)
    assert isinstance(paper.policy(), RCSPaperPolicy)
    assert not hasattr(toy, "standardized")
    assert not hasattr(paper, "standardized_variant")
