from pathlib import Path
import re

from turnkey.components.detectors import available_detectors, load_detector
from turnkey.components.detectors.rcs import RCSPaperPolicy
from turnkey.config import DetectorConfig, load_config


def test_builtins_expose_only_stable_unversioned_names() -> None:
    names = set(available_detectors())

    assert {
        "allow_all",
        "keyword",
        "jailguard",
        "gradsafe",
        "smoothllm",
        "rcs_toy",
        "rcs",
    } <= names
    assert all(re.search(r"_v[0-9]+$", name) is None for name in names)


def test_gradsafe_smoke_uses_policy_runtime_name() -> None:
    cfg = load_config(Path("configs/runs/smoke_gradsafe.yaml"))

    assert cfg.detector.name == "gradsafe"


def test_simple_detector_names_load_directly() -> None:
    assert load_detector(DetectorConfig(name="allow_all")).manifest().name
    assert load_detector(
        DetectorConfig(name="keyword", params={"keywords": ["blocked"]})
    ).manifest().name


def test_rcs_registry_selects_mode_without_class_compatibility_state() -> None:
    toy = load_detector(DetectorConfig(name="rcs_toy", params={"mode": "toy"}))
    paper = load_detector(
        DetectorConfig(
            name="rcs",
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
