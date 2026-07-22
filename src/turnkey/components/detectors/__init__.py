from __future__ import annotations

from turnkey.config import DetectorConfig
from turnkey.components.detectors.allow_all import AllowAllDetector
from turnkey.components.detectors.base import Detector
from turnkey.components.detectors.gradsafe import GradSafeDetector
from turnkey.components.detectors.jailguard import JailGuardDetector
from turnkey.components.detectors.keyword import KeywordDetector
from turnkey.components.detectors.rcs import RCSDetector
from turnkey.components.detectors.smoothllm import SmoothLLMDetector
from turnkey.registry import DETECTORS, register_detector


@register_detector("allow_all")
def _build_allow_all(_: DetectorConfig) -> Detector:
    return AllowAllDetector()


@register_detector("allow_all_v3")
def _build_allow_all_v3(_: DetectorConfig) -> Detector:
    return AllowAllDetector()


@register_detector("keyword")
def _build_keyword(cfg: DetectorConfig) -> Detector:
    return _build_keyword_detector(cfg)


@register_detector("keyword_v3")
def _build_keyword_v3(cfg: DetectorConfig) -> Detector:
    return _build_keyword_detector(cfg)


def _build_keyword_detector(cfg: DetectorConfig) -> Detector:
    keywords = cfg.params.get("keywords")
    if not isinstance(keywords, list) or not all(isinstance(x, str) for x in keywords):
        raise ValueError("keyword detector requires params.keywords: list[str]")
    return KeywordDetector(tuple(keywords))


@register_detector("jailguard_v3")
def _build_jailguard_v3(cfg: DetectorConfig) -> Detector:
    return JailGuardDetector(**cfg.params)


@register_detector("rcs_toy_v3")
def _build_rcs_toy_v3(cfg: DetectorConfig) -> Detector:
    return _build_rcs_detector(cfg, expected_mode="toy")


@register_detector("rcs_paper_v3")
def _build_rcs_paper_v3(cfg: DetectorConfig) -> Detector:
    return _build_rcs_detector(cfg, expected_mode="paper")


def _build_rcs_detector(cfg: DetectorConfig, *, expected_mode: str) -> Detector:
    params = dict(cfg.params)
    mode = str(params.get("mode", expected_mode)).strip().lower()
    if mode != expected_mode:
        raise ValueError(f"{cfg.name} only supports mode={expected_mode}")
    params["mode"] = expected_mode
    return RCSDetector(**params)


@register_detector("gradsafe_v3")
def _build_gradsafe_v3(cfg: DetectorConfig) -> Detector:
    return GradSafeDetector(**cfg.params)


@register_detector("smoothllm_v3")
def _build_smoothllm_v3(cfg: DetectorConfig) -> Detector:
    return SmoothLLMDetector(**cfg.params)


def available_detectors() -> list[str]:
    return DETECTORS.list()


def load_detector(cfg: DetectorConfig) -> Detector:
    return DETECTORS.get(cfg.name)(cfg)
