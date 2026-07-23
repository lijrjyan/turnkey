from __future__ import annotations

from dataclasses import dataclass, field, fields

import pytest

from turnkey.components.detectors.allow_all import AllowAllDetector
from turnkey.components.detectors.base import Detector
from turnkey.components.detectors.gradsafe import GradSafeDetector
from turnkey.config import Config, DetectorConfig
from turnkey.pipeline_states import (
    STATE_ATTACK_METADATA,
    STATE_IMAGES,
    STATE_INPUT_MANIFEST,
    STATE_PREFIX_LOGPROBS,
    STATE_PROMPT,
    STATE_PROMPT_LOGPROBS,
    STATE_SAMPLE,
    registered_state_names,
)
from turnkey.signals import SignalRequest
from turnkey.schema import DetectorDecision, Sample
from turnkey.runner.run import _effective_config_for_components


@dataclass
class _MutableParameterDetector(Detector):
    labels: list[str]

    def decide(self, sample: Sample) -> DetectorDecision:  # noqa: ARG002
        return DetectorDecision(block=False)


@dataclass
class _NonJsonParameterDetector(Detector):
    value: object

    def decide(self, sample: Sample) -> DetectorDecision:  # noqa: ARG002
        return DetectorDecision(block=False)


@dataclass
class _SensitiveParameterDetector(Detector):
    api_key: str = field(metadata={"component_parameter": False})
    threshold: float = 0.5

    def decide(self, sample: Sample) -> DetectorDecision:  # noqa: ARG002
        return DetectorDecision(block=False)


@dataclass
class _PolicyMutatingDetector(Detector):
    threshold: float = 0.5

    def decide(self, sample: Sample) -> DetectorDecision:  # noqa: ARG002
        return DetectorDecision(block=False)

    def policy(self, *, calibration_artifact=None):  # noqa: ANN001, ANN201, ARG002
        self.threshold = 0.9
        return super().policy()


def test_detector_adapter_does_not_declare_runtime_requirements() -> None:
    assert not hasattr(AllowAllDetector(), "requirements")


def test_detector_component_contains_effective_parameters_and_runtime_parts() -> None:
    detector = GradSafeDetector(model_id="gradient-model")

    component = detector.component(name="gradsafe")

    assert component.name == "gradsafe"
    assert component.parameters["model_id"] == "gradient-model"
    assert component.parameters["score_mode"] == "gradient_norm"
    assert component.parameters["prompt_template"] == "simple_chat"
    assert component.policy is not None
    assert len(component.providers) == 1


def test_detector_component_copies_effective_parameters() -> None:
    detector = _MutableParameterDetector(labels=["initial"])

    component = detector.component(name="mutable")
    detector.labels.append("later")

    assert component.parameters == {"labels": ["initial"]}


def test_detector_component_rejects_non_json_parameters() -> None:
    detector = _NonJsonParameterDetector(value=object())

    with pytest.raises(TypeError, match=r"value.*JSON-serializable"):
        detector.component(name="non-json")


def test_detector_component_rejects_nested_non_string_mapping_keys() -> None:
    detector = _NonJsonParameterDetector(value={"nested": {1: "integer", "1": "string"}})

    with pytest.raises(TypeError, match=r"value.*string keys"):
        detector.component(name="non-string-key")


def test_detector_component_excludes_opted_out_parameters() -> None:
    detector = _SensitiveParameterDetector(api_key="secret")

    component = detector.component(name="sensitive")

    assert component.parameters == {"threshold": 0.5}


def test_detector_component_snapshots_parameters_before_policy_construction() -> None:
    detector = _PolicyMutatingDetector()

    component = detector.component(name="mutating", calibration_artifact=object())

    assert component.parameters == {"threshold": 0.5}
    assert detector.threshold == 0.9


def test_effective_config_excludes_opted_out_component_parameters() -> None:
    baseline = AllowAllDetector().component(name="allow_all")
    detector = _SensitiveParameterDetector(
        api_key="secret",
        threshold=0.75,
    ).component(name="sensitive")
    cfg = Config(
        detector=DetectorConfig(
            name="sensitive",
            params={"api_key": "secret", "threshold": 0.75},
        )
    )

    effective = _effective_config_for_components(
        cfg=cfg,
        baseline_component=baseline,
        detector_component=detector,
    )

    assert effective["detector"]["params"] == {"threshold": 0.75}
    assert "secret" not in str(effective)


def test_signal_request_only_exposes_materialized_model_signals() -> None:
    assert [item.name for item in fields(SignalRequest)] == [
        "prompt_logprobs",
        "prefix_logprob_text",
    ]
    request = SignalRequest(prompt_logprobs=True, prefix_logprob_text=" probe")
    assert request.requested_names() == (STATE_PROMPT_LOGPROBS, STATE_PREFIX_LOGPROBS)


def test_pipeline_state_registry_only_contains_reachable_inputs_and_signals() -> None:
    assert registered_state_names() == {
        STATE_SAMPLE,
        STATE_PROMPT,
        STATE_IMAGES,
        STATE_ATTACK_METADATA,
        STATE_INPUT_MANIFEST,
        STATE_PROMPT_LOGPROBS,
        STATE_PREFIX_LOGPROBS,
    }
