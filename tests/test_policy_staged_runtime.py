from dataclasses import dataclass

import pytest

from turnkey.components.backends.base import LLMBackend
from turnkey.components.detectors.gradsafe import GradSafeDetector
from turnkey.components.judges.dummy import DummyRefusalJudge
from turnkey.config import Config, ModelConfig, NaturalnessConfig
from turnkey.methods import Request
from turnkey.policy import Component, PolicyChain
from turnkey.runtime_providers import (
    GradientScoreRequest,
    GradientScoreResult,
    ProviderSummary,
)
from turnkey.runner.policy_executor import _attach_secondary_failure, run_policy_pair
from turnkey.schema import ModelOutput, Sample


@dataclass
class _State:
    target_active: bool = True
    gradient_active: bool = False

    def __post_init__(self) -> None:
        self.events: list[str] = []


@dataclass
class _TargetBackend(LLMBackend):
    state: _State

    def generate(
        self,
        *,
        prompt: str,
        images=None,
        max_new_tokens: int,
        temperature: float,
    ) -> ModelOutput:  # noqa: ARG002
        assert self.state.target_active is True
        assert self.state.gradient_active is False
        self.state.events.append(f"target:{prompt}")
        return ModelOutput(
            executed=True,
            backend="staged-test",
            model_id="target",
            response_text="OK",
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
            latency_s=0.01,
        )


@dataclass
class _GradientProvider:
    state: _State
    request_type = GradientScoreRequest
    model_forwards_per_call = 1

    def provide(self, request: GradientScoreRequest) -> GradientScoreResult:
        assert self.state.target_active is False
        self.state.gradient_active = True
        self.state.events.append(f"gradient:{request.prompt}")
        return GradientScoreResult(
            score=0.1,
            target_tokens=2,
            n_tensors=1,
            provider=ProviderSummary(
                name="gradient_score",
                kind="gradients",
                requested=("anchor_loss_gradient",),
                materialized=("gradient_norm",),
                status="ok",
            ),
        )

    def close(self) -> None:
        self.state.gradient_active = False
        self.state.events.append("gradient:close")


def test_staged_policy_pair_releases_target_before_typed_intervention_provider() -> None:
    state = _State()
    backend = _TargetBackend(state)
    gradient = _GradientProvider(state)
    detector = GradSafeDetector(
        model_id="gradient-model",
        device="cpu",
        threshold=0.5,
    )
    samples = [
        Sample(
            sample_id=f"sample-{index}",
            behavior_id=f"behavior-{index}",
            is_benign=True,
            prompt=f"prompt-{index}",
        )
        for index in range(2)
    ]

    def release_target() -> None:
        assert state.gradient_active is False
        state.target_active = False
        state.events.append("target:release")

    result = run_policy_pair(
        samples=samples,
        reference=Component(name="reference", policy=PolicyChain()),
        intervention=Component(
            name="gradsafe",
            policy=detector.policy(),
            providers=(gradient,),
        ),
        cfg=Config(
            model=ModelConfig(backend="staged-test", model_id="target", max_new_tokens=8),
            naturalness=NaturalnessConfig(enabled=False),
        ),
        backend=backend,
        judge=DummyRefusalJudge(),
        release_target_before_intervention=release_target,
    )

    assert [scope for scope in result.target_scopes] == ["reference", "reference"]
    assert all(record.model.executed for record in result.reference_records)
    assert all(record.model.executed for record in result.intervention_records)
    gradient_uses = [
        use for use in result.request_uses if use.request_type.endswith(".GradientScoreRequest")
    ]
    assert [use.model_forwards for use in gradient_uses] == [1, 1]
    assert state.events == [
        "target:prompt-0",
        "target:prompt-1",
        "target:release",
        "gradient:prompt-0",
        "gradient:prompt-1",
        "gradient:close",
    ]


def test_policy_pair_closes_components_once_in_reverse_order() -> None:
    events: list[str] = []
    reference = Component(
        name="reference",
        policy=PolicyChain(),
        cleanup=lambda: events.append("reference:cleanup"),
    )
    intervention = Component(
        name="intervention",
        policy=PolicyChain(),
        cleanup=lambda: events.append("intervention:cleanup"),
    )

    run_policy_pair(
        samples=[],
        reference=reference,
        intervention=intervention,
        cfg=Config(),
        backend=_TargetBackend(_State()),
        judge=DummyRefusalJudge(),
    )

    assert events == ["intervention:cleanup", "reference:cleanup"]


def test_policy_pair_closes_components_after_policy_failure() -> None:
    events: list[str] = []

    class FailingPolicy:
        def apply(self, request, call_next, context):  # noqa: ANN001, ARG002
            raise RuntimeError("policy failed")

    sample = Sample(
        sample_id="sample",
        behavior_id="behavior",
        is_benign=True,
        prompt="prompt",
    )
    reference = Component(
        name="reference",
        policy=PolicyChain(),
        cleanup=lambda: events.append("reference:cleanup"),
    )
    intervention = Component(
        name="intervention",
        policy=FailingPolicy(),
        cleanup=lambda: events.append("intervention:cleanup"),
    )

    with pytest.raises(RuntimeError, match="policy failed"):
        run_policy_pair(
            samples=[sample],
            reference=reference,
            intervention=intervention,
            cfg=Config(),
            backend=_TargetBackend(_State()),
            judge=DummyRefusalJudge(),
        )

    assert events == ["intervention:cleanup", "reference:cleanup"]


def test_policy_pair_closes_shared_component_only_once() -> None:
    events: list[str] = []
    component = Component(
        name="shared",
        policy=PolicyChain(),
        cleanup=lambda: events.append("cleanup"),
    )

    run_policy_pair(
        samples=[],
        reference=component,
        intervention=component,
        cfg=Config(),
        backend=_TargetBackend(_State()),
        judge=DummyRefusalJudge(),
    )

    assert events == ["cleanup"]


def test_policy_pair_preserves_policy_error_when_component_cleanup_also_fails() -> None:
    class FailingPolicy:
        def apply(self, request, call_next, context):  # noqa: ANN001, ARG002
            raise ValueError("policy failed")

    def fail_cleanup() -> None:
        raise OSError("cleanup failed")

    sample = Sample(
        sample_id="sample",
        behavior_id="behavior",
        is_benign=True,
        prompt="prompt",
    )
    intervention = Component(
        name="intervention",
        policy=FailingPolicy(),
        cleanup=fail_cleanup,
    )

    with pytest.raises(ValueError, match="policy failed") as error:
        run_policy_pair(
            samples=[sample],
            reference=Component(name="reference", policy=PolicyChain()),
            intervention=intervention,
            cfg=Config(),
            backend=_TargetBackend(_State()),
            judge=DummyRefusalJudge(),
        )

    assert error.value.__turnkey_secondary_failures__[0][0] == "component cleanup"


def test_secondary_failure_attachment_supports_python_310_exceptions() -> None:
    class LegacyError(Exception):
        add_note = None

    primary = LegacyError("primary")
    secondary = OSError("secondary")

    _attach_secondary_failure(primary, label="cleanup", secondary=secondary)

    assert primary.__turnkey_secondary_failures__ == (("cleanup", secondary),)


def test_policy_pair_preserves_policy_error_when_provider_cleanup_also_fails() -> None:
    @dataclass(frozen=True)
    class ValueRequest(Request[int]):
        value: int

    class FailingCloseProvider:
        request_type = ValueRequest

        def provide(self, request: ValueRequest) -> int:
            return request.value

        def close(self) -> None:
            raise OSError("provider cleanup failed")

    class FailingPolicy:
        def apply(self, request, call_next, context):  # noqa: ANN001, ARG002
            context.get(ValueRequest(1))
            raise ValueError("policy failed")

    sample = Sample(
        sample_id="sample",
        behavior_id="behavior",
        is_benign=True,
        prompt="prompt",
    )

    with pytest.raises(ValueError, match="policy failed") as error:
        run_policy_pair(
            samples=[sample],
            reference=Component(name="reference", policy=PolicyChain()),
            intervention=Component(
                name="intervention",
                policy=FailingPolicy(),
                providers=(FailingCloseProvider(),),
            ),
            cfg=Config(),
            backend=_TargetBackend(_State()),
            judge=DummyRefusalJudge(),
        )

    assert error.value.__turnkey_secondary_failures__[0][0] == "method provider cleanup"
