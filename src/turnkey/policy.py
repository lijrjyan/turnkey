from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
import json
from typing import Any, Literal, Protocol

from turnkey.methods import MethodContext, Provider, Request
from turnkey.schema import DetectorDecision, ImageInput, ModelOutput, Sample


Action = Literal["allow", "block"]


@dataclass(frozen=True)
class Generate(Request[ModelOutput]):
    prompt: str
    images: tuple[ImageInput, ...] = ()
    max_new_tokens: int = 96
    temperature: float = 0.0


@dataclass(frozen=True)
class PolicyRequest:
    sample: Sample
    target: Generate


@dataclass(frozen=True)
class Outcome:
    action: Action
    target: Generate
    model: ModelOutput | None = None
    score: float | None = None
    reason: str | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.action == "allow" and self.model is None:
            raise ValueError("allow outcome requires a model output")
        if self.action == "block" and self.model is not None:
            raise ValueError("block outcome cannot contain a model output")

    @classmethod
    def blocked(
        cls,
        target: Generate,
        *,
        score: float | None = None,
        reason: str | None = None,
        diagnostics: Mapping[str, Any] | None = None,
    ) -> "Outcome":
        return cls(
            action="block",
            target=target,
            score=score,
            reason=reason,
            diagnostics=dict(diagnostics or {}),
        )

    @classmethod
    def generated(
        cls,
        target: Generate,
        model: ModelOutput,
        *,
        score: float | None = None,
        reason: str | None = None,
        diagnostics: Mapping[str, Any] | None = None,
    ) -> "Outcome":
        return cls(
            action="allow",
            target=target,
            model=model,
            score=score,
            reason=reason,
            diagnostics=dict(diagnostics or {}),
        )


NextPolicy = Callable[[PolicyRequest], Outcome]


class Policy(Protocol):
    def apply(
        self,
        request: PolicyRequest,
        call_next: NextPolicy,
        context: MethodContext,
    ) -> Outcome:
        ...


@dataclass(frozen=True)
class DetectorPolicy:
    decide: Callable[[Sample], DetectorDecision]

    def apply(
        self,
        request: PolicyRequest,
        call_next: NextPolicy,
        context: MethodContext,  # noqa: ARG002
    ) -> Outcome:
        decision = self.decide(request.sample)
        if decision.block:
            return Outcome.blocked(
                request.target,
                score=decision.score,
                reason=decision.reason,
                diagnostics=decision.diagnostics,
            )

        outcome = call_next(request)
        return replace(
            outcome,
            score=outcome.score if outcome.score is not None else decision.score,
            reason=outcome.reason if outcome.reason is not None else decision.reason,
            diagnostics={**decision.diagnostics, **outcome.diagnostics},
        )


@dataclass(frozen=True)
class TargetSession:
    context: MethodContext

    def run(self, request: PolicyRequest) -> Outcome:
        model = self.context.get(request.target)
        return Outcome.generated(request.target, model)


@dataclass(frozen=True)
class PolicyChain:
    policies: tuple[Policy, ...] = ()

    def apply(
        self,
        request: PolicyRequest,
        call_next: NextPolicy,
        context: MethodContext,
    ) -> Outcome:
        def dispatch(index: int, current: PolicyRequest) -> Outcome:
            if index == len(self.policies):
                return call_next(current)
            policy = self.policies[index]
            return policy.apply(
                current,
                lambda next_request: dispatch(index + 1, next_request),
                context,
            )

        return dispatch(0, request)

    def run(self, request: PolicyRequest, target: TargetSession) -> Outcome:
        return self.apply(request, target.run, target.context)


@dataclass(frozen=True)
class Component:
    name: str
    policy: Policy
    providers: tuple[Provider[Any], ...] = ()
    parameters: Mapping[str, Any] = field(default_factory=dict)
    cleanup: Callable[[], None] | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("component name must be non-empty")
        if self.cleanup is not None and not callable(self.cleanup):
            raise TypeError("component cleanup must be callable")
        parameters = dict(self.parameters)
        _require_string_mapping_keys(parameters, loc=f"Component[{self.name!r}].parameters")
        try:
            canonical = json.loads(
                json.dumps(parameters, ensure_ascii=False, allow_nan=False)
            )
        except (TypeError, ValueError) as exc:
            raise TypeError("component parameters must be JSON-serializable") from exc
        object.__setattr__(self, "providers", tuple(self.providers))
        object.__setattr__(self, "parameters", canonical)


def _require_string_mapping_keys(value: Any, *, loc: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{loc} must use string keys in nested mappings")
            _require_string_mapping_keys(item, loc=f"{loc}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _require_string_mapping_keys(item, loc=f"{loc}[{index}]")
