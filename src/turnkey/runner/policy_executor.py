from dataclasses import dataclass, replace
from typing import Any, Callable, Literal

from turnkey._internal.redact import sha256_hex
from turnkey.components.backends.base import LLMBackend
from turnkey.config import Config
from turnkey.methods import MethodContext, RequestUse, _attach_secondary_failure
from turnkey.policy import Component, Generate, Outcome, PolicyChain, PolicyRequest, TargetSession
from turnkey.runtime_events import EventRecorder, RuntimeEvent
from turnkey.schema import DetectorDecision, JudgeOutput, ModelOutput, Record, Sample
from turnkey.signals import SignalBundle, materialize_signals, signal_request_from_model_config

from .artifacts import (
    model_output_signals,
    record_for_sample,
    signal_summary_with_providers,
)


TargetScope = Literal["reference", "intervention"]


@dataclass
class BackendGenerateProvider:
    backend: LLMBackend | None
    request_type = Generate
    model_forwards_per_call = 1

    def __post_init__(self) -> None:
        self.calls: list[Generate] = []
        self.outputs: list[ModelOutput] = []
        self.scopes: list[TargetScope] = []
        self.scope: TargetScope | None = None
        self.released = False

    def provide(self, request: Generate) -> ModelOutput:
        if self.scope is None:
            raise RuntimeError("target generation scope is not set")
        if self.backend is None:
            raise RuntimeError("target backend was released before an uncached request")
        output = self.backend.generate(
            prompt=request.prompt,
            images=request.images,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
        )
        self.calls.append(request)
        self.outputs.append(output)
        self.scopes.append(self.scope)
        return output

    def release(self, callback: Callable[[], None]) -> None:
        if self.released:
            return
        try:
            callback()
        finally:
            self.backend = None
            self.released = True


@dataclass(frozen=True)
class _PreparedCase:
    sample: Sample
    request: PolicyRequest
    signals: SignalBundle
    signal_summary: dict[str, Any]
    reference_outcome: Outcome


@dataclass(frozen=True)
class PolicyPairResult:
    reference_records: tuple[Record, ...]
    intervention_records: tuple[Record, ...]
    target_calls: tuple[Generate, ...]
    target_outputs: tuple[ModelOutput, ...]
    target_scopes: tuple[TargetScope, ...]
    request_uses: tuple[RequestUse, ...]
    events: tuple[RuntimeEvent, ...]


def run_policy_pair(
    *,
    samples: list[Sample],
    reference: Component,
    intervention: Component,
    cfg: Config,
    backend: LLMBackend,
    judge: Any,
    release_target_before_intervention: Callable[[], None] | None = None,
) -> PolicyPairResult:
    target_provider = BackendGenerateProvider(backend)
    providers = (target_provider, *reference.providers, *intervention.providers)
    reference_chain = PolicyChain((reference.policy,))
    intervention_chain = PolicyChain((intervention.policy,))
    reference_records: list[Record] = []
    intervention_records: list[Record] = []
    prepared_cases: list[_PreparedCase] = []
    event_recorder = EventRecorder()

    primary_error: BaseException | None = None
    primary_traceback = None
    try:
        with MethodContext(providers, event_recorder=event_recorder) as context:
            target = TargetSession(context)
            for sample in samples:
                signals = materialize_signals(
                    sample=sample,
                    backend=backend,
                    request=signal_request_from_model_config(
                        return_prompt_logprobs=cfg.model.return_prompt_logprobs,
                        prefix_logprob_text=cfg.model.prefix_logprob_text,
                    ),
                )
                request = PolicyRequest(
                    sample=sample,
                    target=Generate(
                        prompt=sample.prompt,
                        images=sample.images,
                        max_new_tokens=cfg.model.max_new_tokens,
                        temperature=cfg.model.temperature,
                    ),
                )
                signal_summary = signal_summary_with_providers(signals, list(signals.providers))
                target_provider.scope = "reference"
                with event_recorder.span(
                    case_id=sample.sample_id,
                    pass_name="reference",
                    kind="policy",
                    name=reference.name,
                ) as policy_event:
                    with context.scope(
                        case_id=sample.sample_id,
                        pass_name="reference",
                        parent_event_id=policy_event.event_id,
                    ):
                        reference_outcome = reference_chain.run(request, target)
                    policy_event.result = _policy_result_summary(reference_outcome)
                    reference_record = _record_outcome(
                        sample=sample,
                        outcome=reference_outcome,
                        cfg=cfg,
                        judge=judge,
                        signals=signals,
                        signal_summary=signal_summary,
                        pass_name="reference",
                        parent_event_id=policy_event.event_id,
                        event_recorder=event_recorder,
                    )
                reference_records.append(reference_record)
                prepared_cases.append(
                    _PreparedCase(
                        sample=sample,
                        request=request,
                        signals=signals,
                        signal_summary=signal_summary,
                        reference_outcome=reference_outcome,
                    )
                )

            if release_target_before_intervention is not None:
                _validate_staged_reference(prepared_cases)
                target_provider.release(release_target_before_intervention)
                backend = None  # release the function-frame reference before provider materialization

            for prepared in prepared_cases:
                target_provider.scope = "intervention"
                with event_recorder.span(
                    case_id=prepared.sample.sample_id,
                    pass_name="intervention",
                    kind="policy",
                    name=intervention.name,
                ) as policy_event:
                    with context.scope(
                        case_id=prepared.sample.sample_id,
                        pass_name="intervention",
                        parent_event_id=policy_event.event_id,
                    ):
                        intervention_outcome = intervention_chain.run(prepared.request, target)
                    policy_event.result = _policy_result_summary(intervention_outcome)
                    intervention_record = _record_outcome(
                        sample=prepared.sample,
                        outcome=intervention_outcome,
                        cfg=cfg,
                        judge=judge,
                        signals=prepared.signals,
                        signal_summary=prepared.signal_summary,
                        pass_name="intervention",
                        parent_event_id=policy_event.event_id,
                        event_recorder=event_recorder,
                    )
                intervention_records.append(intervention_record)
            request_uses = context.uses
    except BaseException as exc:
        primary_error = exc
        primary_traceback = exc.__traceback__

    if release_target_before_intervention is not None and not target_provider.released:
        try:
            target_provider.release(release_target_before_intervention)
            backend = None
        except BaseException as exc:
            if primary_error is None:
                primary_error = exc
                primary_traceback = exc.__traceback__
            else:
                _attach_secondary_failure(
                    primary_error,
                    label="target release",
                    secondary=exc,
                )
    try:
        cleanup_components(reference, intervention)
    except BaseException as exc:
        if primary_error is None:
            primary_error = exc
            primary_traceback = exc.__traceback__
        else:
            _attach_secondary_failure(
                primary_error,
                label="component cleanup",
                secondary=exc,
            )

    if primary_error is not None:
        raise primary_error.with_traceback(primary_traceback)

    return PolicyPairResult(
        reference_records=tuple(reference_records),
        intervention_records=tuple(intervention_records),
        target_calls=tuple(target_provider.calls),
        target_outputs=tuple(target_provider.outputs),
        target_scopes=tuple(target_provider.scopes),
        request_uses=request_uses,
        events=event_recorder.events,
    )


def cleanup_components(*components: Component) -> None:
    errors: list[Exception] = []
    seen: set[int] = set()
    for component in reversed(components):
        identity = id(component)
        if identity in seen:
            continue
        seen.add(identity)
        if component.cleanup is None:
            continue
        try:
            component.cleanup()
        except Exception as exc:  # cleanup must continue for remaining components
            errors.append(exc)
    if errors:
        noun = "component" if len(errors) == 1 else "components"
        raise RuntimeError(f"failed to clean up {len(errors)} runtime {noun}") from errors[0]


def cleanup_components_after_error(
    primary: BaseException,
    *components: Component,
) -> None:
    try:
        cleanup_components(*components)
    except BaseException as cleanup_error:
        _attach_secondary_failure(
            primary,
            label="component cleanup",
            secondary=cleanup_error,
        )


def _validate_staged_reference(prepared_cases: list[_PreparedCase]) -> None:
    for prepared in prepared_cases:
        outcome = prepared.reference_outcome
        if outcome.model is None or outcome.target != prepared.request.target:
            raise RuntimeError(
                "staged intervention requires reference to generate each original target request"
            )


def _record_outcome(
    *,
    sample: Sample,
    outcome: Outcome,
    cfg: Config,
    judge: Any,
    signals: SignalBundle,
    signal_summary: dict[str, Any],
    pass_name: TargetScope,
    parent_event_id: str,
    event_recorder: EventRecorder,
) -> Record:
    prompt_logprobs, prefix_logprobs = model_output_signals(cfg=cfg, signals=signals)
    decision = DetectorDecision(
        block=outcome.action == "block",
        score=outcome.score,
        reason=outcome.reason,
        diagnostics=dict(outcome.diagnostics),
    )
    with event_recorder.span(
        case_id=sample.sample_id,
        pass_name=pass_name,
        kind="judge",
        name=f"{type(judge).__module__}.{type(judge).__qualname__}",
        parent_event_id=parent_event_id,
    ) as judge_event:
        if outcome.model is None:
            model = ModelOutput(
                executed=False,
                backend=cfg.model.backend,
                model_id=cfg.model.model_id,
                response_text=None,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                latency_s=0.0,
                prompt_logprobs=prompt_logprobs,
                prefix_logprobs=prefix_logprobs,
            )
            judge_output = JudgeOutput(
                is_refusal=None,
                is_harmful_effective=None,
                details={"blocked": True},
            )
            judge_event.result = {"skipped": True, "reason": "blocked"}
        else:
            model = replace(
                outcome.model,
                prompt_logprobs=prompt_logprobs,
                prefix_logprobs=prefix_logprobs,
            )
            judge_output = judge.judge(sample=sample, model_text=model.response_text or "")
            judge_event.result = {
                "is_refusal": judge_output.is_refusal,
                "is_harmful_effective": judge_output.is_harmful_effective,
            }

    return record_for_sample(
        sample=sample,
        detector_decision=decision,
        model_out=model,
        judge_out=judge_output,
        signal_summary=signal_summary,
    )


def _policy_result_summary(outcome: Outcome) -> dict[str, Any]:
    return {
        "action": outcome.action,
        "target_prompt_sha256": sha256_hex(outcome.target.prompt),
    }
