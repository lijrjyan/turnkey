from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from turnkey.components.backends.base import LLMBackend
from turnkey.capabilities import BackendCapabilities
from turnkey.methods import MethodContext
from turnkey.runtime_providers import ProviderSummary, backend_capabilities_dict
from turnkey.runtime_providers.prompt_logprobs import PromptLogprobsRequest
from turnkey.schema import PrefixLogprobs, PromptLogprobs, Sample
from turnkey.pipeline_states import STATE_PREFIX_LOGPROBS, STATE_PROMPT_LOGPROBS


@dataclass(frozen=True)
class SignalRequest:
    """Target-model signals materialized directly by the benchmark backend."""

    prompt_logprobs: bool = False
    prefix_logprob_text: str | None = None

    @staticmethod
    def merge(*requests: "SignalRequest") -> "SignalRequest":
        prefix_texts = {
            req.prefix_logprob_text for req in requests if req.prefix_logprob_text is not None
        }
        if len(prefix_texts) > 1:
            raise ValueError(f"conflicting prefix_logprob_text requests: {sorted(prefix_texts)}")

        return SignalRequest(
            prompt_logprobs=any(req.prompt_logprobs for req in requests),
            prefix_logprob_text=next(iter(prefix_texts), None),
        )

    def requested_names(self) -> tuple[str, ...]:
        requested: list[str] = []
        if self.prompt_logprobs:
            requested.append(STATE_PROMPT_LOGPROBS)
        if self.prefix_logprob_text is not None:
            requested.append(STATE_PREFIX_LOGPROBS)
        return tuple(requested)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SignalBundle:
    request: SignalRequest
    backend_capabilities: BackendCapabilities
    prompt_logprobs: PromptLogprobs | None = None
    prefix_logprobs: PrefixLogprobs | None = None
    materialized: tuple[str, ...] = field(default_factory=tuple)
    providers: tuple[ProviderSummary, ...] = field(default_factory=tuple)

    def summary(self) -> dict[str, Any]:
        return {
            "requested": list(self.request.requested_names()),
            "materialized": list(self.materialized),
            "request": self.request.to_dict(),
            "providers": [provider.to_dict() for provider in self.providers],
        }


def signal_request_from_model_config(
    *, return_prompt_logprobs: bool, prefix_logprob_text: str | None
) -> SignalRequest:
    return SignalRequest(
        prompt_logprobs=bool(return_prompt_logprobs),
        prefix_logprob_text=prefix_logprob_text,
    )


def materialize_signals(
    *,
    sample: Sample,
    backend: LLMBackend,
    context: MethodContext,
    request: SignalRequest,
) -> SignalBundle:
    caps = backend.capabilities()
    _validate_supported(request=request, caps=caps)

    prompt_logprobs: PromptLogprobs | None = None
    prefix_logprobs: PrefixLogprobs | None = None
    materialized: list[str] = []

    if request.prompt_logprobs:
        prompt_logprobs = context.get(
            PromptLogprobsRequest(prompt=sample.prompt, images=sample.images)
        )
        materialized.append(STATE_PROMPT_LOGPROBS)

    if request.prefix_logprob_text is not None:
        prefix_logprobs = backend.get_prefix_logprobs(
            prompt=sample.prompt,
            prefix_text=request.prefix_logprob_text,
            images=sample.images,
        )
        materialized.append(STATE_PREFIX_LOGPROBS)

    provider_summary = ProviderSummary(
        name="model_signals",
        kind="model_signals",
        requested=request.requested_names(),
        materialized=tuple(materialized),
        capabilities=backend_capabilities_dict(caps),
        status="ok",
    )
    return SignalBundle(
        request=request,
        backend_capabilities=caps,
        prompt_logprobs=prompt_logprobs,
        prefix_logprobs=prefix_logprobs,
        materialized=tuple(materialized),
        providers=(provider_summary,),
    )


def _validate_supported(*, request: SignalRequest, caps: BackendCapabilities) -> None:
    if request.prompt_logprobs and not caps.prompt_logprobs:
        raise ValueError("signal request requires prompt_logprobs, but backend does not support it")
    if request.prefix_logprob_text is not None and not caps.prefix_logprobs:
        raise ValueError("signal request requires prefix_logprobs, but backend does not support it")
