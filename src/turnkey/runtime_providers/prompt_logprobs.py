from __future__ import annotations

from dataclasses import dataclass

from turnkey.components.backends.base import LLMBackend
from turnkey.methods import Request
from turnkey.schema import ImageInput, PromptLogprobs


@dataclass(frozen=True)
class PromptLogprobsRequest(Request[PromptLogprobs]):
    """Request target-model log-probabilities for every token in a prompt."""

    prompt: str
    images: tuple[ImageInput, ...] = ()


class PromptLogprobsProvider:
    """Borrow the runner-owned target backend for an auditable prompt forward."""

    request_type = PromptLogprobsRequest
    model_forwards_per_call = 1

    def __init__(self, backend: LLMBackend):
        self._backend: LLMBackend | None = backend

    def provide(self, request: PromptLogprobsRequest) -> PromptLogprobs:
        backend = self._backend
        if backend is None:
            raise RuntimeError(
                "target backend was released before an uncached prompt-logprob request"
            )
        if not backend.capabilities().prompt_logprobs:
            raise ValueError(
                f"target backend {type(backend).__name__} does not support prompt logprobs"
            )
        return backend.get_prompt_logprobs(
            prompt=request.prompt,
            images=request.images,
        )

    def release(self) -> None:
        """Drop the borrowed backend reference without closing its owner."""

        self._backend = None
