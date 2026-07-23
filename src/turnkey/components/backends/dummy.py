from __future__ import annotations

import re
import time
from dataclasses import dataclass

from turnkey.components.backends.base import LLMBackend
from turnkey.capabilities import BackendCapabilities
from turnkey.schema import ImageInput, ModelOutput, PrefixLogprobs, PromptLogprobs, TokenLogprob


def _dummy_tokens(text: str) -> list[str]:
    parts = re.findall(r"\s+|\S+", text)
    return parts or [text]


def _build_dummy_logprobs(tokens: list[str]) -> PromptLogprobs:
    values: list[TokenLogprob] = []
    scored: list[float] = []
    for idx, token in enumerate(tokens):
        logprob = None if idx == 0 else -0.1 * idx
        if logprob is not None:
            scored.append(logprob)
        values.append(TokenLogprob(token=token, token_id=idx, logprob=logprob))
    avg = sum(scored) / len(scored) if scored else None
    return PromptLogprobs(tokens=tuple(values), logprob_sum=sum(scored) if scored else None, logprob_avg=avg)


@dataclass
class DummyBackend(LLMBackend):
    model_id: str = "dummy"

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(prompt_logprobs=True, prefix_logprobs=True, token_logprobs=True)

    def get_prompt_logprobs(
        self,
        *,
        prompt: str,
        images: tuple[ImageInput, ...] | None = None,
    ) -> PromptLogprobs:  # noqa: ARG002
        return _build_dummy_logprobs(_dummy_tokens(prompt))

    def get_prefix_logprobs(
        self,
        *,
        prompt: str,
        prefix_text: str,
        images: tuple[ImageInput, ...] | None = None,
    ) -> PrefixLogprobs:  # noqa: ARG002
        full_tokens = _dummy_tokens(prompt + prefix_text)
        prompt_len = len(_dummy_tokens(prompt))
        prompt_logprobs = _build_dummy_logprobs(full_tokens)
        prefix_tokens = prompt_logprobs.tokens[prompt_len:]
        scored = [token.logprob for token in prefix_tokens if token.logprob is not None]
        return PrefixLogprobs(
            prefix_text=prefix_text,
            tokens=tuple(prefix_tokens),
            logprob_sum=sum(scored) if scored else None,
            logprob_avg=(sum(scored) / len(scored)) if scored else None,
        )

    def generate(
        self,
        *,
        prompt: str,
        images: tuple[ImageInput, ...] | None = None,
        max_new_tokens: int,
        temperature: float,
    ) -> ModelOutput:  # noqa: ARG002
        t0 = time.time()
        # Deterministic, CI-safe placeholder output.
        text = "OK"
        if images:
            text = "OK_MM"
        if "UNSAFE_PLACEHOLDER" in prompt:
            # Not a refusal (so strict success can be non-zero in plumbing tests),
            # but also contains no harmful content.
            text = "ACK"
        latency_s = time.time() - t0
        prompt_tokens = max(1, len(prompt.split()))
        completion_tokens = max(1, len(text.split()))
        total_tokens = prompt_tokens + completion_tokens
        return ModelOutput(
            executed=True,
            backend="dummy",
            model_id=self.model_id,
            response_text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_s=latency_s,
        )
