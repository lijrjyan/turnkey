from __future__ import annotations

from dataclasses import dataclass

import pytest

from turnkey.capabilities import BackendCapabilities
from turnkey.components.backends.base import LLMBackend
from turnkey.components.judges.dummy import DummyRefusalJudge
from turnkey.config import Config, ModelConfig, NaturalnessConfig
from turnkey.methods import MethodContext
from turnkey.policy import Component, PolicyChain
from turnkey.runner.policy_executor import run_policy_pair
from turnkey.schema import ModelOutput, PromptLogprobs, Sample, TokenLogprob


@dataclass
class _LogprobBackend(LLMBackend):
    supports_logprobs: bool = True

    def __post_init__(self) -> None:
        self.logprob_prompts: list[str] = []

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(prompt_logprobs=self.supports_logprobs)

    def get_prompt_logprobs(self, *, prompt: str, images=None) -> PromptLogprobs:  # noqa: ARG002
        self.logprob_prompts.append(prompt)
        return PromptLogprobs(
            tokens=(TokenLogprob(token=prompt, token_id=1, logprob=-0.25),),
            logprob_sum=-0.25,
            logprob_avg=-0.25,
        )

    def generate(
        self,
        *,
        prompt: str,
        images=None,
        max_new_tokens: int,
        temperature: float,
    ) -> ModelOutput:  # noqa: ARG002
        return ModelOutput(
            executed=True,
            backend="logprob-test",
            model_id="target",
            response_text="OK",
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
            latency_s=0.0,
        )


def _provider_api():
    from turnkey.runtime_providers import PromptLogprobsProvider, PromptLogprobsRequest

    return PromptLogprobsProvider, PromptLogprobsRequest


def test_prompt_logprob_request_cache_counts_only_the_miss_as_a_forward() -> None:
    PromptLogprobsProvider, PromptLogprobsRequest = _provider_api()
    backend = _LogprobBackend()
    request = PromptLogprobsRequest(prompt="hello")

    with MethodContext((PromptLogprobsProvider(backend),)) as context:
        assert context.get(request).logprob_avg == pytest.approx(-0.25)
        assert context.get(request).logprob_avg == pytest.approx(-0.25)
        uses = context.uses

    assert backend.logprob_prompts == ["hello"]
    assert [use.cache_hit for use in uses] == [False, True]
    assert [use.model_forwards for use in uses] == [1, 0]


def test_prompt_logprob_provider_rejects_missing_backend_capability() -> None:
    PromptLogprobsProvider, PromptLogprobsRequest = _provider_api()
    provider = PromptLogprobsProvider(_LogprobBackend(supports_logprobs=False))

    with pytest.raises(ValueError, match="does not support prompt logprobs"):
        provider.provide(PromptLogprobsRequest(prompt="hello"))


def test_prompt_logprob_provider_rejects_uncached_request_after_release() -> None:
    PromptLogprobsProvider, PromptLogprobsRequest = _provider_api()
    provider = PromptLogprobsProvider(_LogprobBackend())
    provider.release()

    with pytest.raises(RuntimeError, match="released"):
        provider.provide(PromptLogprobsRequest(prompt="new prompt"))


def test_runner_reuses_artifact_prompt_logprobs_for_detector_request() -> None:
    _, PromptLogprobsRequest = _provider_api()
    backend = _LogprobBackend()

    class PromptLogprobPolicy:
        def apply(self, request, call_next, context):  # noqa: ANN001
            context.get(
                PromptLogprobsRequest(
                    prompt=request.sample.prompt,
                    images=request.sample.images,
                )
            )
            return call_next(request)

    result = run_policy_pair(
        samples=[
            Sample(
                sample_id="sample-1",
                behavior_id="behavior-1",
                is_benign=True,
                prompt="hello",
            )
        ],
        reference=Component(name="reference", policy=PolicyChain()),
        intervention=Component(name="prompt-logprob", policy=PromptLogprobPolicy()),
        cfg=Config(
            model=ModelConfig(
                backend="logprob-test",
                model_id="target",
                return_prompt_logprobs=True,
            ),
            naturalness=NaturalnessConfig(enabled=False),
        ),
        backend=backend,
        judge=DummyRefusalJudge(),
    )

    prompt_uses = [
        use for use in result.request_uses if use.request_type.endswith(".PromptLogprobsRequest")
    ]
    assert backend.logprob_prompts == ["hello"]
    assert [use.scope for use in prompt_uses] == ["reference", "intervention"]
    assert [use.cache_hit for use in prompt_uses] == [False, True]
    assert [use.model_forwards for use in prompt_uses] == [1, 0]
