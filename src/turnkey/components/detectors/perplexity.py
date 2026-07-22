from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

from turnkey.components.detectors.base import Detector, DetectorManifest
from turnkey.methods import MethodContext
from turnkey.policy import NextPolicy, Outcome, Policy, PolicyRequest
from turnkey.runtime_providers import PromptLogprobsRequest
from turnkey.schema import DetectorDecision, Sample


@dataclass(frozen=True)
class PerplexityDetector(Detector):
    """Block prompts whose target-model perplexity exceeds a fixed threshold."""

    threshold: float = 100.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.threshold) or self.threshold <= 0.0:
            raise ValueError("perplexity threshold must be finite and greater than zero")

    def policy(self, *, calibration_artifact: Any | None = None) -> Policy:  # noqa: ARG002
        return PerplexityPolicy(threshold=self.threshold)

    def decide(self, sample: Sample) -> DetectorDecision:  # noqa: ARG002
        raise RuntimeError("perplexity detector requires typed prompt logprobs")

    def manifest(self, *, name: str | None = None) -> DetectorManifest:
        return DetectorManifest(
            name=name or "perplexity",
            version="baseline",
            required_inputs=("sample", "prompt", "prompt_logprobs"),
            reproducibility={
                "method": "plain_prompt_perplexity",
                "paper": "https://arxiv.org/abs/2308.14132",
            },
        )


@dataclass(frozen=True)
class PerplexityPolicy:
    threshold: float

    def apply(
        self,
        request: PolicyRequest,
        call_next: NextPolicy,
        context: MethodContext,
    ) -> Outcome:
        prompt_logprobs = context.get(
            PromptLogprobsRequest(
                prompt=request.sample.prompt,
                images=request.sample.images,
            )
        )
        mean_logprob = prompt_logprobs.logprob_avg
        if mean_logprob is None or not math.isfinite(mean_logprob):
            raise ValueError("perplexity detector requires a finite mean logprob")
        try:
            score = math.exp(-mean_logprob)
        except OverflowError as exc:
            raise ValueError("perplexity detector requires finite perplexity") from exc
        if not math.isfinite(score):
            raise ValueError("perplexity detector requires finite perplexity")

        diagnostics = {
            "threshold": self.threshold,
            "mean_logprob": mean_logprob,
            "scored_tokens": sum(token.logprob is not None for token in prompt_logprobs.tokens),
        }
        if score > self.threshold:
            return Outcome.blocked(
                request.target,
                score=score,
                reason="prompt perplexity above threshold",
                diagnostics=diagnostics,
            )

        outcome = call_next(request)
        return replace(
            outcome,
            score=score,
            reason="prompt perplexity at or below threshold",
            diagnostics={**diagnostics, **outcome.diagnostics},
        )
