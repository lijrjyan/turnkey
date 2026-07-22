from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, replace
from typing import Any

from turnkey._internal.smoothllm import PERTURBATIONS, is_jailbroken
from turnkey.components.detectors.base import Detector, DetectorManifest
from turnkey.methods import MethodContext
from turnkey.policy import NextPolicy, Outcome, Policy, PolicyRequest
from turnkey.schema import DetectorDecision, Sample


@dataclass
class SmoothLLMDetector(Detector):
    pert_type: str = "RandomSwapPerturbation"
    pert_pct: int = 10
    num_copies: int = 10
    vote_threshold: float = 0.5
    seed: int | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.pert_pct <= 100:
            raise ValueError(f"pert_pct must be 0-100, got {self.pert_pct}")
        if self.num_copies < 1:
            raise ValueError(f"num_copies must be >= 1, got {self.num_copies}")
        if not 0 <= self.vote_threshold <= 1:
            raise ValueError(f"vote_threshold must be 0-1, got {self.vote_threshold}")
        if self.pert_type not in PERTURBATIONS:
            raise ValueError(
                f"Unknown perturbation type: {self.pert_type}. "
                f"Available: {list(PERTURBATIONS)}"
            )

    def _rng(self, sample: Sample, *, purpose: str) -> random.Random:
        if self.seed is None:
            return random.Random()
        identity = f"{self.seed}\n{purpose}\n{sample.sample_id}\n{sample.prompt}"
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        return random.Random(int(digest[:16], 16))

    def _perturbed_prompts(self, sample: Sample) -> tuple[str, ...]:
        rng = self._rng(sample, purpose="perturb")
        perturb = PERTURBATIONS[self.pert_type](self.pert_pct, rng=rng)
        return tuple(perturb(sample.prompt) for _ in range(self.num_copies))

    def policy(self, *, calibration_artifact: Any | None = None) -> Policy:  # noqa: ARG002
        return SmoothLLMPolicy(self)

    def decide(self, sample: Sample) -> DetectorDecision:  # noqa: ARG002
        raise RuntimeError("smoothllm requires Policy target generation")

    def manifest(self, *, name: str | None = None) -> DetectorManifest:
        return DetectorManifest(
            name=name or "smoothllm",
            required_inputs=("sample", "prompt"),
            reproducibility={"seed": self.seed},
        )


@dataclass(frozen=True)
class SmoothLLMPolicy:
    detector: SmoothLLMDetector

    def apply(
        self,
        request: PolicyRequest,
        call_next: NextPolicy,
        context: MethodContext,  # noqa: ARG002
    ) -> Outcome:
        outcomes = [
            call_next(replace(request, target=replace(request.target, prompt=prompt)))
            for prompt in self.detector._perturbed_prompts(request.sample)
        ]
        if any(outcome.model is None for outcome in outcomes):
            raise RuntimeError("SmoothLLM requires a generated response for every perturbed prompt")

        models = []
        for outcome in outcomes:
            assert outcome.model is not None
            models.append(outcome.model)
        labels = [is_jailbroken(model.response_text or "") for model in models]
        jailbreak_votes = sum(labels)
        jailbreak_fraction = jailbreak_votes / len(labels)
        majority_jailbroken = jailbreak_fraction > self.detector.vote_threshold
        candidates = [index for index, label in enumerate(labels) if label == majority_jailbroken]
        selected_index = self.detector._rng(request.sample, purpose="select").choice(candidates)
        selected = outcomes[selected_index]
        assert selected.model is not None
        return Outcome.generated(
            request.target,
            selected.model,
            score=jailbreak_fraction,
            reason=(
                "smoothllm majority vote: jailbroken"
                if majority_jailbroken
                else "smoothllm majority vote: refused"
            ),
            diagnostics={
                "perturbation_type": self.detector.pert_type,
                "perturbation_pct": self.detector.pert_pct,
                "num_copies": self.detector.num_copies,
                "vote_threshold": self.detector.vote_threshold,
                "jailbroken_votes": jailbreak_votes,
                "jailbroken_fraction": jailbreak_fraction,
                "majority_jailbroken": majority_jailbroken,
                "selected_copy": selected_index,
            },
        )
