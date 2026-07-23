from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ImageInput:
    """
    Minimal multimodal input reference.

    For now we keep this as a lightweight file reference so CI can run without heavy
    image-processing deps; backends may choose how/when to load bytes.
    """

    path: str
    mime_type: str | None = None


@dataclass(frozen=True)
class Sample:
    sample_id: str
    behavior_id: str
    is_benign: bool
    prompt: str
    images: tuple[ImageInput, ...] = field(default_factory=tuple)
    attack_family: str = "T0"
    attack_method: str = "none"
    attack_params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ThreatContext:
    tier: str
    attack_family: str
    attack_method: str


@dataclass(frozen=True)
class AttackBudget:
    queries: int | None = None
    steps: int | None = None
    turns: int | None = None
    shots: int | None = None
    suffix_len: int | None = None
    extra_forwards: int | None = None
    bucket: str = "none"


@dataclass(frozen=True)
class DetectorDecision:
    block: bool
    score: float | None = None
    reason: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TokenLogprob:
    token: str
    token_id: int | None
    logprob: float | None


@dataclass(frozen=True)
class PromptLogprobs:
    tokens: tuple[TokenLogprob, ...] = field(default_factory=tuple)
    logprob_sum: float | None = None
    logprob_avg: float | None = None


@dataclass(frozen=True)
class PrefixLogprobs:
    prefix_text: str
    tokens: tuple[TokenLogprob, ...] = field(default_factory=tuple)
    logprob_sum: float | None = None
    logprob_avg: float | None = None


@dataclass(frozen=True)
class ModelOutput:
    executed: bool
    backend: str
    model_id: str
    response_text: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    latency_s: float | None = None
    prompt_logprobs: PromptLogprobs | None = None
    prefix_logprobs: PrefixLogprobs | None = None


@dataclass(frozen=True)
class JudgeOutput:
    is_refusal: bool | None
    is_harmful_effective: bool | None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Record:
    sample_id: str
    behavior_id: str
    is_benign: bool
    attack_family: str
    attack_method: str
    attack_params: dict[str, Any]
    threat: ThreatContext
    budget: AttackBudget

    prompt_hash: str
    prompt_chars: int

    detector: DetectorDecision
    model: ModelOutput
    judge: JudgeOutput

    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
