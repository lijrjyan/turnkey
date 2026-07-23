from __future__ import annotations

from turnkey.config import JudgeConfig
from turnkey.components.judges.dummy import DummyRefusalJudge
from turnkey.components.judges.guardreasoner import GuardReasonerJudge
from turnkey.components.judges.llamaguard import LlamaGuardJudge
from turnkey.components.judges.qwen3guard import Qwen3GuardJudge
from turnkey.components.judges.smoothllm import SmoothLLMDetector
from turnkey.components.judges.strongreject import StrongRejectJudge
from turnkey.components.judges.wildguard import WildGuardJudge
from turnkey.schema import JudgeOutput, Sample
from turnkey.registry import JUDGES, register_judge


class Judge:
    def judge(self, *, sample: Sample, model_text: str) -> JudgeOutput:
        raise NotImplementedError


def load_judge(cfg: JudgeConfig) -> Judge:
    return JUDGES.get(cfg.name)(cfg)


@register_judge("dummy_refusal")
def _build_dummy_refusal(cfg: JudgeConfig) -> Judge:
    return DummyRefusalJudge(**cfg.params)


@register_judge("qwen3guard")
def _build_qwen3guard(cfg: JudgeConfig) -> Judge:
    return Qwen3GuardJudge(**cfg.params)


@register_judge("wildguard")
def _build_wildguard(cfg: JudgeConfig) -> Judge:
    return WildGuardJudge(**cfg.params)


@register_judge("strongreject")
def _build_strongreject(cfg: JudgeConfig) -> Judge:
    return StrongRejectJudge(**cfg.params)


@register_judge("smoothllm")
def _build_smoothllm(cfg: JudgeConfig) -> Judge:
    return SmoothLLMDetector(**cfg.params)


@register_judge("guardreasoner")
def _build_guardreasoner(cfg: JudgeConfig) -> Judge:
    return GuardReasonerJudge(**cfg.params)


@register_judge("llamaguard")
def _build_llamaguard(cfg: JudgeConfig) -> Judge:
    return LlamaGuardJudge(**cfg.params)


def available_judges() -> list[str]:
    return JUDGES.list()
