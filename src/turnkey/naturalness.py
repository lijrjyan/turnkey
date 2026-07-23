from __future__ import annotations

from collections import Counter
import json
import math
import re
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Protocol

from turnkey.schema import Sample


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]")
_ORDER = 4
_START = "<s>"
_END = "</s>"


class PromptPerplexityScorer(Protocol):
    order: int
    backend: str

    def perplexity(self, prompt: str) -> float: ...


def annotate_samples_with_ngram_ppl(
    samples: Iterable[Sample],
    *,
    reference_prompts: Iterable[str] | None = None,
    bucket_reference_prompts: Iterable[str] | None = None,
    reference_id: str | None = None,
    scorer: PromptPerplexityScorer | None = None,
) -> list[Sample]:
    """Annotate samples with prompt perplexity metadata."""

    materialized = list(samples)
    if not materialized:
        return []

    if scorer is None:
        lm_prompts = list(reference_prompts) if reference_prompts is not None else []
        if not lm_prompts:
            lm_prompts = [sample.prompt for sample in materialized if sample.is_benign]
        if not lm_prompts:
            lm_prompts = [sample.prompt for sample in materialized]
        model = NgramLanguageModel.fit(lm_prompts)
        scores = [model.perplexity(sample.prompt) for sample in materialized]
        order = _ORDER
        backend = "turnkey_local_4gram"
    else:
        scores = [scorer.perplexity(sample.prompt) for sample in materialized]
        order = int(scorer.order)
        backend = scorer.backend

    bucket_prompts = list(bucket_reference_prompts) if bucket_reference_prompts is not None else []
    if bucket_prompts:
        if scorer is None:
            bucket_scores = [model.perplexity(prompt) for prompt in bucket_prompts]
        else:
            bucket_scores = [scorer.perplexity(prompt) for prompt in bucket_prompts]
    else:
        bucket_scores = [score for score, sample in zip(scores, materialized, strict=True) if sample.is_benign]
    thresholds = _bucket_thresholds(bucket_scores or scores)

    effective_reference_id = reference_id or "run_benign"
    out = []
    for sample, score in zip(materialized, scores, strict=True):
        params = dict(sample.attack_params)
        params["ngram_ppl"] = float(score)
        params["ngram_ppl_bucket"] = _bucket(score, thresholds=thresholds)
        params["ngram_ppl_order"] = order
        params["ngram_ppl_backend"] = backend
        params["ngram_ppl_reference"] = effective_reference_id
        out.append(replace(sample, attack_params=params))
    return out


def load_prompt_corpus(path: str | Path) -> list[str]:
    path = Path(path)
    prompts: list[str] = []
    if path.suffix.lower() == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        prompts = _prompts_from_json(raw)
    else:
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                prompt = _prompt_from_line(line)
                if prompt is None:
                    raise ValueError(f"{path}:{line_no}: missing prompt/text field")
                prompts.append(prompt)
    if not prompts:
        raise ValueError(f"{path}: prompt corpus is empty")
    return prompts


class NgramLanguageModel:
    def __init__(
        self,
        *,
        context_counts: Counter[tuple[str, ...]],
        ngram_counts: Counter[tuple[str, ...]],
        vocab: set[str],
    ):
        self.context_counts = context_counts
        self.ngram_counts = ngram_counts
        self.vocab = set(vocab) | {_END}

    @classmethod
    def fit(cls, prompts: Iterable[str]) -> "NgramLanguageModel":
        context_counts: Counter[tuple[str, ...]] = Counter()
        ngram_counts: Counter[tuple[str, ...]] = Counter()
        vocab: set[str] = set()
        for prompt in prompts:
            tokens = _tokens(prompt)
            vocab.update(tokens)
            padded = [_START] * (_ORDER - 1) + tokens + [_END]
            for index in range(_ORDER - 1, len(padded)):
                context = tuple(padded[index - (_ORDER - 1) : index])
                token = padded[index]
                context_counts[context] += 1
                ngram_counts[(*context, token)] += 1
        return cls(context_counts=context_counts, ngram_counts=ngram_counts, vocab=vocab)

    def perplexity(self, prompt: str) -> float:
        tokens = _tokens(prompt)
        padded = [_START] * (_ORDER - 1) + tokens + [_END]
        log_prob = 0.0
        count = 0
        vocab_size = max(1, len(self.vocab))
        for index in range(_ORDER - 1, len(padded)):
            context = tuple(padded[index - (_ORDER - 1) : index])
            token = padded[index]
            numerator = self.ngram_counts[(*context, token)] + 1.0
            denominator = self.context_counts[context] + vocab_size
            log_prob += math.log(numerator / denominator)
            count += 1
        return float(math.exp(-log_prob / max(1, count)))


def _tokens(prompt: str) -> list[str]:
    return [match.group(0).lower() for match in _TOKEN_RE.finditer(prompt)]


def _prompts_from_json(value: object) -> list[str]:
    if isinstance(value, list):
        prompts: list[str] = []
        for item in value:
            prompt = _prompt_from_json_value(item)
            if prompt is not None:
                prompts.append(prompt)
        return prompts
    if isinstance(value, dict):
        for key in ("prompts", "samples", "rows"):
            nested = value.get(key)
            if isinstance(nested, list):
                return _prompts_from_json(nested)
        prompts = []
        for item in value.values():
            prompt = _prompt_from_json_value(item)
            if prompt is not None:
                prompts.append(prompt)
        return prompts
    return []


def _prompt_from_line(line: str) -> str | None:
    try:
        value = json.loads(line)
    except json.JSONDecodeError:
        return line
    return _prompt_from_json_value(value)


def _prompt_from_json_value(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value
    if isinstance(value, dict):
        for key in ("prompt", "text", "instruction", "query"):
            prompt = value.get(key)
            if isinstance(prompt, str) and prompt.strip():
                return prompt
    return None


def _bucket_thresholds(scores: list[float]) -> tuple[float, float]:
    ordered = sorted(float(score) for score in scores if math.isfinite(float(score)))
    if not ordered:
        return (0.0, 0.0)
    return (_quantile(ordered, 1.0 / 3.0), _quantile(ordered, 2.0 / 3.0))


def _quantile(ordered: list[float], q: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _bucket(score: float, *, thresholds: tuple[float, float]) -> str:
    low, medium = thresholds
    if score <= low:
        return "low"
    if score <= medium:
        return "medium"
    return "high"
