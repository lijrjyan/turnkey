from __future__ import annotations

from typing import Any

from turnkey.schema import AttackBudget, Sample, ThreatContext


THREAT_TIERS = frozenset({"T0", "T1", "T2", "T3", "T4", "T5"})


def normalize_threat_tier(value: object) -> str:
    if isinstance(value, str):
        tier = value.strip().upper()
        if tier in THREAT_TIERS:
            return tier
    return "unknown"


def threat_context_for_sample(sample: Sample) -> ThreatContext:
    explicit_tier = sample.attack_params.get("threat_tier")
    tier = normalize_threat_tier(explicit_tier if explicit_tier is not None else sample.attack_family)
    return ThreatContext(
        tier=tier,
        attack_family=sample.attack_family,
        attack_method=sample.attack_method,
    )


def budget_for_sample(sample: Sample) -> AttackBudget:
    params = sample.attack_params
    queries = _first_int(params, ("query_budget", "queries", "num_queries"))
    steps = _first_int(params, ("steps", "tap_steps", "deepinception_layers"))
    turns = _first_int(params, ("turns", "crescendo_turns"))
    shots = _first_int(params, ("shots", "n_shots", "manyshot_n_shots"))
    suffix_len = _first_int(params, ("suffix_len", "suffix_length", "gcg_suffix_len"))
    extra_forwards = _first_int(params, ("extra_forwards", "detector_extra_forwards"))

    if turns is None and sample.attack_method == "crescendo":
        turns = 4 if params.get("crescendo_mode") == "template" else None

    return AttackBudget(
        queries=queries,
        steps=steps,
        turns=turns,
        shots=shots,
        suffix_len=suffix_len,
        extra_forwards=extra_forwards,
        bucket=_budget_bucket((queries, steps, turns, shots, suffix_len, extra_forwards)),
    )


def _first_int(params: dict[str, Any], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        value = params.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int) and value >= 0:
            return value
    return None


def _budget_bucket(values: tuple[int | None, ...]) -> str:
    present = [value for value in values if value is not None]
    if not present:
        return "none"
    peak = max(present)
    if peak <= 1:
        return "low"
    if peak <= 8:
        return "medium"
    return "high"
