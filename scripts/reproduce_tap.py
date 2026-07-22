#!/usr/bin/env python3
"""
Reproduce TAP (Tree of Attacks with Pruning).

Paper / reference implementation lives at: references/repos/TAP/main_TAP.py

This repo provides:
1) A "reference-cmd" mode that prints the paper-form command (requires upstream deps + API keys).
2) A CI-safe "toy" mode that reproduces TAP's *algorithmic skeleton* (branch → prune → query → prune)
   without external LLM/API dependencies. (EXTRA: toy scorer + toy target.)

Safety note:
- Do NOT commit raw jailbreak artifacts or harmful prompts. Keep generated prompt maps local.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

from turnkey.config import DatasetConfig
from turnkey.components.datasets import load_dataset


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z0-9_]+", text.lower()))


def _on_topic_score(prompt: str, goal: str) -> float:
    a = _tokens(prompt)
    b = _tokens(goal)
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return 10.0 * (inter / union)


def _toy_target(prompt: str, *, goal: str, target_str: str) -> str:
    # EXTRA: deterministic stub. "Jailbreak success" triggers when the prompt contains a magic marker.
    if "MAGIC_SUCCESS_MARKER" in prompt:
        return f"{target_str} {goal}"
    return "I'm sorry, I can't help with that."


def _judge_score(prompt: str, response: str, *, target_str: str) -> float:
    if response.startswith(target_str):
        return 10.0
    # Prefer "more on-topic" prompts as a weak proxy.
    return 1.0 if target_str in prompt else 0.0


def _branch(prompts: list[str], *, goal: str, branching_factor: int, rng: random.Random) -> list[str]:
    templates = [
        "Task: {goal}\nAnswer concisely.",
        "You are a helpful assistant.\nUser request: {goal}",
        "Please respond to: {goal}",
        # EXTRA: a toy marker to demonstrate early-stopping mechanics (not a real TAP prompt).
        "MAGIC_SUCCESS_MARKER\nRequest: {goal}",
    ]

    out: list[str] = []
    for base in prompts:
        for _ in range(branching_factor):
            t = rng.choice(templates)
            p = t.format(goal=goal)
            out.append((base + "\n\n" + p).strip() if base.strip() else p.strip())
    return out


def _prune(items: list[tuple[float, Any]], *, width: int) -> list[Any]:
    items = [x for x in items if x[0] > 0]
    items.sort(key=lambda x: x[0], reverse=True)
    return [v for _, v in items[: max(1, min(width, len(items)))]]


def toy_tap(
    *,
    goal: str,
    target_str: str,
    depth: int,
    width: int,
    branching_factor: int,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)

    # Start with an empty root prompt (mirrors reference impl shape).
    frontier = [""]
    query_count = 0

    for _depth in range(1, depth + 1):
        branched = _branch(frontier, goal=goal, branching_factor=branching_factor, rng=rng)

        # Prune phase 1 (on-topic)
        on_topic = [(float(_on_topic_score(p, goal)), p) for p in branched]
        kept = _prune(on_topic, width=width)

        # Query + assess
        responses: list[str] = []
        judge_scores: list[float] = []
        for p in kept:
            resp = _toy_target(p, goal=goal, target_str=target_str)
            responses.append(resp)
            query_count += 1
            judge_scores.append(float(_judge_score(p, resp, target_str=target_str)))

        # Prune phase 2 (judge score)
        frontier = _prune(list(zip(judge_scores, kept)), width=width)
        best_idx = max(range(len(kept)), key=lambda i: judge_scores[i])
        if judge_scores[best_idx] >= 10.0:
            return {
                "best_prompt": kept[best_idx],
                "best_response": responses[best_idx],
                "depth": _depth,
                "queries": query_count,
                "success": True,
            }

    # No success: return best-on-topic prompt.
    best_prompt = max(frontier, key=lambda p: _on_topic_score(p, goal)) if frontier else goal
    return {"best_prompt": best_prompt, "best_response": None, "depth": depth, "queries": query_count, "success": False}


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["toy", "reference-cmd"], default="toy")
    p.add_argument("--goal", default="Explain the water cycle in one sentence.")
    p.add_argument("--target-str", default="Sure, here is the answer:")
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--width", type=int, default=4)
    p.add_argument("--branching-factor", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--dataset", default=None, help="Optional: turnkey dataset name (e.g., fixtures_smoke)")
    p.add_argument("--dataset-n-samples", type=int, default=4)
    p.add_argument("--key-field", choices=["sample_id", "behavior_id"], default="sample_id")
    p.add_argument("--out", default=None, help="Optional: write a JSONL prompt map for turnkey tap replay")

    args = p.parse_args(argv)

    if args.mode == "reference-cmd":
        print("Paper-form TAP command (run inside references/repos/TAP with proper deps+keys):\n")
        print(
            "python main_TAP.py "
            "--attack-model gpt-4 "
            "--target-model gpt-4 "
            "--evaluator-model gpt-4 "
            f"--goal {json.dumps(args.goal)} "
            f"--target-str {json.dumps(args.target_str)} "
            f"--depth {args.depth} --width {args.width} --branching-factor {args.branching_factor}"
        )
        return 0

    rows: list[dict[str, Any]] = []
    if args.dataset:
        samples = load_dataset(DatasetConfig(name=args.dataset, params={"n_samples": args.dataset_n_samples}))
        for sample in samples:
            res = toy_tap(
                goal=sample.prompt,
                target_str=args.target_str,
                depth=args.depth,
                width=args.width,
                branching_factor=args.branching_factor,
                seed=args.seed,
            )
            rows.append(
                {
                    args.key_field: getattr(sample, args.key_field),
                    "prompt": res["best_prompt"],
                    "meta": {k: res[k] for k in ("success", "depth", "queries")},
                }
            )
    else:
        res = toy_tap(
            goal=args.goal,
            target_str=args.target_str,
            depth=args.depth,
            width=args.width,
            branching_factor=args.branching_factor,
            seed=args.seed,
        )
        print("Toy TAP result (EXTRA):")
        print(json.dumps(res, indent=2, ensure_ascii=False))
        if args.out:
            rows = [{args.key_field: "goal-0", "prompt": res["best_prompt"], "meta": res}]

    if args.out and rows:
        _write_jsonl(Path(args.out), rows)
        print(f"\nWrote prompt map: {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
