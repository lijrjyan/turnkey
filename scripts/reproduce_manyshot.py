#!/usr/bin/env python3
"""
Reproduce Many-shot Jailbreaking (prompt template attack).

Reference paper: references/papers/NeurIPS2024_Many-shot_Jailbreaking.pdf

This script is CI-safe: it only generates the attacked prompt (prompt-only).
"""

from __future__ import annotations

import argparse

from turnkey.components.attacks import load_attack
from turnkey.config import AttackConfig
from turnkey.schema import Sample


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--instruction", default="Explain the water cycle in one sentence.")
    p.add_argument("--n-shots", type=int, default=8)
    p.add_argument("--demos-path", default=None, help="Optional .json/.jsonl of {q,a} pairs (keep private).")
    p.add_argument("--no-shuffle", action="store_true")
    p.add_argument("--user-tag", default="Human:")
    p.add_argument("--assistant-tag", default="Assistant:")
    p.add_argument("--no-final-assistant-tag", action="store_true")
    args = p.parse_args(argv)

    params: dict[str, object] = {
        "n_shots": args.n_shots,
        "shuffle": (not args.no_shuffle),
        "user_tag": args.user_tag,
        "assistant_tag": args.assistant_tag,
        "include_final_assistant_tag": (not args.no_final_assistant_tag),
    }
    if isinstance(args.demos_path, str) and args.demos_path.strip():
        params["demos_path"] = args.demos_path

    attack = load_attack(AttackConfig(name="manyshot", params=params))
    sample = Sample(sample_id="manyshot-demo-0", behavior_id="manyshot:demo", is_benign=True, prompt=args.instruction)
    out = attack.apply(sample)
    print(out.prompt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
