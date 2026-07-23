#!/usr/bin/env python3
"""
Reproduce Crescendo (multi-turn prompt template attack).

Reference paper: references/papers/2404.01833_Crescendo.pdf

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
    p.add_argument("--topic", default="the water cycle")
    p.add_argument("--mode", default="template", choices=["tag", "template", "replay"])
    p.add_argument("--prompt-map-path", default=None, help="For mode=replay; keep private.")
    p.add_argument("--key-field", default="sample_id")
    p.add_argument("--no-stub-responses", action="store_true")
    p.add_argument("--no-final-assistant-tag", action="store_true")
    args = p.parse_args(argv)

    params: dict[str, object] = {
        "mode": args.mode,
        "topic": args.topic,
        "key_field": args.key_field,
        "include_stub_responses": (not args.no_stub_responses),
        "include_final_assistant_tag": (not args.no_final_assistant_tag),
    }
    if isinstance(args.prompt_map_path, str) and args.prompt_map_path.strip():
        params["prompt_map_path"] = args.prompt_map_path

    attack = load_attack(AttackConfig(name="crescendo", params=params))
    sample = Sample(sample_id="crescendo-demo-0", behavior_id="crescendo:demo", is_benign=True, prompt=args.instruction)
    out = attack.apply(sample)
    print(out.prompt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
