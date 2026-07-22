#!/usr/bin/env python3
"""
Reproduce DeepInception (prompt template attack).

Reference repo: references/repos/DeepInception

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
    p.add_argument("--layers", type=int, default=5)
    p.add_argument("--min-characters", type=int, default=5)
    p.add_argument("--scene", default="dream")
    args = p.parse_args(argv)

    attack = load_attack(
        AttackConfig(
            name="deepinception",
            params={"layers": args.layers, "min_characters": args.min_characters, "scene": args.scene},
        )
    )
    sample = Sample(sample_id="di-demo-0", behavior_id="deepinception:demo", is_benign=True, prompt=args.instruction)
    out = attack.apply(sample)
    print(out.prompt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
