#!/usr/bin/env python3
"""
Reproduce Persona Modulation (prompt template attack).

Reference paper: references/papers/2311.03348_Persona_Modulation.pdf

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
    p.add_argument("--persona", default="Helpful Teacher")
    p.add_argument("--persona-prompt", default=None)
    p.add_argument("--no-markers", action="store_true", help="Do not include [System]/[User] plaintext markers.")
    args = p.parse_args(argv)

    params: dict[str, object] = {"persona": args.persona, "use_markers": (not args.no_markers)}
    if isinstance(args.persona_prompt, str) and args.persona_prompt.strip():
        params["persona_prompt"] = args.persona_prompt

    attack = load_attack(AttackConfig(name="persona", params=params))
    sample = Sample(sample_id="persona-demo-0", behavior_id="persona:demo", is_benign=True, prompt=args.instruction)
    out = attack.apply(sample)
    print(out.prompt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
