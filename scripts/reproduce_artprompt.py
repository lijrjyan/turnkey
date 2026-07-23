#!/usr/bin/env python3
"""
Reproduce ArtPrompt (ASCII art jailbreak prompt transformation).

Reference repo: references/repos/ArtPrompt

This script is CI-safe: it only generates the *prompt* (prompt-only), without calling any LLMs.
"""

from __future__ import annotations

import argparse

from turnkey.components.attacks import load_attack
from turnkey.config import AttackConfig
from turnkey.schema import Sample


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--instruction", default="Say [MASK] as a greeting.")
    p.add_argument("--masked-word", default="SAFE")
    p.add_argument("--mask-token", default="[MASK]")
    p.add_argument("--replace-substring", default=None)
    args = p.parse_args(argv)

    params = {"masked_word": args.masked_word, "mask_token": args.mask_token}
    if args.replace_substring is not None:
        params["replace_substring"] = args.replace_substring

    attack = load_attack(AttackConfig(name="artprompt", params=params))
    sample = Sample(sample_id="demo-0", behavior_id="artprompt:demo", is_benign=True, prompt=args.instruction)
    out = attack.apply(sample)
    print(out.prompt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
