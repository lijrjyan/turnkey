#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from turnkey.boreiko_ngram_resource import build_boreiko_ngram_manifest


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a manifest for the Boreiko n-gram parquet resource")
    parser.add_argument("--repo", default="references/repos/llm-threat-model")
    parser.add_argument("--base-path", default="artifacts/paper/boreiko/ngrams_results_final/joined_final")
    parser.add_argument("--out", default="artifacts/paper/boreiko/ngram_resource.manifest.json")
    parser.add_argument("--lock", default="references/LOCK.json")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    manifest = build_boreiko_ngram_manifest(
        repo_path=Path(args.repo),
        base_path=Path(args.base_path),
        out_path=Path(args.out),
        lock_path=Path(args.lock) if args.lock else None,
    )
    print(
        json.dumps(
            {
                "out": args.out,
                "ready": manifest["ready"],
                "missing": manifest["missing_files"],
                "invalid": manifest.get("invalid_files", []),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
