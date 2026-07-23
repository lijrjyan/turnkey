#!/usr/bin/env python3
"""
Reproduce RCS (Representational Contrastive Scoring) - CI-safe toy mode.

Reference paper: arXiv:2512.12069
"""

from __future__ import annotations

import argparse

from turnkey.components.detectors import load_detector
from turnkey.config import DetectorConfig
from turnkey.schema import ImageInput, Sample


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--method", default="mcd", choices=["mcd", "kcd"])
    p.add_argument("--threshold", type=float, default=0.0)
    p.add_argument("--with-image", action="store_true")
    args = p.parse_args(argv)

    det = load_detector(
        DetectorConfig(
            name="rcs",
            params={
                "mode": "toy",
                "method": args.method,
                "threshold": args.threshold,
                "prototype_image_count": 1 if args.with_image else 0,
            },
        )
    )

    images = (ImageInput(path="__demo__"),) if args.with_image else ()
    samples = [
        Sample(sample_id="rcs-demo-0", behavior_id="rcs:demo:benign", is_benign=True, prompt="Say hello.", images=images),
        Sample(
            sample_id="rcs-demo-1",
            behavior_id="rcs:demo:unsafe_placeholder",
            is_benign=False,
            prompt="UNSAFE_PLACEHOLDER: request redacted (demo).",
            images=images,
        ),
    ]

    for s in samples:
        d = det.decide(s)
        print(f"{s.sample_id}\tblock={d.block}\tscore={d.score}\treason={d.reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
