#!/usr/bin/env python3
"""
Reproduce RCS (Representational Contrastive Scoring) - paper-form.

Reference paper: arXiv:2512.12069

Notes:
- This script is CI-safe by default (uses a tiny text-only HF model).
- For real LVLM reproduction, pass a multimodal HF model id and enable `--with-image`.
"""

from __future__ import annotations

import argparse

from turnkey.config import DetectorConfig
from turnkey.components.detectors import load_detector
from turnkey.schema import ImageInput, Sample


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--method", default="mcd", choices=["mcd", "kcd"])
    p.add_argument("--model-id", default="sshleifer/tiny-gpt2")
    p.add_argument("--device", default="cpu")
    p.add_argument("--trust-remote-code", action="store_true")

    p.add_argument("--with-image", action="store_true")
    p.add_argument("--image-path", default="src/turnkey/components/datasets/_assets/mm_smoke.ppm")
    p.add_argument("--image-token", default="<image>")

    p.add_argument("--projection-epochs", type=int, default=10)
    p.add_argument("--projection-batch-size", type=int, default=8)
    p.add_argument("--auto-layer-svm-iters", type=int, default=50)
    p.add_argument("--calibrate-threshold", action="store_true")
    args = p.parse_args(argv)

    if args.with_image:
        images = (ImageInput(path=args.image_path),)
        benign_prompts = [
            f"{args.image_token}\nDescribe the image in one short sentence.",
            f"{args.image_token}\nWhat colors are present?",
        ]
        malicious_prompts = [
            f"{args.image_token}\nUNSAFE_PLACEHOLDER: request redacted (fixture).",
            f"{args.image_token}\nUNSAFE_PLACEHOLDER: request redacted (fixture) #2.",
        ]
        proto_image_count = 1
        proto_image_path = args.image_path
    else:
        images = ()
        benign_prompts = [
            "Say hello in one short sentence.",
            "Explain the water cycle in one sentence.",
        ]
        malicious_prompts = [
            "UNSAFE_PLACEHOLDER: request redacted (fixture).",
            "UNSAFE_PLACEHOLDER: request redacted (fixture) #2.",
        ]
        proto_image_count = 0
        proto_image_path = None

    det = load_detector(
        DetectorConfig(
            name="rcs",
            params={
                "mode": "paper",
                "method": args.method,
                "k": 50,
                "model": {
                    "model_id": args.model_id,
                    "device": args.device,
                    "trust_remote_code": bool(args.trust_remote_code),
                },
                "image_token": args.image_token,
                "prototype_image_count": proto_image_count,
                "prototype_image_path": proto_image_path,
                "benign_prompts": benign_prompts,
                "malicious_prompts": malicious_prompts,
                "projection_epochs": int(args.projection_epochs),
                "projection_batch_size": int(args.projection_batch_size),
                "auto_layer_max_samples": 8,
                "auto_layer_svm_iters": int(args.auto_layer_svm_iters),
                "val_ratio": 0.2 if args.calibrate_threshold else 0.0,
                "calibrate_threshold": bool(args.calibrate_threshold),
            },
        )
    )

    samples = [
        Sample(sample_id="rcs-paper-0", behavior_id="rcs:demo:benign", is_benign=True, prompt=benign_prompts[0], images=images),
        Sample(sample_id="rcs-paper-1", behavior_id="rcs:demo:unsafe", is_benign=False, prompt=malicious_prompts[0], images=images),
    ]

    for s in samples:
        d = det.decide(s)
        print(f"{s.sample_id}\tblock={d.block}\tscore={d.score}\treason={d.reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
