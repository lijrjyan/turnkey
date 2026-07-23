"""Optional integration test for RCS paper-form mode.

Skipped by default because it downloads HF weights.
Enable with: `TURNKEY_RUN_RCS_PAPER_TESTS=1`.
"""

from __future__ import annotations

import os
import pytest

if not os.environ.get("TURNKEY_RUN_RCS_PAPER_TESTS"):
    pytest.skip("RCS paper-form integration tests are opt-in.", allow_module_level=True)

from turnkey.config import DetectorConfig
from turnkey.components.detectors import load_detector
from turnkey.schema import Sample


def test_rcs_paper_runs_on_tiny_text_model() -> None:
    det = load_detector(
        DetectorConfig(
            name="rcs",
            params={
                "mode": "paper",
                "method": "kcd",
                "k": 1,
                "threshold": 0.0,
                "calibrate_threshold": False,
                "val_ratio": 0.0,
                "model": {"model_id": "sshleifer/tiny-gpt2", "device": "cpu", "trust_remote_code": False},
                "projection_epochs": 1,
                "projection_batch_size": 4,
                "auto_layer_max_samples": 8,
                "auto_layer_svm_iters": 10,
                "benign_prompts": [
                    "Say hello in one short sentence.",
                    "Explain the water cycle in one sentence.",
                ],
                "malicious_prompts": [
                    "UNSAFE_PLACEHOLDER: request redacted (integration test).",
                    "UNSAFE_PLACEHOLDER: request redacted (integration test) #2.",
                ],
            },
        )
    )

    sample = Sample(sample_id="s", behavior_id="b", is_benign=True, prompt="Say hello in one short sentence.")
    out = det.decide(sample)
    assert isinstance(out.block, bool)
    assert isinstance(out.score, float)
    assert isinstance(out.reason, str)
