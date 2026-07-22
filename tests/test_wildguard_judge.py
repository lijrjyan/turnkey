"""Optional integration tests for the WildGuard judge adapter.

These tests are skipped by default because WildGuard weights are gated on HF.
Set `TURNKEY_RUN_WILDGUARD_TESTS=1` and provide appropriate HF auth to run.
"""

import os
import pytest

if not os.environ.get("TURNKEY_RUN_WILDGUARD_TESTS"):
    pytest.skip("WildGuard tests are opt-in (gated weights).", allow_module_level=True)

from turnkey.components.judges.wildguard import WildGuardJudge
from turnkey.schema import Sample


@pytest.fixture
def wildguard_judge():
    """Create a WildGuard judge instance (HF backend for testing)."""
    return WildGuardJudge(use_vllm=False, batch_size=2, device="cuda")


def test_wildguard_import():
    """Test that WildGuard can be imported."""
    judge = WildGuardJudge(use_vllm=False)
    assert judge is not None


def test_wildguard_refusal_detection(wildguard_judge):
    """Test WildGuard detects refusals correctly."""
    sample = Sample(
        sample_id="test_001",
        behavior_id="harmful_001",
        is_benign=False,
        prompt="UNSAFE_PLACEHOLDER: request redacted (integration test).",
        attack_family="T0",
        attack_method="none",
    )

    # Test refusal response
    refusal_text = "I'm sorry, but I can't help with that request."
    result = wildguard_judge.judge(sample=sample, model_text=refusal_text)

    assert result.is_refusal is True
    assert result.is_harmful_effective is False
    assert result.details["judge_backend"] == "wildguard"
    assert result.details["response_refusal"] == "refusal"


def test_wildguard_benign_prompt(wildguard_judge):
    """Test WildGuard handles benign prompts correctly."""
    sample = Sample(
        sample_id="test_003",
        behavior_id="benign_001",
        is_benign=True,
        prompt="What's the weather like today?",
        attack_family="T0",
        attack_method="none",
    )

    response_text = "The weather is sunny and warm."
    result = wildguard_judge.judge(sample=sample, model_text=response_text)

    assert result.is_harmful_effective is False
    assert result.details["prompt_harmfulness"] == "unharmful"


def test_wildguard_batch_processing(wildguard_judge):
    """Test WildGuard batch processing."""
    samples_and_texts = [
        (
            Sample(
                sample_id="batch_001",
                behavior_id="harmful_001",
                is_benign=False,
                prompt="UNSAFE_PLACEHOLDER: request redacted (integration test).",
            ),
            "I can't help with that.",
        ),
        (
            Sample(
                sample_id="batch_002",
                behavior_id="benign_001",
                is_benign=True,
                prompt="How to bake a cake?",
            ),
            "Here's a simple recipe...",
        ),
    ]

    results = wildguard_judge.judge_batch(samples_and_texts)

    assert len(results) == 2
    # First should be refusal
    assert results[0].is_refusal is True
    # Second should be benign
    assert results[1].is_harmful_effective is False
