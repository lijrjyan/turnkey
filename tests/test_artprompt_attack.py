from __future__ import annotations

from turnkey.components.attacks import load_attack
from turnkey.config import AttackConfig
from turnkey.schema import Sample


def test_artprompt_attack_wraps_prompt_and_updates_metadata() -> None:
    attack = load_attack(AttackConfig(name="artprompt", params={"masked_word": "SAFE"}))
    sample = Sample(
        sample_id="s1",
        behavior_id="b1",
        is_benign=True,
        prompt="Say [MASK] as a greeting.",
        attack_family="T0",
        attack_method="none",
        attack_params={"x": 1},
    )
    out = attack.apply(sample)

    assert out.attack_method == "artprompt"
    assert out.attack_family == "T0"  # default: preserve unless overridden
    assert out.attack_params["base_attack_method"] == "none"
    assert out.attack_params["artprompt_masked_word"] == "SAFE"

    assert "ASCII art" in out.prompt
    assert "Instruction:" in out.prompt
    assert "Say [MASK] as a greeting." in out.prompt
