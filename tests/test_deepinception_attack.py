from __future__ import annotations

from turnkey.components.attacks import load_attack
from turnkey.config import AttackConfig
from turnkey.schema import Sample


def test_deepinception_attack_wraps_instruction_and_updates_metadata() -> None:
    attack = load_attack(AttackConfig(name="deepinception", params={"layers": 3, "min_characters": 5}))
    sample = Sample(
        sample_id="s1",
        behavior_id="b1",
        is_benign=True,
        prompt="Explain the water cycle in one sentence.",
        attack_family="T0",
        attack_method="none",
    )
    out = attack.apply(sample)

    assert out.attack_method == "deepinception"
    assert out.attack_family == "T0"
    assert out.attack_params["base_attack_method"] == "none"
    assert out.attack_params["deepinception_layers"] == 3

    assert "reach layer 3" in out.prompt
    assert "Explain the water cycle" in out.prompt
