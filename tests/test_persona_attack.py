from __future__ import annotations

from turnkey.components.attacks import load_attack
from turnkey.config import AttackConfig
from turnkey.schema import Sample


def test_persona_attack_wraps_instruction_and_updates_metadata() -> None:
    attack = load_attack(AttackConfig(name="persona", params={"persona": "Helpful Teacher"}))
    sample = Sample(
        sample_id="s1",
        behavior_id="b1",
        is_benign=True,
        prompt="Explain the water cycle in one sentence.",
        attack_family="T0",
        attack_method="none",
    )
    out = attack.apply(sample)

    assert out.attack_method == "persona"
    assert out.attack_family == "T0"
    assert out.attack_params["base_attack_method"] == "none"
    assert out.attack_params["persona_persona"] == "Helpful Teacher"
    assert out.attack_params["persona_use_markers"] is True
    assert out.prompt.startswith("[System]")
    assert "Explain the water cycle" in out.prompt


def test_persona_attack_can_disable_markers() -> None:
    attack = load_attack(AttackConfig(name="persona", params={"persona": "Helpful Teacher", "use_markers": False}))
    sample = Sample(sample_id="s1", behavior_id="b1", is_benign=True, prompt="Hello.")
    out = attack.apply(sample)
    assert out.attack_params["persona_use_markers"] is False
    assert "[System]" not in out.prompt
