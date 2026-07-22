from __future__ import annotations

import json

from turnkey.components.attacks import load_attack
from turnkey.config import AttackConfig
from turnkey.schema import Sample


def test_crescendo_attack_template_wraps_instruction_and_updates_metadata() -> None:
    attack = load_attack(AttackConfig(name="crescendo", params={"mode": "template", "topic": "the water cycle"}))
    sample = Sample(
        sample_id="s1",
        behavior_id="b1",
        is_benign=True,
        prompt="Explain the water cycle in one sentence.",
        attack_family="T0",
        attack_method="none",
    )
    out = attack.apply(sample)

    assert out.attack_method == "crescendo"
    assert out.attack_family == "T0"
    assert out.attack_params["base_attack_method"] == "none"
    assert out.attack_params["crescendo_mode"] == "template"
    assert out.attack_params["crescendo_topic"] == "the water cycle"
    assert "Tell me about" in out.prompt
    assert "Explain the water cycle" in out.prompt


def test_crescendo_attack_tag_mode_does_not_modify_prompt() -> None:
    attack = load_attack(AttackConfig(name="crescendo", params={"mode": "tag", "topic": "X"}))
    sample = Sample(sample_id="s1", behavior_id="b1", is_benign=True, prompt="Hello.", attack_method="none")
    out = attack.apply(sample)
    assert out.attack_method == "crescendo"
    assert out.prompt == "Hello."


def test_crescendo_attack_replay_uses_prompt_map(tmp_path) -> None:
    prompt_map = {"s1": "REPLAYED PROMPT"}
    path = tmp_path / "map.json"
    path.write_text(json.dumps(prompt_map), encoding="utf-8")

    attack = load_attack(
        AttackConfig(
            name="crescendo",
            params={"mode": "replay", "topic": "X", "prompt_map_path": str(path), "key_field": "sample_id"},
        )
    )
    sample = Sample(sample_id="s1", behavior_id="b1", is_benign=True, prompt="Hello.")
    out = attack.apply(sample)
    assert out.prompt == "REPLAYED PROMPT"
