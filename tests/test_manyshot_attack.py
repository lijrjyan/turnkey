from __future__ import annotations

from turnkey.components.attacks import load_attack
from turnkey.config import AttackConfig
from turnkey.schema import Sample


def test_manyshot_attack_wraps_instruction_and_updates_metadata() -> None:
    attack = load_attack(
        AttackConfig(
            name="manyshot",
            params={
                "n_shots": 3,
                "shuffle": False,
                "demos": [
                    {"q": "Q1?", "a": "A1."},
                    {"q": "Q2?", "a": "A2."},
                ],
            },
        )
    )
    sample = Sample(
        sample_id="s1",
        behavior_id="b1",
        is_benign=True,
        prompt="FINAL?",
        attack_family="T0",
        attack_method="none",
    )
    out = attack.apply(sample)

    assert out.attack_method == "manyshot"
    assert out.attack_family == "T0"
    assert out.attack_params["base_attack_method"] == "none"
    assert out.attack_params["manyshot_n_shots"] == 3
    assert out.attack_params["manyshot_demos_source"] == "inline"

    assert out.prompt.count("Human:") == 4
    assert out.prompt.count("Assistant:") == 4
    assert "Human: FINAL?" in out.prompt


def test_manyshot_attack_shuffle_is_deterministic_per_sample() -> None:
    cfg = AttackConfig(name="manyshot", params={"n_shots": 4, "shuffle": True})
    attack = load_attack(cfg)
    sample = Sample(sample_id="s1", behavior_id="b1", is_benign=True, prompt="Hello.")
    out1 = attack.apply(sample)
    out2 = attack.apply(sample)
    assert out1.prompt == out2.prompt


def test_manyshot_attack_can_disable_final_assistant_tag() -> None:
    attack = load_attack(
        AttackConfig(
            name="manyshot",
            params={"n_shots": 2, "shuffle": False, "include_final_assistant_tag": False},
        )
    )
    sample = Sample(sample_id="s1", behavior_id="b1", is_benign=True, prompt="FINAL?")
    out = attack.apply(sample)
    assert out.prompt.rstrip().endswith("Human: FINAL?")
