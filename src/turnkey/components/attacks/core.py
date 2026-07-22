from __future__ import annotations

from turnkey.config import AttackConfig
from turnkey.registry import Registry
from turnkey.schema import Sample


ATTACKS = Registry("attack")


class Attack:
    def apply(self, sample: Sample) -> Sample:
        raise NotImplementedError


def register_attack(name: str):
    def deco(factory):
        ATTACKS.register(name, factory)
        return factory

    return deco


def available_attacks() -> list[str]:
    return ATTACKS.list()


def load_attack(cfg: AttackConfig) -> Attack:
    return ATTACKS.get(cfg.name)(cfg)
