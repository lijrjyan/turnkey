from __future__ import annotations

from collections.abc import Callable
from typing import Any


class Registry:
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, Callable[..., Any]] = {}

    def register(self, name: str, factory: Callable[..., Any]) -> None:
        if name in self._items:
            raise KeyError(f"Duplicate registration: {self.kind}:{name}")
        self._items[name] = factory

    def get(self, name: str) -> Callable[..., Any]:
        try:
            return self._items[name]
        except KeyError as e:
            raise KeyError(f"Unknown {self.kind}: {name}. Available: {sorted(self._items)}") from e

    def list(self) -> list[str]:
        return sorted(self._items)


BACKENDS = Registry("backend")
DATASETS = Registry("dataset")
DETECTORS = Registry("detector")
JUDGES = Registry("judge")


def register_backend(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def deco(factory: Callable[..., Any]) -> Callable[..., Any]:
        BACKENDS.register(name, factory)
        return factory

    return deco


def register_dataset(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def deco(factory: Callable[..., Any]) -> Callable[..., Any]:
        DATASETS.register(name, factory)
        return factory

    return deco


def register_detector(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def deco(factory: Callable[..., Any]) -> Callable[..., Any]:
        DETECTORS.register(name, factory)
        return factory

    return deco


def register_judge(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def deco(factory: Callable[..., Any]) -> Callable[..., Any]:
        JUDGES.register(name, factory)
        return factory

    return deco
