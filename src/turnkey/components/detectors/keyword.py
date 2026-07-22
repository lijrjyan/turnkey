from __future__ import annotations

from dataclasses import dataclass

from turnkey.components.detectors.base import Detector, DetectorManifest
from turnkey.schema import DetectorDecision, Sample


@dataclass(frozen=True)
class KeywordDetector(Detector):
    keywords: tuple[str, ...]

    def manifest(self, *, name: str | None = None) -> DetectorManifest:
        return DetectorManifest(
            name=name or "keyword",
            required_inputs=("sample", "prompt"),
            reproducibility={"keywords": list(self.keywords)},
        )

    def decide(self, sample: Sample) -> DetectorDecision:
        text = sample.prompt.lower()
        for kw in self.keywords:
            if kw.lower() in text:
                return DetectorDecision(block=True, score=1.0, reason=f"keyword:{kw}")
        return DetectorDecision(block=False, score=0.0, reason="no_keyword")
