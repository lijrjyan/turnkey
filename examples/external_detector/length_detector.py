from __future__ import annotations

from dataclasses import dataclass

from turnkey.components.detectors.base import Detector
from turnkey.schema import DetectorDecision, Sample


@dataclass(frozen=True)
class LengthDetector(Detector):
    threshold: int = 120

    def decide(self, sample: Sample) -> DetectorDecision:
        score = min(len(sample.prompt) / self.threshold, 1.0)
        return DetectorDecision(
            block=len(sample.prompt) >= self.threshold,
            score=score,
            reason="prompt_length",
        )


def build(*, threshold: int = 120):
    detector = LengthDetector(threshold=threshold)
    return detector.component(name="length-detector")
