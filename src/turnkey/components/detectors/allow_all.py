from __future__ import annotations

from turnkey.components.detectors.base import Detector, DetectorManifest
from turnkey.schema import DetectorDecision, Sample


class AllowAllDetector(Detector):
    def manifest(self, *, name: str | None = None) -> DetectorManifest:
        return DetectorManifest(
            name=name or "allow_all",
            required_inputs=("sample",),
        )

    def decide(self, sample: Sample) -> DetectorDecision:  # noqa: ARG002
        return DetectorDecision(block=False, score=0.0, reason="allow_all")
