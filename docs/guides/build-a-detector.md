# Build an external detector

Start outside the Turnkey package. A file entrypoint gives fast iteration and records the exact source bytes used by a run.

The fastest path creates a complete CPU-safe project without overwriting an
existing directory:

```bash
uv run turnkey init my-detector
cd my-detector
uv run turnkey dev ./detector.py:build --max-samples 4
```

The scaffold contains `detector.py`, `run.yaml`, and a short README. Continue
below to understand and change the generated detector.

Create `my_detector.py`:

```python
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
```

Run the bounded development loop:

```bash
uv run turnkey dev ./my_detector.py:build --max-samples 4
```

For a formal run, set `detector.name` to `./my_detector.py:build` and pass JSON-serializable constructor arguments under `detector.params`. Every configured argument must be preserved in `Component.parameters` so another run can identify the effective method.

Use a custom Policy only when the method must block before generation, rewrite the target request, call the target multiple times, select a response, or request model-side state. See [Policies and typed signals](../concepts/runtime.md).
