from __future__ import annotations

from pathlib import Path
import re


DETECTOR_SOURCE = '''from __future__ import annotations

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
'''


def create_detector_project(destination: str | Path) -> Path:
    project_dir = Path(destination)
    try:
        project_dir.mkdir(parents=True)
    except FileExistsError as exc:
        raise FileExistsError(f"destination already exists: {project_dir}") from exc

    project_name = _project_name(project_dir.name)
    files = {
        "detector.py": DETECTOR_SOURCE,
        "run.yaml": _run_config(project_name),
        "README.md": _project_readme(project_name),
    }
    for relative_path, content in files.items():
        with (project_dir / relative_path).open("x", encoding="utf-8") as file:
            file.write(content)
    return project_dir


def _project_name(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return normalized or "my-detector"


def _run_config(project_name: str) -> str:
    return f'''run:
  name: {project_name}
  out_dir: outputs
  redact: true
  max_samples: 4

dataset:
  name: fixtures_smoke@v1
  params:
    n_samples: 4

model:
  backend: dummy
  model_id: dummy-smoke
  max_new_tokens: 32
  temperature: 0.0

nsg:
  enabled: true
  baseline_detector:
    name: allow_all
    params: {{}}

detector:
  name: ./detector.py:build
  params:
    threshold: 120

judge:
  name: dummy_refusal
  params: {{}}
'''


def _project_readme(project_name: str) -> str:
    return f'''# {project_name}

This scaffold is a runnable external Turnkey detector. It starts with a small
prompt-length policy so you can focus on the development loop before adding
model-side state.

```bash
turnkey dev ./detector.py:build --max-samples 4
turnkey run --config run.yaml
turnkey inspect outputs/<run-id> --json
```

Edit `LengthDetector.decide`, rerun the bounded `dev` command, and inspect the
paired reference/intervention artifacts before switching to a real model.
'''
