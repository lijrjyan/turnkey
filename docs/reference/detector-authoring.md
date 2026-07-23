# Detector And Defense Authoring

Turnkey executes every detector or defense through one Policy runtime. A simple
detector implements `decide()` and uses the default `DetectorPolicy`; a method
that blocks, rewrites, samples repeatedly, or consumes external model state
implements `policy()` and typed Requests from `method_providers()`.

## Minimal Detector

Subclass `Detector`, implement `decide()`, and register a builder in
`components/detectors/__init__.py`. The default Policy and Detector-to-Component
adapter are enough for a local classifier.

```python
from turnkey.components.detectors.base import Detector
from turnkey.schema import DetectorDecision, Sample


class MyDetector(Detector):
    def decide(self, sample: Sample) -> DetectorDecision:
        score = score_locally(sample.prompt)
        return DetectorDecision(
            block=score > 0.5,
            score=score,
            reason="my_detector",
        )
```

Public dataclass constructor fields become the Component's normalized effective
parameters, so keep them JSON-serializable and free of secrets, runtime objects,
or raw training payloads. Store an environment-variable name, artifact reference,
or content identity instead. A public constructor field that must not be persisted
can opt out with `field(metadata={"component_parameter": False})`. Loaded
calibration artifacts and their operating points are recorded and audited
separately; they may override a constructor threshold without rewriting the
Component parameters. Generated config records a hash of the complete normalized
parameter map so audit can detect changed defaults, injected values, and missing
values without another manifest schema. Runtime side effects are measured from
actual target and provider requests; decision fields and diagnostics come from
`DetectorDecision`, not a second manifest schema. Test the exact action, score,
reason, and diagnostics produced by representative samples.

## Built-in Alias Or External Component

Registration is optional. A built-in short name such as `keyword` resolves through
`components/detectors/__init__.py`; the registry is only an alias-to-builder table.
An external method can instead export a builder from an importable module or Python
file. A Python-file entrypoint may also export a concrete `Component` for a simple
single-file smoke:

```python
from turnkey.policy import Component


def build(*, threshold: float) -> Component:
    return Component(
        name="my-method",
        policy=MyPolicy(threshold),
        providers=(MyProvider(),),
        parameters={"threshold": threshold},
    )
```

```yaml
detector:
  name: ./my_method.py:build
  params:
    threshold: 0.5
```

`turnkey dev ./my_method.py:component` and normal `turnkey run` use the same
resolver and Policy runtime. Formal run config passes `detector.params` to a
builder as keyword arguments. A concrete exported `Component` therefore requires
empty params. Module entrypoints must export a builder because Python caches imported
module objects across runs. Every configured builder argument must appear under the
returned `Component.parameters`; use an environment-variable name or artifact identity
instead of passing a secret that cannot be persisted. Automatic calibration-artifact
injection remains a built-in adapter contract; an external builder must load its own
calibration state explicitly.

Turnkey records installed modules by distribution/version only when the resolved source
belongs to that distribution. Other importable modules are loaded from and identified by
their exact local source bytes. If local code was imported before Turnkey can establish
that identity, use a `path.py:builder` entrypoint or restart the process.

`run.json` persists both the requested entrypoint and its resolved Component/source
identity. Providers are closed after their last runtime
use; a Component that owns resources outside its providers may supply `cleanup=...`.

## Policy And Typed Requests

Use a Policy when a method blocks before generation, rewrites a target request,
performs extra target calls, returns a selected response, or needs externally
materialized state such as gradients or hidden states.

```python
from dataclasses import dataclass

from turnkey.methods import MethodContext, Request
from turnkey.policy import Outcome, PolicyRequest


@dataclass(frozen=True)
class ScoreRequest(Request[float]):
    prompt: str


class ScoreProvider:
    request_type = ScoreRequest
    model_forwards_per_call = 1

    def provide(self, request: ScoreRequest) -> float:
        return score_externally(request.prompt)


class MyPolicy:
    def apply(self, request: PolicyRequest, call_next, context: MethodContext) -> Outcome:
        score = context.get(ScoreRequest(request.sample.prompt))
        if score > 0.5:
            return Outcome.blocked(request.target, score=score)
        return call_next(request)
```

Requests must be frozen, hashable value objects. Equality should include only
inputs that can change the materialized result; IDs, labels, and diagnostic
metadata should not defeat safe cache reuse. Providers that own models must
implement `close()`. Set `requires_exclusive_target = True` when the generation
backend must be released before provider materialization.

Behavior tests for a Policy-native method should cover the Outcome action,
score, reason, diagnostics, target request, selected model output, actual
target/provider calls, cache reuse, and cleanup order. Full pipeline coverage
must include `run_eval(...)`, artifact audit, and measured forward counts.

`run.json` persists Component/source identity and effective parameters;
`events.jsonl` records each typed request use with its scope, provider,
cache-hit state, duration, and model-forward count. `turnkey audit` recomputes
`metrics.cost.extra_forwards_avg` from intervention request events; normal
runtime artifacts do not carry a second detector requirements or manifest
schema.

## Cache Sidecars

Calibration or provider preparation caches use `cache_manifest.json` as the
ready sentinel. Writers stage files in `<entry>.tmp/`, write the manifest at
commit time, and only then promote the directory. `cache_entry_ready(...)`
checks the ready flag, file paths, SHA256 values, and byte counts.

When a calibration artifact references a sidecar under a directory containing
`cache_manifest.json`, `turnkey audit` verifies the cache entry before treating
the run as reproducible.
