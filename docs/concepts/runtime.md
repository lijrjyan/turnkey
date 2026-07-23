# Policies and typed signals

A detector is a `Component`: one Policy, zero or more typed providers, replay-relevant parameters, and optional cleanup.

```text
PolicyRequest -> Policy chain -> TargetSession -> Outcome
                         |
                         +-> MethodContext.get(TypedRequest)
                                      |
                                      +-> Provider -> cached value + event
```

Simple classifiers implement `Detector.decide()` and use the default `DetectorPolicy`. A method implements a Policy when it controls generation or needs model-side state.

Typed requests are frozen, hashable value objects. Their concrete Python type selects a provider; equality controls cache reuse. Providers own expensive resources and close after their last runtime use. Turnkey records cache hits, duration, and declared model forwards for every materialized request.

This separates method logic from model integration: a Policy asks for a signal, while a provider decides how to load a model and produce it. The [typed signals notebook](../notebooks/03_typed_signals.ipynb) demonstrates the contract without a GPU.
