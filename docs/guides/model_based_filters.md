# Model-based filters

Turnkey includes three small, inspectable examples of model-based safety
policies. Their public detector names are stable; model revisions and experiment
campaigns belong in configuration rather than in versioned detector aliases.

## Perplexity filter

`perplexity` requests prompt token log-probabilities from the target backend,
computes `exp(-mean_logprob)`, and blocks when the result is strictly greater
than `threshold`. The provider is typed and runner-owned, so an identical request
is computed once and then served from the method cache.

```yaml
detector:
  name: perplexity
  params:
    threshold: 100.0
```

This is the plain perplexity baseline discussed in *Detecting Language Model
Attacks with Perplexity*. It is useful for learning and comparison, but the
paper reports important false-positive limitations; a threshold is not a
universal safety boundary.

## Self-Exam

`self_exam` first obtains the target response and then makes one deterministic
screening request with the response followed by a yes/no harm question. A
leading `yes` blocks and a leading `no` returns the original response. Ambiguous
labels fail closed by default.

```yaml
detector:
  name: self_exam
  params:
    max_new_tokens: 8
    fail_closed: true
```

This is an output-side policy based on the LLM Self Defense prompt ordering. It
adds one screening request per distinct sample; equal deterministic requests can
reuse the method cache.

## Llama Guard

`llamaguard` classifies the input prompt with a separate, lazy-loaded Llama Guard
runtime. It blocks `unsafe`, allows `safe`, and fails closed on an unparseable
label by default. Keep an immutable Hugging Face revision in reproducible run
configuration.

```yaml
detector:
  name: llamaguard
  params:
    model_id: meta-llama/Llama-Guard-3-1B
    revision: acf7aafa60f0410f8f42b1fa35e077d705892029
    token_env: HF_TOKEN
```

The model is gated. Use an approved Hugging Face login or token, and do not put
tokens in YAML or committed environment files. Detector diagnostics contain the
parsed label and categories, never the raw prompt or model output.

## Choose the boundary deliberately

| Method | Screening boundary | Extra model work |
| --- | --- | --- |
| `perplexity` | Input prompt | One prompt-logprob forward on a cache miss |
| `self_exam` | Generated output | One target generation plus one screening request |
| `llamaguard` | Input prompt | One separate guard-model forward |

Start with the corresponding `configs/runs/smoke_*.yaml`, inspect
`events.jsonl`, and audit the resulting run directory before changing thresholds
or model revisions.
