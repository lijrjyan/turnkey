# Included methods

Turnkey ships a compact set of methods that exercise different runtime capabilities.

| Method | Signal or control pattern | Best use |
|---|---|---|
| `allow_all` | No-op baseline | Pipeline and metric sanity |
| `keyword` | Local prompt rule | First detector and CI smoke |
| `smoothllm` | Perturbation and repeated target calls | Policy control flow |
| `jailguard` | Mutation and divergence | Multi-sample detector behavior |
| `gradsafe` | Gradient provider | White-box gradient signals |
| `perplexity` | Typed prompt log-probabilities | Input anomaly baseline |
| `self_exam` | Deterministic target self-screening | Output-side policy flow |
| `llamaguard` | Separate guard-model provider | Input safety classification |
| `rcs_toy` | CPU-safe representation proxy | Lightweight method development |
| `rcs` | Hidden-state provider and calibration | White-box representation methods |

These implementations demonstrate the harness; they do not form a leaderboard claim. Each method still depends on its original paper, model, dataset, and license conditions for strict scientific reproduction.
