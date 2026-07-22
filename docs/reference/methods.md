# Included methods

Turnkey ships a compact set of methods that exercise different runtime capabilities.

| Method | Signal or control pattern | Best use |
|---|---|---|
| `allow_all` | No-op baseline | Pipeline and metric sanity |
| `keyword` | Local prompt rule | First detector and CI smoke |
| `smoothllm_v3` | Perturbation and repeated target calls | Policy control flow |
| `jailguard_v3` | Mutation and divergence | Multi-sample detector behavior |
| `gradsafe_v3` | Gradient provider | White-box gradient signals |
| `rcs_toy_v3` | CPU-safe representation proxy | Lightweight method development |
| `rcs_paper_v3` | Hidden-state provider and calibration | White-box representation methods |

These implementations demonstrate the harness; they do not form a leaderboard claim. Each method still depends on its original paper, model, dataset, and license conditions for strict scientific reproduction.
