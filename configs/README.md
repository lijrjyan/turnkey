# Configuration map

Start with a task, not the directory listing.

| Goal | Config |
|---|---|
| First CPU-only run | `runs/smoke.yaml` |
| External detector | copy `runs/smoke.yaml`, replace `detector.name` with `path.py:build` |
| Local Hugging Face model | `runs/smoke_hf.yaml` |
| vLLM/SGLang/OpenAI-compatible server | `runs/smoke_openai_compat.yaml` |
| Prompt logprobs | `runs/smoke_logprobs.yaml` |
| Hidden-state provider | `runs/smoke_rcs_paper.yaml` |
| Gradient provider | `runs/smoke_gradsafe.yaml` |
| Matrix planning | `matrix/example.yaml` |

The repository intentionally ships generic examples only. Model-specific,
versioned, and paper-campaign profiles belong in a private experiment workspace,
where their model caches, gated data, and local artifacts can be managed without
turning historical runs into public defaults.
