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
| Matrix planning | `matrix/tiny_v5.yaml` |

Files with explicit model names, version suffixes, or reproduction labels are pinned research profiles. Use them to reproduce that exact lane, not as beginner templates.
