Refs: #<primary-tracking-issue>
<!-- Closes: #<completed-issue> (optional; only when all acceptance items are complete) -->

## Summary

- 1–3 bullets: what changed and why. No plan restatement.

## Verification

- [ ] `uv run make test`
- [ ] `uv run make lint`
- [ ] Offline small-model smoke (required if runtime path is affected; see `AGENTS.md`):
  `HF_HUB_OFFLINE=1 RUN_DIR=$(uv run turnkey run --config configs/runs/qwen3_0_6b_pipeline_smoke.yaml) && uv run turnkey audit "$RUN_DIR"`

## Notes / Risks

- Only if non-trivial. Skip the section if there are none.

<!--
Title: feat|fix|perf|docs|test|chore: <short imperative>. No [codex]/[ai] prefixes.
Detailed design lives in code, docs, or the local refactor records.
Roadmap work references its owning track. Several PRs may share one track.
Create a PR only when explicitly requested; direct verified commits are the default.
-->
