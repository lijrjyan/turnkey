#!/usr/bin/env bash
set -euo pipefail

# Runs a minimal end-to-end smoke on the key v1.0 dataset adapters.
# Safe-by-default: Turnkey redacts public artifacts; `--private` writes a local plaintext sidecar.

turnkey run --config configs/runs/jbb_smoke.yaml
turnkey run --config configs/runs/xstest_smoke.yaml
turnkey run --config configs/runs/sorrybench_smoke.yaml
