# Ten-minute quickstart

The goal is a complete loop: run a paired evaluation, read its metrics, inspect one failure category, and audit the bundle.

## 1. Run the fixture

```bash
uv run turnkey run --config configs/runs/smoke.yaml
```

Copy the printed run directory into `RUN_DIR`:

```bash
RUN_DIR=outputs/<run-id>
```

## 2. Read the result

```bash
python -m json.tool "$RUN_DIR/metrics.json"
uv run turnkey inspect "$RUN_DIR"
```

Turnkey compares a reference path with an intervention path on the same selected and attacked input. This makes a detector's added safety, benign blocking, and runtime cost visible in the same record.

## 3. Drill into a category

```bash
uv run turnkey inspect "$RUN_DIR" --category missed_harm
```

If the fixture has no row in that category, try `blocked_benign`, `blocked_harm`, or inspect a specific `--case-id` from the unfiltered output.

## 4. Audit independently

```bash
uv run turnkey audit "$RUN_DIR"
```

An empty JSON list means the bundle passed the current audit checks. Audit recomputes metrics and validates redaction, identity, event relationships, and measured forward counts; it does not trust a precomputed summary.

Next: [build your own detector](../guides/build-a-detector.md) or run the [quickstart notebook](../notebooks/00_quickstart.ipynb).
