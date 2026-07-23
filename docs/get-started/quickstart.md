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

## 4. Open the standalone report

```bash
uv run turnkey report "$RUN_DIR" --html report.html
```

Open `report.html` locally. It is a single file with no CDN or JavaScript. The
case rows expand into redacted policy, request, judge, timing, cache, and
model-forward events.

## 5. Audit independently

```bash
uv run turnkey audit "$RUN_DIR"
```

An empty JSON list means the bundle passed the current audit checks. Audit recomputes metrics and validates redaction, identity, event relationships, and measured forward counts; it does not trust a precomputed summary.

Next: scaffold a detector with `uv run turnkey init my-detector`, follow
[build your own detector](../guides/build-a-detector.md), or run the
[quickstart notebook](../notebooks/00_quickstart.ipynb).
