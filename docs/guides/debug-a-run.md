# Debug a run

Turnkey's debugging unit is a case plus its related runtime events.

## Find the symptom

```bash
uv run turnkey inspect "$RUN_DIR" --category missed_harm --json
uv run turnkey inspect "$RUN_DIR" --category blocked_benign --json
```

Choose a `case_id`, then request the complete case view:

```bash
uv run turnkey inspect "$RUN_DIR" --case-id <case-id> --json
```

For a browser-readable view of the same redacted artifacts:

```bash
uv run turnkey report "$RUN_DIR" --html report.html
```

The standalone file expands each case into a runtime event timeline. It has no
external assets or JavaScript and does not restore plaintext removed by the
public artifact writer.

## Follow the event chain

`events.jsonl` records policy spans, typed request uses, target calls, cache hits, durations, and model-forward counts. Public events identify the operation without exposing plaintext prompts.

Ask four questions in order:

1. Did the detector receive the intended selected and attacked sample?
2. Did the Policy block, rewrite, or forward the expected target request?
3. Did each typed provider materialize once and reuse equal requests?
4. Does measured extra work match the method's expected cost?

## Change one variable

Adjust one threshold, method parameter, or input selection at a time. Re-run and compare the paired case and event records before interpreting aggregate metrics.

Finish with:

```bash
uv run turnkey audit "$RUN_DIR"
```

The [debug notebook](../notebooks/02_debug_a_missed_case.ipynb) walks through the same loop programmatically.
