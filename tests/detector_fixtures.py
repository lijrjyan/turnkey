from __future__ import annotations

from turnkey.schema import Sample


CONFORMANCE_SAMPLES = [
    Sample(
        sample_id="benign",
        behavior_id="fixtures:benign",
        is_benign=True,
        prompt="Say hello in one short sentence.",
    ),
    Sample(
        sample_id="unsafe",
        behavior_id="fixtures:unsafe",
        is_benign=False,
        prompt="UNSAFE_PLACEHOLDER: request redacted.",
    ),
]
