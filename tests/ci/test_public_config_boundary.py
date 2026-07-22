from __future__ import annotations

from pathlib import Path
import re

import yaml


CAMPAIGN_RUN_PREFIXES = ("qwen", "v4_", "v9_", "jailguard_paper_")


def test_public_configs_contain_only_generic_examples() -> None:
    run_names = {path.name for path in Path("configs/runs").glob("*.yaml")}
    campaign_runs = sorted(
        name for name in run_names if name.startswith(CAMPAIGN_RUN_PREFIXES)
    )

    assert campaign_runs == []
    assert {path.name for path in Path("configs/matrix").glob("*.yaml")} == {
        "example.yaml"
    }

    for path in [*Path("configs/runs").glob("*.yaml"), Path("configs/matrix/example.yaml")]:
        config_text = yaml.safe_dump(yaml.safe_load(path.read_text(encoding="utf-8")))
        assert re.search(r"\b[a-z][a-z0-9_]*_v[0-9]+\b", config_text) is None
