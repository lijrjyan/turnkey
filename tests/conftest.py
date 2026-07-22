from __future__ import annotations

from pathlib import Path
import sys


TEST_CI_DIR = Path(__file__).resolve().parent / "ci"
if str(TEST_CI_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_CI_DIR))
