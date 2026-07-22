from __future__ import annotations

import hashlib
import json
from typing import Any


def runtime_cache_entry_key_sha256(key: Any) -> str:
    try:
        payload = json.dumps(key, sort_keys=True, ensure_ascii=False, default=str)
    except TypeError:
        payload = repr(key)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
