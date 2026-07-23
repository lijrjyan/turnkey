from __future__ import annotations

from turnkey._internal.data import sha256_hex


def redact_text(text: str | None, *, keep_prefix: int = 0) -> str | None:
    if text is None:
        return None
    if keep_prefix <= 0:
        return f"<redacted sha256={sha256_hex(text)} chars={len(text)}>"
    prefix = text[:keep_prefix]
    return f"{prefix}<redacted sha256={sha256_hex(text)} chars={len(text)}>"
