from __future__ import annotations

from turnkey._internal.data import (
    binary_f1,
    binary_f1_or_none,
    finite_number,
    non_negative_int,
    safe_div_if_positive,
    safe_div_or_none,
    sha256_hex,
    sub_or_none,
)


def test_optional_numeric_helpers_return_none_for_undefined_values() -> None:
    assert safe_div_or_none(3, 0) is None
    assert safe_div_or_none(3, 2) == 1.5
    assert safe_div_if_positive(3, 0) is None
    assert safe_div_if_positive(3, -1) is None
    assert safe_div_if_positive(3, 2) == 1.5
    assert sub_or_none(3, "missing") is None
    assert sub_or_none(3, 2) == 1.0
    assert binary_f1_or_none(tp=0, fp=0, fn=0) is None
    assert binary_f1_or_none(tp=1, fp=1, fn=0) == 2 / 3
    assert binary_f1(tp=0, fp=0, fn=0) == 0.0
    assert binary_f1(tp=1, fp=1, fn=0) == 2 / 3
    assert finite_number(1.0)
    assert not finite_number(True)
    assert non_negative_int(0) == 0
    assert non_negative_int(3) == 3
    assert non_negative_int(-1) is None
    assert non_negative_int(True) is None
    assert non_negative_int(1.5) is None


def test_sha256_hex_hashes_utf8_text() -> None:
    assert sha256_hex("OK_MM") == "a708bee37d513888cefbd5f7f0f6052d611d05a618a1fe9931e83e3015054460"
    assert sha256_hex("安全") == "afb63a620bdcff15d8cd88695b46731cc2206e8c25b5b53793c6f2c0080d6ed5"
