from datetime import date
from decimal import Decimal

import pytest

from claims.normalize import fix_ocr_identifier, mask_ic, parse_amount, parse_date, redact_ics


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("03/04/2026", date(2026, 4, 3)),  # day first
        ("3-Apr-2026", date(2026, 4, 3)),
        ("3 April 2026", date(2026, 4, 3)),
        ("2026-04-03", date(2026, 4, 3)),
        ("O3/O4/2O26", date(2026, 4, 3)),  # OCR O for 0
        ("13/12/2026", date(2026, 12, 13)),
    ],
)
def test_parse_date(raw, expected):
    assert parse_date(raw) == expected


@pytest.mark.parametrize("raw", ["31/02/2026", "12/13/2026", "3-Foo-2026", "", None, "soon"])
def test_parse_date_rejects_impossible_dates(raw):
    assert parse_date(raw) is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("RM 1,250.00", Decimal("1250.00")),
        ("1250", Decimal("1250.00")),
        ("MYR 1,25O.5", Decimal("1250.50")),
        ("I,000.005", Decimal("1000.01")),
    ],
)
def test_parse_amount(raw, expected):
    value = parse_amount(raw)
    assert value == expected
    assert value.as_tuple().exponent == -2


def test_parse_amount_rejects_garbage():
    assert parse_amount("RM abc") is None


def test_identifier_repair_keeps_letter_prefix():
    assert fix_ocr_identifier("MED-2O26-0I2345") == "MED-2026-012345"
    assert fix_ocr_identifier("PR/2026/6O9852") == "PR/2026/609852"


def test_mask_and_redact_ic():
    assert mask_ic("900101-00-1234") == "XXXXXX-XX-1234"
    assert mask_ic("bad") == "XXXXXX-XX-XXXX"
    text = "IC No: 9OO1O1-00-I234 and 900101001234"
    redacted = redact_ics(text)
    assert "XXXXXX-XX-1234" in redacted
    assert "900101" not in redacted and "9OO1O1" not in redacted
