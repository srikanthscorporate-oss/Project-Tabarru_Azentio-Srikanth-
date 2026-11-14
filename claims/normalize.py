"""Normalisation of OCR'd values: dates, MYR amounts, identifiers and IC masking.

Date convention: numeric dates with slashes or dashes are read day first
(DD/MM/YYYY), the Malaysian convention, so 03/04/2026 is 3 April 2026. A
four digit first part is read as ISO (YYYY-MM-DD). Dates that do not parse to a
real calendar date return None rather than a guess.
"""

from __future__ import annotations

import re
from datetime import date
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

# OCR commonly reads 0 as O and 1 as I or l inside digit runs.
_OCR_DIGITS = str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1"})

_MONTHS = {
    m: i
    for i, m in enumerate(
        ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
        start=1,
    )
}

CENT = Decimal("0.01")


def fix_ocr_digits(token: str) -> str:
    """Repair O/I/l swaps in a token that is meant to be numeric."""
    return token.translate(_OCR_DIGITS)


def fix_ocr_identifier(value: str) -> str:
    """Repair digit swaps inside an identifier such as MED-2026-0I2345.

    Only segments that already contain a real digit are repaired, so letter
    prefixes like MED or PR are left alone.
    """
    parts = re.split(r"([-/ ])", value.strip().upper())
    return "".join(fix_ocr_digits(p) if re.search(r"\d", p) else p for p in parts)


def parse_date(raw: str | None) -> date | None:
    if not raw:
        return None
    text = raw.strip().rstrip(".,")
    numeric = re.fullmatch(r"([0-9OIl]{1,4})[/-]([0-9OIl]{1,2})[/-]([0-9OIl]{2,4})", text)
    try:
        if numeric:
            a, b, c = (int(fix_ocr_digits(p)) for p in numeric.groups())
            if len(numeric.group(1)) == 4:
                return date(a, b, c)
            year = c + 2000 if c < 100 else c
            return date(year, b, a)
        named = re.fullmatch(r"([0-9OIl]{1,2})[ -]([A-Za-z]{3,9})[ -]([0-9OIl]{4})", text)
        if named:
            month = _MONTHS.get(named.group(2)[:3].lower())
            if month is None:
                return None
            return date(
                int(fix_ocr_digits(named.group(3))), month, int(fix_ocr_digits(named.group(1)))
            )
    except ValueError:
        return None
    return None


def parse_amount(raw: str | None) -> Decimal | None:
    """Parse 'RM 1,250.00', 'MYR 1250', '1,25O.5' into Decimal('1250.00')."""
    if not raw:
        return None
    text = re.sub(r"(?i)\b(RM|MYR)\b", "", raw).replace(",", "").replace(" ", "")
    try:
        return to_money(Decimal(fix_ocr_digits(text)))
    except InvalidOperation:
        return None


def to_money(value: Decimal) -> Decimal:
    return value.quantize(CENT, rounding=ROUND_HALF_UP)


_IC_PATTERN = re.compile(
    r"(?<![0-9A-Za-z])[0-9OoIl]{6}[- ]?[0-9OoIl]{2}[- ]?([0-9OoIl]{4})(?![0-9A-Za-z])"
)


def mask_ic(ic: str | None) -> str | None:
    """Return XXXXXX-XX-1234 style masking, keeping only the last four digits."""
    if ic is None:
        return None
    digits = re.sub(r"\D", "", fix_ocr_digits(ic))
    return f"XXXXXX-XX-{digits[-4:]}" if len(digits) == 12 else "XXXXXX-XX-XXXX"


def redact_ics(text: str) -> str:
    """Mask every IC-shaped number inside free text (errors, stored documents)."""
    return _IC_PATTERN.sub(lambda m: f"XXXXXX-XX-{fix_ocr_digits(m.group(1))}", text)


def contains_ic(text: str) -> bool:
    return _IC_PATTERN.search(text) is not None
