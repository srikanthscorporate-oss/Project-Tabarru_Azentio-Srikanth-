"""Certificate register lookup, field precedence and business rules."""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from claims.schemas import DocType, Product

REQUIRED_DOCS: dict[Product, set[DocType]] = {
    Product.MEDICAL: {DocType.CLAIM_FORM, DocType.HOSPITAL_BILL, DocType.DISCHARGE_SUMMARY},
    Product.MOTOR: {DocType.CLAIM_FORM, DocType.REPAIR_ESTIMATE, DocType.POLICE_REPORT},
}

COMMON_FIELDS = [
    "certificate_no", "product", "participant_name", "participant_ic", "incident_date",
    "amount_claimed",
]  # fmt: skip
REQUIRED_FIELDS: dict[Product, list[str]] = {
    Product.MEDICAL: COMMON_FIELDS + [
        "admission_date", "discharge_date", "diagnosis", "hospital_name", "room_rate_per_day",
        "bill_total",
    ],
    Product.MOTOR: COMMON_FIELDS + [
        "vehicle_reg_no", "cause_of_loss", "repair_total", "police_report_no",
    ],
}  # fmt: skip

# Field precedence: for each field, document types in the order they are
# trusted. The first processed document of the earliest type with a non-null
# value wins; ties within a type go to packet order. The claim form is the
# source for what the participant asserts; bills, summaries, estimates and
# police reports are the source for what they themselves certify.
_CF, _HB, _DS, _RE, _PR = (
    DocType.CLAIM_FORM,
    DocType.HOSPITAL_BILL,
    DocType.DISCHARGE_SUMMARY,
    DocType.REPAIR_ESTIMATE,
    DocType.POLICE_REPORT,
)
PRECEDENCE: dict[str, list[DocType]] = {
    "certificate_no": [_CF],
    "product": [_CF],
    "participant_name": [_CF, _HB, _DS],
    "participant_ic": [_CF],
    "incident_date": [_CF, _PR],
    "amount_claimed": [_CF],
    "admission_date": [_DS, _HB],
    "discharge_date": [_DS, _HB],
    "diagnosis": [_DS],
    "hospital_name": [_HB, _DS, _CF],
    "room_rate_per_day": [_HB],
    "bill_total": [_HB],
    "vehicle_reg_no": [_CF, _PR, _RE],
    "cause_of_loss": [_PR, _CF],
    "repair_total": [_RE],
    "police_report_no": [_PR, _CF],
}

AMOUNT_TOLERANCE = Decimal("0.01")  # 1 percent of the bill or estimate total


@dataclass(frozen=True)
class Extraction:
    doc_id: str
    doc_type: DocType
    values: dict


def load_register(path: Path) -> dict[str, dict]:
    """Certificate register keyed by certificate_no. Missing file is an error."""
    register = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                register[row["certificate_no"]] = row
    return register


def merge_fields(extractions: list[Extraction]) -> dict:
    merged = {}
    for field_name, order in PRECEDENCE.items():
        merged[field_name] = next(
            (
                ex.values[field_name]
                for doc_type in order
                for ex in extractions
                if ex.doc_type == doc_type and ex.values.get(field_name) is not None
            ),
            None,
        )
    return merged


def resolve_product(merged: dict, register: dict, present: set[DocType]) -> Product | None:
    """Claim form first, then the register entry, then the document mix."""
    if merged.get("product"):
        return Product(merged["product"])
    entry = register.get(merged.get("certificate_no") or "")
    if entry:
        return Product(entry["product"])
    if present & {DocType.HOSPITAL_BILL, DocType.DISCHARGE_SUMMARY}:
        return Product.MEDICAL
    if present & {DocType.REPAIR_ESTIMATE}:
        return Product.MOTOR
    return None


def amounts_differ(claimed: Decimal, total: Decimal) -> bool:
    """True when the claimed amount differs from the total by more than 1% of the total."""
    return abs(claimed - total) > total * AMOUNT_TOLERANCE


def validate(
    extractions: list[Extraction],
    classified: set[DocType],
    register: dict[str, dict],
) -> tuple[dict, list[str]]:
    """Return (merged fields, reason codes) for the processed documents.

    ``classified`` holds the types of every document that was classified, even
    if its extraction later failed: the document is present in the packet.
    """
    merged = merge_fields(extractions)
    product = resolve_product(merged, register, classified)
    merged["product"] = product.value if product else None
    codes: list[str] = []

    cert = merged.get("certificate_no")
    if not cert or cert not in register:
        codes.append("CERT_NOT_FOUND")

    if product is None or REQUIRED_DOCS[product] - classified:
        codes.append("MISSING_DOCUMENT")

    incident, admission, discharge = (
        merged.get("incident_date"),
        merged.get("admission_date"),
        merged.get("discharge_date"),
    )
    if (admission and discharge and discharge < admission) or (
        incident and admission and admission < incident
    ):
        codes.append("DATE_ORDER_INVALID")

    # Compare against every per-document total, not only the merged one, so a
    # conflict on any bill or estimate in the packet is caught.
    claimed = merged.get("amount_claimed")
    totals = [
        ex.values[key]
        for ex in extractions
        for key in ("bill_total", "repair_total")
        if ex.values.get(key) is not None
    ]
    if claimed is not None and any(amounts_differ(claimed, t) for t in totals):
        codes.append("AMOUNT_MISMATCH")

    if product is not None and any(merged.get(f) is None for f in REQUIRED_FIELDS[product]):
        codes.append("FIELDS_INCOMPLETE")
    return merged, codes
