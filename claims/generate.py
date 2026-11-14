"""Synthetic data generator.

    python -m claims.generate --claims 40 --seed 7 --out data/

Writes the certificate register, one folder of OCR text documents per claim and
a separate labels file. Output depends only on the arguments: one seeded
random.Random drives every choice, nothing reads the clock, and files are
written in a fixed order, so the same seed gives byte-identical files.

All people, IC numbers, certificates, hospitals and workshops are invented and
every document carries the word SPECIMEN. This module is the only place that
knows the answers; the runtime pipeline never imports it.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

FIRST_NAMES = ["Aminah", "Farid", "Mei Ling", "Rajesh", "Siti", "Daniel", "Kavitha", "Hafiz"]
SURNAMES = ["Contoh", "Specimen", "Sampel", "Ujian"]
HOSPITALS = [
    "Hospital Contoh Seri Damai",
    "Hospital Specimen Bukit Indah",
    "Pusat Perubatan Contoh Ampang",
]
DIAGNOSES = [
    "Dengue fever",
    "Acute appendicitis",
    "Community acquired pneumonia",
    "Gastroenteritis",
]
CAUSES = [
    "collision with another vehicle at a junction",
    "hit a stationary object while reversing",
    "flood damage during heavy rain",
    "fallen tree branch on the roof",
]
ACCIDENT_DIAGNOSIS = "Fractured left radius after road accident"
WORKSHOPS = ["Bengkel Contoh Jaya", "Specimen Auto Works"]
PLATE_LETTERS = "ABCDEFGHJKLMNPQRSTUVWXY"  # Malaysian plates skip I and O
# Share of documents that get OCR noise. Chosen by a seeded sample rather than a
# per-document coin flip, so every run meets the 25% minimum, even a small one.
NOISE_SHARE = 0.4

# Scenario per claim index, cycling every ten claims: 6 clean, 1 of each anomaly.
SCENARIO_CYCLE = [
    "clean",
    "clean",
    "cert_not_found",
    "clean",
    "missing_document",
    "clean",
    "date_order_invalid",
    "clean",
    "amount_mismatch",
    "clean",
]
REQUIRED_DOCS = {
    "MEDICAL": ["claim_form", "hospital_bill", "discharge_summary"],
    "MOTOR": ["claim_form", "repair_estimate", "police_report"],
}
SCENARIO_CODE = {
    "cert_not_found": "CERT_NOT_FOUND",
    "missing_document": "MISSING_DOCUMENT",
    "date_order_invalid": "DATE_ORDER_INVALID",
    "amount_mismatch": "AMOUNT_MISMATCH",
}
COMMON_FIELDS = [
    "certificate_no", "product", "participant_name", "participant_ic", "incident_date",
    "amount_claimed",
]  # fmt: skip
REQUIRED_FIELDS = {
    "MEDICAL": COMMON_FIELDS + [
        "admission_date", "discharge_date", "diagnosis", "hospital_name", "room_rate_per_day",
        "bill_total",
    ],
    "MOTOR": COMMON_FIELDS + ["vehicle_reg_no", "cause_of_loss", "repair_total", "police_report_no"],
}  # fmt: skip
# Ground-truth fields each document type shows, used to build labels.
FIELDS_SHOWN = {
    "claim_form": {
        "certificate_no", "product", "participant_name", "participant_ic", "incident_date",
        "amount_claimed", "hospital_name", "vehicle_reg_no", "cause_of_loss", "police_report_no",
    },
    "hospital_bill": {
        "participant_name", "hospital_name", "admission_date", "discharge_date",
        "room_rate_per_day", "bill_total",
    },
    "discharge_summary": {
        "participant_name", "hospital_name", "admission_date", "discharge_date", "diagnosis",
    },
    "repair_estimate": {"vehicle_reg_no", "repair_total"},
    "police_report": {"police_report_no", "vehicle_reg_no", "incident_date", "cause_of_loss"},
}  # fmt: skip


@dataclass
class Claim:
    claim_id: str
    product: str
    scenario: str
    facts: dict
    doc_types: list[str] = field(default_factory=list)


class Writer:
    """Formats values for one document, applying OCR noise when enabled."""

    def __init__(self, rng: random.Random, noisy: bool):
        self.rng = rng
        self.noisy = noisy

    def ocr(self, text: str) -> str:
        """In a noisy document, swap one 0 to O or 1 to I in a numeric value."""
        positions = [i for i, ch in enumerate(text) if ch in "01"]
        if not self.noisy or not positions or self.rng.random() < 0.4:
            return text
        i = self.rng.choice(positions)
        return text[:i] + {"0": "O", "1": "I"}[text[i]] + text[i + 1 :]

    def date(self, d: date) -> str:
        style = self.rng.randrange(4)
        text = [
            d.strftime("%d/%m/%Y"),
            f"{d.day}-{d.strftime('%b')}-{d.year}",
            d.isoformat(),
            f"{d.day} {d.strftime('%B')} {d.year}",
        ][style]
        return self.ocr(text)

    def money(self, amount: Decimal) -> str:
        style = self.rng.randrange(3)
        if style == 0:
            text = f"RM {amount:,.2f}"
        elif style == 1:
            text = f"{amount:.2f}" if amount != amount.to_integral() else f"{int(amount)}"
        else:
            text = f"MYR {amount:,.2f}"
        return self.ocr(text)


def money(value: float | int) -> Decimal:
    return Decimal(str(value)).quantize(Decimal("0.01"))


def make_facts(rng: random.Random, index: int, product: str, scenario: str) -> dict:
    name = f"{rng.choice(FIRST_NAMES)} {rng.choice(SURNAMES)}"
    birth = date(1960, 1, 1) + timedelta(days=rng.randrange(14000))
    ic = f"{birth:%y%m%d}-00-{rng.randrange(10000):04d}"  # place code 00 is not issued
    incident = date(2026, 1, 5) + timedelta(days=rng.randrange(280))
    start = incident - timedelta(days=rng.randrange(30, 300))
    prefix = "MED" if product == "MEDICAL" else "MTR"
    facts: dict = {
        "certificate_no": f"{prefix}-2026-{100000 + index * 37 + rng.randrange(30):06d}",
        "product": product,
        "participant_name": name,
        "participant_ic": ic,
        "incident_date": incident,
        "start_date": start,
        "end_date": start + timedelta(days=364),
    }
    if product == "MEDICAL":
        admission = incident + timedelta(days=rng.randrange(0, 2))
        discharge = admission + timedelta(days=rng.randrange(1, 6))
        if scenario == "date_order_invalid":
            if index % 2:
                discharge = admission - timedelta(days=rng.randrange(1, 3))
            else:
                admission = incident - timedelta(days=rng.randrange(1, 4))
        nights = max((discharge - admission).days, 1)
        rate = money(rng.choice([150, 200, 250, 300, 400]))
        other = [money(rng.randrange(80, 900)), money(rng.randrange(100, 2500))]
        total = rate * nights + sum(other)
        facts |= {
            "admission_date": admission,
            "discharge_date": discharge,
            "diagnosis": rng.choice(DIAGNOSES),
            "hospital_name": rng.choice(HOSPITALS),
            "room_rate_per_day": rate,
            "nights": nights,
            "other_charges": other,
            "bill_total": total,
            "sum_covered": money(rng.choice([50000, 100000, 150000])),
        }
        facts["claimed_base"] = total
    else:
        parts = money(rng.randrange(400, 6000)) + money(rng.randrange(0, 99)) / 100
        labour = money(rng.randrange(150, 1500))
        facts |= {
            "vehicle_reg_no": f"{''.join(rng.choice(PLATE_LETTERS) for _ in range(3))} "
            f"{rng.randrange(1, 9999)}",
            "cause_of_loss": rng.choice(CAUSES),
            "police_report_no": f"PR/2026/{rng.randrange(1, 999999):06d}",
            "parts": parts.quantize(Decimal("0.01")),
            "labour": labour,
            "repair_total": (parts + labour).quantize(Decimal("0.01")),
            "workshop": rng.choice(WORKSHOPS),
            "sum_covered": money(rng.choice([30000, 60000, 90000])),
        }
        facts["claimed_base"] = facts["repair_total"]
    claimed = facts["claimed_base"]
    if scenario == "amount_mismatch":
        claimed = (claimed * Decimal(str(1 + rng.randrange(5, 25) / 100))).quantize(Decimal("0.01"))
    facts["amount_claimed"] = claimed
    return facts


# ---- Document templates: two layouts per type, label and value on one line
# so that shuffled lines keep each pair intact.


def claim_form(f: dict, w: Writer, layout: int) -> list[str]:
    medical = f["product"] == "MEDICAL"
    if layout == 0:
        lines = [
            "SPECIMEN - TAKAFUL CLAIM FORM",
            f"Certificate No: {w.ocr(f['certificate_no'])}",
            f"Product: {'Medical' if medical else 'Motor'} Takaful",
            f"Participant Name: {f['participant_name']}",
            f"IC No: {w.ocr(f['participant_ic'])}",
            f"Date of Incident: {w.date(f['incident_date'])}",
            f"Amount Claimed: {w.money(f['amount_claimed'])}",
        ]
        if medical:
            lines.append(f"Hospital: {f['hospital_name']}")
        else:
            lines += [
                f"Vehicle Reg No: {w.ocr(f['vehicle_reg_no'])}",
                f"Cause of Loss: {f['cause_of_loss']}",
                f"Police Report No: {w.ocr(f['police_report_no'])}",
            ]
        return lines + ["Declaration: I confirm the above is true and complete. SPECIMEN ONLY"]
    lines = [
        "NOTIFICATION OF CLAIM (SPECIMEN DOCUMENT)",
        "Section A - Participant",
        f"  Name of participant ..... {f['participant_name']}",
        f"  NRIC ..... {w.ocr(f['participant_ic'])}",
        f"  Takaful certificate ..... {w.ocr(f['certificate_no'])}",
        f"  Plan ..... {f['product']}",
        "Section B - Loss details",
        f"  Loss date ..... {w.date(f['incident_date'])}",
        f"  Total sum claimed ..... {w.money(f['amount_claimed'])}",
    ]
    if medical:
        lines.append(f"  Hospital admitted to ..... {f['hospital_name']}")
    else:
        lines += [
            f"  Vehicle registration ..... {w.ocr(f['vehicle_reg_no'])}",
            f"  Description of loss ..... {f['cause_of_loss']}",
            f"  Police report reference ..... {w.ocr(f['police_report_no'])}",
        ]
    return lines + ["Claimant declaration signed. Form CLM-B rev 3."]


def hospital_bill(f: dict, w: Writer, layout: int) -> list[str]:
    room = f["room_rate_per_day"] * f["nights"]
    pharmacy, procedures = f["other_charges"]
    if layout == 0:
        return [
            "HOSPITAL BILL - SPECIMEN",
            f["hospital_name"],
            f"Invoice No: INV-{w.ocr(str(abs(hash_text(f['certificate_no'])) % 90000 + 10000))}",
            f"Patient: {f['participant_name']}",
            f"Admission Date: {w.date(f['admission_date'])}",
            f"Discharge Date: {w.date(f['discharge_date'])}",
            f"Room rate per day: {w.money(f['room_rate_per_day'])}",
            f"Room and board ({f['nights']} days): {w.money(room)}",
            f"Pharmacy: {w.money(pharmacy)}",
            f"Procedures and investigations: {w.money(procedures)}",
            f"TOTAL AMOUNT DUE: {w.money(f['bill_total'])}",
        ]
    return [
        "SPECIMEN STATEMENT OF ACCOUNT",
        f"Facility: {f['hospital_name']}",
        f"Patient name: {f['participant_name']}",
        f"Admitted: {w.date(f['admission_date'])}   Discharged: {w.date(f['discharge_date'])}",
        "Item | Amount",
        f"Room charges per day | {w.money(f['room_rate_per_day'])}",
        f"Room charges x {f['nights']} | {w.money(room)}",
        f"Medication | {w.money(pharmacy)}",
        f"Procedures | {w.money(procedures)}",
        f"Grand total | {w.money(f['bill_total'])}",
    ]


def discharge_summary(f: dict, w: Writer, layout: int) -> list[str]:
    if layout == 0:
        return [
            "DISCHARGE SUMMARY (SPECIMEN)",
            f"Hospital: {f['hospital_name']}",
            f"Patient: {f['participant_name']}",
            f"Date of admission: {w.date(f['admission_date'])}",
            f"Date of discharge: {w.date(f['discharge_date'])}",
            f"Final diagnosis: {f['diagnosis']}",
            "Attending physician: Dr Specimen Rahim",
        ]
    return [
        "SPECIMEN - CLINICAL SUMMARY ON DISCHARGE",
        f["hospital_name"],
        f"{f['participant_name']} was admitted on {w.date(f['admission_date'])} "
        f"and discharged on {w.date(f['discharge_date'])}.",
        f"Diagnosis at discharge: {f['diagnosis']}.",
        "Attending physician: Dr Contoh Lim. Follow up in two weeks.",
    ]


def repair_estimate(f: dict, w: Writer, layout: int) -> list[str]:
    if layout == 0:
        return [
            "REPAIR ESTIMATE - SPECIMEN",
            f"Workshop: {f['workshop']} Sdn Bhd",
            f"Vehicle Reg No: {w.ocr(f['vehicle_reg_no'])}",
            f"Parts: {w.money(f['parts'])}",
            f"Labour: {w.money(f['labour'])}",
            f"Estimated repair total: {w.money(f['repair_total'])}",
        ]
    return [
        "SPECIMEN QUOTATION FOR REPAIRS",
        f"Panel workshop: {f['workshop']}",
        f"Vehicle: {w.ocr(f['vehicle_reg_no'])}",
        "Description | Cost",
        f"Replacement parts | {w.money(f['parts'])}",
        f"Labour charges | {w.money(f['labour'])}",
        f"Total quotation | {w.money(f['repair_total'])}",
    ]


def police_report(f: dict, w: Writer, layout: int) -> list[str]:
    if layout == 0:
        return [
            "POLICE REPORT (SPECIMEN)",
            f"Report No: {w.ocr(f['police_report_no'])}",
            "Police station: Balai Polis Contoh Sentral",
            f"Date of incident: {w.date(f['incident_date'])}",
            f"Vehicle No: {w.ocr(f['vehicle_reg_no'])}",
            f"Statement: {f['cause_of_loss']}",
            "Investigating officer: Insp. Specimen Wong",
        ]
    return [
        "SPECIMEN - LAPORAN POLIS",
        f"Report reference: {w.ocr(f['police_report_no'])}",
        "Station: Balai Polis Specimen Utara",
        f"Incident date: {w.date(f['incident_date'])}",
        f"Vehicle involved: {w.ocr(f['vehicle_reg_no'])}",
        f"Nature of incident: {f['cause_of_loss']}",
        "Investigating officer: Sjn. Contoh Ali",
    ]


TEMPLATES = {
    "claim_form": claim_form,
    "hospital_bill": hospital_bill,
    "discharge_summary": discharge_summary,
    "repair_estimate": repair_estimate,
    "police_report": police_report,
}

NOISE_HEADERS = [
    "*** SCANNED COPY *** SPECIMEN *** page 1 of 1",
    "Fax recv 14/11 ref {ref} -- SPECIMEN",
    "Received by e-mail intake, batch {ref}",
]
NOISE_FOOTERS = ["-- end of scan --", "SPECIMEN copy for testing. Not a real document.", "pg 1/1"]


def hash_text(text: str) -> int:
    """Stable small hash (Python's hash() is salted per process)."""
    value = 0
    for ch in text:
        value = (value * 131 + ord(ch)) % 1_000_003
    return value


def render(rng: random.Random, claim: Claim, doc_type: str, noisy: bool) -> str:
    w = Writer(rng, noisy)
    lines = TEMPLATES[doc_type](claim.facts, w, rng.randrange(2))
    if noisy:
        ref = f"{rng.randrange(16**6):06x}"
        lines = [rng.choice(NOISE_HEADERS).format(ref=ref), *lines, rng.choice(NOISE_FOOTERS)]
        if rng.random() < 0.5:
            rng.shuffle(lines)
    return "\n".join(lines) + "\n"


def build_claims(rng: random.Random, count: int) -> list[Claim]:
    claims = []
    for i in range(count):
        scenario = SCENARIO_CYCLE[i % len(SCENARIO_CYCLE)]
        product = "MEDICAL" if (i + i // len(SCENARIO_CYCLE)) % 2 == 0 else "MOTOR"
        if scenario == "date_order_invalid":
            product = "MEDICAL"  # the date-order rule is defined on admission/discharge
        facts = make_facts(rng, i, product, scenario)
        doc_types = list(REQUIRED_DOCS[product])
        if scenario == "missing_document":
            doc_types.remove(rng.choice(doc_types[1:]))
        elif product == "MEDICAL" and rng.random() < 0.25:
            # Road-accident hospitalisation: packet also carries a police report.
            doc_types.append("police_report")
            facts["diagnosis"] = ACCIDENT_DIAGNOSIS
            facts["cause_of_loss"] = "road accident, rider thrown from motorcycle"
            facts["police_report_no"] = f"PR/2026/{rng.randrange(1, 999999):06d}"
            facts["vehicle_reg_no"] = f"{''.join(rng.choice(PLATE_LETTERS) for _ in range(3))} 77"
        rng.shuffle(doc_types)
        claims.append(Claim(f"CLM-{i + 1:04d}", product, scenario, facts, doc_types))
    return claims


def label_value(value):
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Decimal):
        return f"{value:.2f}"
    return value


def write_dataset(count: int, seed: int, out: Path) -> None:
    rng = random.Random(seed)
    claims = build_claims(rng, count)
    claims_dir = out / "claims"
    if claims_dir.exists():
        shutil.rmtree(claims_dir)  # stale folders from an earlier run would break reproducibility
    claims_dir.mkdir(parents=True)

    total_docs = sum(len(c.doc_types) for c in claims)
    noisy_positions = set(rng.sample(range(total_docs), math.ceil(total_docs * NOISE_SHARE)))

    labels, position = [], 0
    for claim in claims:
        folder = claims_dir / claim.claim_id
        folder.mkdir()
        docs = []
        for doc_type in claim.doc_types:
            doc_id = f"{rng.randrange(16**8):08x}"
            noisy = position in noisy_positions
            position += 1
            text = render(rng, claim, doc_type, noisy)
            (folder / f"{doc_id}.txt").write_text(text, encoding="utf-8", newline="\n")
            docs.append({"doc_id": doc_id, "doc_type": doc_type, "ocr_noise": noisy})
        shown = set().union(*(FIELDS_SHOWN[d] for d in claim.doc_types))
        fields = {k: label_value(claim.facts.get(k)) for k in sorted(shown)}
        if claim.product == "MEDICAL":
            for motor_only in ("vehicle_reg_no", "cause_of_loss", "police_report_no"):
                if "police_report" not in claim.doc_types:
                    fields.pop(motor_only, None)
        else:
            fields.pop("hospital_name", None)
        labels.append(
            {
                "claim_id": claim.claim_id,
                "product": claim.product,
                "scenario": claim.scenario,
                "documents": docs,
                "fields": fields,
                "expected_reason_codes": expected_codes(claim, fields),
            }
        )

    policies = [
        {
            "certificate_no": c.facts["certificate_no"],
            "participant_name": c.facts["participant_name"],
            "participant_ic": c.facts["participant_ic"],
            "product": c.product,
            "start_date": c.facts["start_date"].isoformat(),
            "end_date": c.facts["end_date"].isoformat(),
            "sum_covered": f"{c.facts['sum_covered']:.2f}",
        }
        for c in claims
        if c.scenario != "cert_not_found"
    ]
    write_jsonl(out / "policies.jsonl", sorted(policies, key=lambda p: p["certificate_no"]))
    write_jsonl(out / "labels.jsonl", labels)
    print(
        f"wrote {count} claims, {total_docs} documents ({len(noisy_positions)} with OCR noise), "
        f"{len(policies)} certificates to {out}"
    )


def expected_codes(claim: Claim, fields: dict) -> list[str]:
    codes = [SCENARIO_CODE[claim.scenario]] if claim.scenario in SCENARIO_CODE else []
    if any(fields.get(k) is None for k in REQUIRED_FIELDS[claim.product]):
        codes.append("FIELDS_INCOMPLETE")
    return codes


def write_jsonl(path: Path, rows: list[dict]) -> None:
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8", newline="\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic SPECIMEN claim packets.")
    parser.add_argument("--claims", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=Path("data"))
    args = parser.parse_args(argv)
    if args.claims < 1:
        parser.error("--claims must be at least 1")
    args.out.mkdir(parents=True, exist_ok=True)
    write_dataset(args.claims, args.seed, args.out)


if __name__ == "__main__":
    main()
