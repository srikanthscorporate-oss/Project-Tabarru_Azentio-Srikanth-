"""Synthetic SPECIMEN certificate wording and schedules, written out by the generator.

Only claims.generate imports this module. The runtime reads the generated
files in data/certificates/ and data/schedules/, never these constants.
Amounts in the wording must match SCHEDULES; tests check that they do.
"""

from __future__ import annotations

SCHEDULES: dict[str, dict] = {
    "MEDICAL": {
        "product": "MEDICAL",
        "room_board_per_day": "250.00",
        "annual_limit": "100000.00",
        "waiting_period_days": 30,
    },
    "MOTOR": {
        "product": "MOTOR",
        "excess": "500.00",
    },
}

_MEDICAL_CLAUSES = [
    ("MED-1.1", "Who is covered. This certificate covers the participant named in the schedule while their contribution is paid."),
    ("MED-1.2", "Period of cover. We only pay for a hospital admission caused by an illness or injury that occurs between the start date and the end date of the certificate. An incident before the start date or after the end date is not covered."),
    ("MED-2.1", "Hospitalisation benefit. We pay the reasonable and customary hospital charges for an admission that is medically necessary, including room and board, medication, procedures and investigations."),
    ("MED-2.2", "Day care. Day care surgery that does not need an overnight stay is paid under the hospitalisation benefit."),
    ("MED-3.1", "Waiting period for illness. An illness that first shows itself within the waiting period of 30 days from the start date is not covered. The waiting period does not apply to injury from an accident."),
    ("MED-3.2", "Pre-existing conditions. A condition diagnosed or treated in the 12 months before the start date is not covered during the first year."),
    ("MED-4.1", "Room and board sub-limit. Room and board is paid up to RM 250.00 per day of the stay. Any room charge above RM 250.00 per day is borne by the participant."),
    ("MED-4.2", "Annual limit. The total paid in one certificate year will not exceed the annual limit of RM 100000.00 or the sum covered in the register, whichever is lower."),
    ("MED-4.3", "Days of stay. The number of days of stay is the number of nights between the admission date and the discharge date, and at least one day."),
    ("MED-5.1", "Cosmetic treatment exclusion. We do not pay for cosmetic or aesthetic treatment, including surgery to change appearance, unless it repairs damage from an accident."),
    ("MED-5.2", "Self-inflicted injury exclusion. We do not pay for an intentional self-inflicted injury."),
    ("MED-5.3", "Experimental treatment exclusion. We do not pay for treatment that is experimental or not generally accepted medical practice."),
    ("MED-6.1", "Documents needed. A claim must include the claim form, the itemised hospital bill and the discharge summary."),
    ("MED-6.2", "No excess. No excess or co-payment applies to the hospitalisation benefit."),
]  # fmt: skip

_MOTOR_CLAUSES = [
    ("MTR-1.1", "Insured vehicle. This certificate covers the vehicle named in the schedule while it is driven by an authorised driver."),
    ("MTR-1.2", "Period of cover. We only pay for loss or damage that occurs between the start date and the end date of the certificate. Loss before the start date or after the end date is not covered."),
    ("MTR-2.1", "Own damage. We pay the reasonable cost of repairing the insured vehicle after accidental collision, fire, theft, flood or a falling tree."),
    ("MTR-2.2", "Repair at a panel workshop. Repairs are paid at the estimate of the workshop, after any excess."),
    ("MTR-3.1", "Third party liability. We pay amounts the participant is legally liable for because of death, injury or damage to the property of others."),
    ("MTR-3.2", "Windscreen. Damage to the windscreen alone is paid without affecting the no claim discount."),
    ("MTR-4.1", "Racing exclusion. We do not pay for loss or damage while the vehicle is used for racing, speed testing or rallies."),
    ("MTR-4.2", "Unlicensed driver exclusion. We do not pay for loss or damage while the vehicle is driven by an unlicensed driver or a person not holding a valid licence."),
    ("MTR-4.3", "Wear and tear exclusion. We do not pay for wear and tear, mechanical breakdown or depreciation."),
    ("MTR-5.1", "Collision excess. An excess of RM 500.00 is deducted from each claim for own damage caused by a collision or impact. The participant bears this amount."),
    ("MTR-5.2", "No excess for special perils. No excess applies to damage caused only by flood or a falling tree."),
    ("MTR-5.3", "Limit of cover. We will not pay more than the sum covered in the register for one claim."),
    ("MTR-6.1", "Documents needed. A claim must include the claim form, the repair estimate and the police report."),
]  # fmt: skip


def _render(title: str, clauses: list[tuple[str, str]]) -> str:
    lines = [
        f"# {title} (SPECIMEN)",
        "",
        "SPECIMEN wording for testing only. Not a real certificate.",
        "",
    ]
    lines += [f"{clause_id} {text}" for clause_id, text in clauses]
    return "\n".join(lines) + "\n"


CERTIFICATES: dict[str, str] = {
    "MEDICAL": _render("Medical Takaful certificate wording", _MEDICAL_CLAUSES),
    "MOTOR": _render("Motor Takaful certificate wording", _MOTOR_CLAUSES),
}
