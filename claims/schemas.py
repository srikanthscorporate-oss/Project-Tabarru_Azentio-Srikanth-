"""Typed schemas: provider outputs, the assembled claim, and the API contract."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import Enum
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, field_validator

from claims.normalize import contains_ic, to_money


class DocType(str, Enum):
    CLAIM_FORM = "claim_form"
    HOSPITAL_BILL = "hospital_bill"
    DISCHARGE_SUMMARY = "discharge_summary"
    REPAIR_ESTIMATE = "repair_estimate"
    POLICE_REPORT = "police_report"


class Product(str, Enum):
    MOTOR = "MOTOR"
    MEDICAL = "MEDICAL"


Money = Annotated[Decimal, Field(ge=0, max_digits=12), AfterValidator(to_money)]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---- Provider outputs. Every key is required (missing key = schema error,
# retried); a value may be null when the document does not show it.


class ClassifyOutput(_Strict):
    doc_type: DocType


class ClaimFormFields(_Strict):
    certificate_no: str | None
    product: Product | None
    participant_name: str | None
    participant_ic: str | None
    incident_date: date | None
    amount_claimed: Money | None
    hospital_name: str | None
    vehicle_reg_no: str | None
    cause_of_loss: str | None
    police_report_no: str | None


class HospitalBillFields(_Strict):
    participant_name: str | None
    hospital_name: str | None
    admission_date: date | None
    discharge_date: date | None
    room_rate_per_day: Money | None
    bill_total: Money | None


class DischargeSummaryFields(_Strict):
    participant_name: str | None
    hospital_name: str | None
    admission_date: date | None
    discharge_date: date | None
    diagnosis: str | None


class RepairEstimateFields(_Strict):
    vehicle_reg_no: str | None
    repair_total: Money | None


class PoliceReportFields(_Strict):
    police_report_no: str | None
    vehicle_reg_no: str | None
    incident_date: date | None
    cause_of_loss: str | None


EXTRACTION_SCHEMAS: dict[DocType, type[_Strict]] = {
    DocType.CLAIM_FORM: ClaimFormFields,
    DocType.HOSPITAL_BILL: HospitalBillFields,
    DocType.DISCHARGE_SUMMARY: DischargeSummaryFields,
    DocType.REPAIR_ESTIMATE: RepairEstimateFields,
    DocType.POLICE_REPORT: PoliceReportFields,
}


# ---- Assembled claim


class ClaimFields(BaseModel):
    """The claim record. participant_ic only ever holds the masked value."""

    certificate_no: str | None = None
    product: Product | None = None
    participant_name: str | None = None
    participant_ic: str | None = None
    incident_date: date | None = None
    amount_claimed: Money | None = None
    admission_date: date | None = None
    discharge_date: date | None = None
    diagnosis: str | None = None
    hospital_name: str | None = None
    room_rate_per_day: Money | None = None
    bill_total: Money | None = None
    vehicle_reg_no: str | None = None
    cause_of_loss: str | None = None
    repair_total: Money | None = None
    police_report_no: str | None = None

    @field_validator("participant_ic")
    @classmethod
    def _must_be_masked(cls, value: str | None) -> str | None:
        if value is not None and not value.startswith("XXXXXX-XX-"):
            raise ValueError("participant_ic must be masked before it enters a claim record")
        return value


ClaimStatus = Literal["intake_complete", "flagged", "failed"]
DocStatus = Literal["processed", "failed"]
StepStatus = Literal["ok", "partial", "failed", "skipped"]


class TraceEntry(BaseModel):
    step: str
    doc_ids: list[str]
    status: StepStatus
    duration_ms: float
    provider_calls: int
    error: str | None = None


# ---- API contract


def _not_ic(value: str) -> str:
    # Ids are echoed in responses, the trace and logs, so they must not carry an IC.
    if contains_ic(value):
        raise ValueError("id must not contain an IC number")
    return value


Identifier = Annotated[
    str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$"), AfterValidator(_not_ic)
]


class InputDocument(_Strict):
    doc_id: Identifier
    text: str = Field(min_length=1, max_length=100_000)


class ClaimRequest(_Strict):
    claim_id: Identifier
    documents: list[InputDocument] = Field(min_length=1, max_length=10)

    @field_validator("documents")
    @classmethod
    def _unique_doc_ids(cls, docs: list[InputDocument]) -> list[InputDocument]:
        if len({d.doc_id for d in docs}) != len(docs):
            raise ValueError("doc_id values must be unique within a claim")
        return docs


class DocumentResult(BaseModel):
    doc_id: str
    doc_type: DocType | None
    status: DocStatus


class ClaimResponse(BaseModel):
    claim_id: str
    status: ClaimStatus
    documents: list[DocumentResult]
    fields: ClaimFields
    reason_codes: list[str]


class ClaimDetail(ClaimResponse):
    trace: list[TraceEntry]
