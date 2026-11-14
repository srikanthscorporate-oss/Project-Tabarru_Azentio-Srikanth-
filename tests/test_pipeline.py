from datetime import date
from decimal import Decimal

import pytest

from claims.normalize import mask_ic
from claims.pipeline import run_intake
from claims.providers import FakeProvider
from claims.schemas import InputDocument
from claims.validation import Extraction, amounts_differ, validate
from tests.conftest import first_label, load_packet

STEPS = ["classify_documents", "extract_fields", "validate", "assemble_claim"]


class RecordingProvider(FakeProvider):
    def __init__(self, faults=None):
        super().__init__(faults)
        self.prompts = []

    def complete(self, prompt):
        self.prompts.append(prompt)
        return super().complete(prompt)


def doc_types(label):
    return {d["doc_id"]: d["doc_type"] for d in label["documents"]}


@pytest.mark.parametrize("product", ["MEDICAL", "MOTOR"])
def test_complete_packet(dataset, labels, register, product):
    label = first_label(labels, "clean", product)
    claim = run_intake(label["claim_id"], load_packet(dataset, label["claim_id"]), FakeProvider(), register)

    assert claim.status == "intake_complete"
    assert claim.reason_codes == []
    assert {d.doc_id: d.doc_type.value for d in claim.documents} == doc_types(label)
    assert all(d.status == "processed" for d in claim.documents)
    expected = dict(label["fields"], participant_ic=mask_ic(label["fields"]["participant_ic"]))
    got = claim.fields.model_dump(mode="json")
    assert {k: got[k] for k in expected} == expected
    assert [t.step for t in claim.trace] == STEPS


def test_every_generated_claim_matches_its_labels(dataset, labels, register):
    """Whole dataset, including noisy and shuffled documents."""
    for label in labels:
        claim = run_intake(
            label["claim_id"], load_packet(dataset, label["claim_id"]), FakeProvider(), register
        )
        assert {d.doc_id: d.doc_type.value for d in claim.documents} == doc_types(label)
        assert sorted(claim.reason_codes) == sorted(label["expected_reason_codes"]), label["claim_id"]


@pytest.mark.parametrize(
    "scenario", ["cert_not_found", "missing_document", "date_order_invalid", "amount_mismatch"]
)
def test_reason_code_scenarios_are_flagged(dataset, labels, register, scenario):
    label = first_label(labels, scenario)
    claim = run_intake(label["claim_id"], load_packet(dataset, label["claim_id"]), FakeProvider(), register)
    code = {
        "cert_not_found": "CERT_NOT_FOUND",
        "missing_document": "MISSING_DOCUMENT",
        "date_order_invalid": "DATE_ORDER_INVALID",
        "amount_mismatch": "AMOUNT_MISMATCH",
    }[scenario]
    assert code in claim.reason_codes
    assert claim.status == "flagged"


def test_malformed_json_then_successful_retry(dataset, labels, register):
    label = first_label(labels, "clean", "MEDICAL")
    docs = load_packet(dataset, label["claim_id"])
    provider = RecordingProvider(faults={1: "malformed_json"})
    claim = run_intake(label["claim_id"], docs, provider, register)

    assert claim.status == "intake_complete"
    assert all(d.status == "processed" for d in claim.documents)
    classify = claim.trace[0]
    assert classify.provider_calls == len(docs) + 1
    assert classify.status == "ok"
    assert "not valid JSON" in provider.prompts[1]  # the retry carries the error
    assert "rejected" not in provider.prompts[0]


def test_malformed_twice_fails_document_and_flags_claim(dataset, labels, register):
    label = first_label(labels, "clean", "MOTOR")
    docs = load_packet(dataset, label["claim_id"])
    claim = run_intake(
        label["claim_id"], docs, FakeProvider(faults={1: "malformed_json", 2: "malformed_json"}), register
    )

    failed = [d for d in claim.documents if d.status == "failed"]
    assert [d.doc_id for d in failed] == [docs[0].doc_id]
    assert failed[0].doc_type is None
    assert claim.status == "flagged"
    assert claim.trace[0].status == "partial"
    assert docs[0].doc_id in claim.trace[0].error
    assert len(claim.trace) == 4


def test_missing_required_field_is_retried(dataset, labels, register):
    label = first_label(labels, "clean", "MEDICAL")
    docs = load_packet(dataset, label["claim_id"])
    extract_call = len(docs) + 1  # first extraction call follows the classify calls
    provider = RecordingProvider(faults={extract_call: "missing_field"})
    claim = run_intake(label["claim_id"], docs, provider, register)

    assert claim.status == "intake_complete"
    assert claim.trace[1].provider_calls == len(docs) + 1
    assert "missing" in provider.prompts[extract_call]


def test_missing_field_twice_fails_extraction_only(dataset, labels, register):
    label = first_label(labels, "clean", "MEDICAL")
    docs = load_packet(dataset, label["claim_id"])
    n = len(docs)
    claim = run_intake(
        label["claim_id"], docs, FakeProvider(faults={n + 1: "missing_field", n + 2: "missing_field"}), register
    )
    first = claim.documents[0]
    assert first.status == "failed"
    assert first.doc_type is not None  # classified, but extraction failed
    assert claim.status == "flagged"
    assert "MISSING_DOCUMENT" not in claim.reason_codes  # the document is present


def test_all_documents_failing_gives_failed_claim(dataset, labels, register):
    label = first_label(labels, "clean", "MOTOR")
    docs = load_packet(dataset, label["claim_id"])
    faults = {i: "malformed_json" for i in range(1, 2 * len(docs) + 1)}
    claim = run_intake(label["claim_id"], docs, FakeProvider(faults=faults), register)

    assert claim.status == "failed"
    assert all(d.status == "failed" for d in claim.documents)
    assert [t.status for t in claim.trace] == ["failed", "skipped", "skipped", "ok"]


def test_unrecognisable_document_fails_without_exception(register):
    docs = [InputDocument(doc_id="x1", text="SPECIMEN lorem ipsum")]
    claim = run_intake("C-1", docs, FakeProvider(), register)
    assert claim.status == "failed"


def test_raising_provider_is_not_replaced_by_fake_answers(dataset, labels, register):
    class Broken:
        name = "broken"

        def complete(self, prompt):
            raise ConnectionError("upstream down")

    label = first_label(labels, "clean", "MEDICAL")
    claim = run_intake(label["claim_id"], load_packet(dataset, label["claim_id"]), Broken(), register)
    assert claim.status == "failed"
    assert "provider error (ConnectionError)" in claim.trace[0].error
    assert claim.fields.certificate_no is None


def test_fake_ignores_file_names_and_packet_order(dataset, labels, register):
    """Same text under meaningless ids in reverse order classifies the same."""
    label = first_label(labels, "clean", "MOTOR")
    original = load_packet(dataset, label["claim_id"])
    renamed = [
        InputDocument(doc_id=f"doc{i}", text=d.text) for i, d in enumerate(reversed(original))
    ]
    claim = run_intake("other-id", renamed, FakeProvider(), register)
    expected = [doc_types(label)[d.doc_id] for d in reversed(original)]
    assert [d.doc_type.value for d in claim.documents] == expected


def test_fake_works_without_labels_or_generator(dataset, labels, register, tmp_path, monkeypatch):
    """Delete labels.jsonl and block the generator module: results are unchanged."""
    import sys

    label = first_label(labels, "clean", "MEDICAL")
    docs = load_packet(dataset, label["claim_id"])
    monkeypatch.setitem(sys.modules, "claims.generate", None)  # import would now fail
    monkeypatch.chdir(tmp_path)  # no data/labels.jsonl reachable
    claim = run_intake(label["claim_id"], docs, FakeProvider(), register)
    assert claim.status == "intake_complete"


# ---- validation rules on hand-built extractions

REGISTER = {"MED-2026-000001": {"certificate_no": "MED-2026-000001", "product": "MEDICAL"}}
MED_DOCS = {"claim_form", "hospital_bill", "discharge_summary"}


def med_extractions(**overrides):
    from claims.schemas import DocType

    claim_form = {
        "certificate_no": "MED-2026-000001", "product": "MEDICAL", "participant_name": "A B",
        "participant_ic": "900101-00-1234", "incident_date": date(2026, 3, 1),
        "amount_claimed": Decimal("1000.00"), "hospital_name": "Hospital X",
        "vehicle_reg_no": None, "cause_of_loss": None, "police_report_no": None,
    }  # fmt: skip
    bill = {
        "participant_name": "A B", "hospital_name": "Hospital X",
        "admission_date": date(2026, 3, 1), "discharge_date": date(2026, 3, 3),
        "room_rate_per_day": Decimal("200.00"), "bill_total": Decimal("1000.00"),
    }  # fmt: skip
    summary = {
        "participant_name": "A B", "hospital_name": "Hospital X",
        "admission_date": date(2026, 3, 1), "discharge_date": date(2026, 3, 3), "diagnosis": "D",
    }  # fmt: skip
    for key, value in overrides.items():
        for values in (claim_form, bill, summary):
            if key in values:
                values[key] = value
    return [
        Extraction("cf", DocType.CLAIM_FORM, claim_form),
        Extraction("hb", DocType.HOSPITAL_BILL, bill),
        Extraction("ds", DocType.DISCHARGE_SUMMARY, summary),
    ], {DocType(t) for t in MED_DOCS}


def test_clean_hand_built_claim_has_no_codes():
    assert validate(*med_extractions(), REGISTER)[1] == []


def test_cert_not_found():
    assert "CERT_NOT_FOUND" in validate(*med_extractions(certificate_no="MED-2026-999999"), REGISTER)[1]


def test_discharge_before_admission():
    extractions, present = med_extractions(discharge_date=date(2026, 2, 28))
    assert "DATE_ORDER_INVALID" in validate(extractions, present, REGISTER)[1]


def test_admission_before_incident():
    extractions, present = med_extractions(admission_date=date(2026, 2, 27))
    assert "DATE_ORDER_INVALID" in validate(extractions, present, REGISTER)[1]


def test_missing_document():
    from claims.schemas import DocType

    extractions, present = med_extractions()
    codes = validate(extractions[:2], present - {DocType.DISCHARGE_SUMMARY}, REGISTER)[1]
    assert "MISSING_DOCUMENT" in codes
    assert "FIELDS_INCOMPLETE" in codes  # diagnosis only comes from the summary


@pytest.mark.parametrize(
    ("claimed", "mismatch"),
    [("1010.00", False), ("990.00", False), ("1010.01", True), ("989.99", True)],
)
def test_amount_mismatch_threshold_is_strictly_more_than_one_percent(claimed, mismatch):
    extractions, present = med_extractions(amount_claimed=Decimal(claimed))
    assert ("AMOUNT_MISMATCH" in validate(extractions, present, REGISTER)[1]) is mismatch
    assert amounts_differ(Decimal(claimed), Decimal("1000.00")) is mismatch
