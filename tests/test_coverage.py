import json
import re
from datetime import date
from decimal import Decimal

import pytest

from claims.coverage import CoverageContext, calculate_payable, run_coverage
from claims.pipeline import run_intake
from claims.providers import FakeProvider, ProviderError
from claims.retrieval import build_query
from claims.schemas import ClaimFields
from tests.conftest import load_packet

MEDICAL_SCHEDULE = {"room_board_per_day": "250.00", "annual_limit": "100000.00"}
MOTOR_SCHEDULE = {"excess": "500.00"}


@pytest.fixture(scope="module")
def ctx(dataset, register):
    return CoverageContext.load(dataset, register)


def coverage_label(labels, variant, product=None):
    return next(
        lab
        for lab in labels
        if lab["coverage_scenario"] == variant and (product is None or lab["product"] == product)
    )


def intake(dataset, register, label):
    return run_intake(
        label["claim_id"], load_packet(dataset, label["claim_id"]), FakeProvider(), register
    )


# ---- generated data


def test_certificates_have_twelve_stable_clauses_and_match_schedules(dataset):
    for product, prefix in (("MEDICAL", "MED"), ("MOTOR", "MTR")):
        text = (dataset / "certificates" / f"{product}.md").read_text()
        assert "SPECIMEN" in text
        ids = re.findall(rf"^({prefix}-\d+\.\d+) ", text, re.M)
        assert len(ids) >= 12 and len(set(ids)) == len(ids)
        schedule = json.loads((dataset / "schedules" / f"{product}.json").read_text())
        for key, value in schedule.items():
            if key not in ("product",):
                assert str(value).replace(".00", "") in text  # wording quotes each limit


def test_each_coverage_scenario_has_at_least_four_claims(labels):
    counts = {}
    for lab in labels:
        key = lab["coverage_scenario"]
        if lab["scenario"] == "missing_document":
            key = "missing_document"
        counts[key] = counts.get(key, 0) + 1
    for variant in ("covered", "outside_period", "waiting_period", "excluded", "partial"):
        assert counts[variant] >= 4
    assert counts["missing_document"] >= 4
    assert all(
        lab["expected_decision"] == "REFER"
        for lab in labels
        if lab["scenario"] == "missing_document"
    )


# ---- retrieval


def test_retrieval_is_filtered_by_product(ctx):
    medical = ctx.index.retrieve("collision excess racing own damage", "MEDICAL", 20)
    motor = ctx.index.retrieve("room and board cosmetic hospital", "MOTOR", 20)
    assert all(c.clause_id.startswith("MED-") for c in medical)
    assert all(c.clause_id.startswith("MTR-") for c in motor)
    assert ctx.index.retrieve("anything", "TRAVEL", 5) == []


def test_retrieval_finds_relevant_clauses_and_is_deterministic(ctx):
    top = ctx.index.retrieve("Cosmetic rhinoplasty for appearance", "MEDICAL", 3)
    assert top[0].clause_id == "MED-5.1"
    motor = ctx.index.retrieve("damaged while taking part in a racing event", "MOTOR", 3)
    assert motor[0].clause_id == "MTR-4.1"
    assert ctx.index.retrieve("period of cover", "MOTOR", 5) == ctx.index.retrieve(
        "period of cover", "MOTOR", 5
    )
    scores = [c.score for c in ctx.index.retrieve("zzz", "MEDICAL", 14)]
    ids = [c.clause_id for c in ctx.index.retrieve("zzz", "MEDICAL", 14)]
    assert set(scores) == {0.0} and ids == sorted(ids)  # ties ordered by clause id


def test_query_uses_extracted_fields_only():
    fields = ClaimFields(
        product="MEDICAL", diagnosis="Dengue fever", incident_date=date(2026, 3, 10)
    )
    assert "Dengue fever" in build_query(fields, date(2026, 3, 1))
    assert "waiting period" in build_query(fields, date(2026, 3, 1))
    assert "waiting period" not in build_query(fields, date(2025, 3, 1))


# ---- decisions on generated claims


@pytest.mark.parametrize(
    "variant,product",
    [
        ("covered", "MEDICAL"),
        ("covered", "MOTOR"),
        ("partial", "MEDICAL"),
        ("partial", "MOTOR"),
        ("outside_period", None),
        ("waiting_period", "MEDICAL"),
    ],
)
def test_fake_decisions_match_labels(dataset, register, ctx, labels, variant, product):
    label = coverage_label(labels, variant, product)
    claim = run_coverage(intake(dataset, register, label), FakeProvider(), ctx)
    rec = claim.recommendation
    assert claim.status == "pending_adjuster"
    assert rec.decision.value == label["expected_decision"]
    assert f"{rec.payable_amount:.2f}" == label["expected_payable"]
    assert [c.clause_id for c in rec.cited_clauses] == label["expected_clause_ids"]
    texts = {c.clause_id: c.text for c in ctx.index.clauses[label["product"]]}
    assert all(c.quote in texts[c.clause_id] for c in rec.cited_clauses)


# ---- citation guardrails


@pytest.fixture
def partial_claim(dataset, register, labels):
    return intake(dataset, register, coverage_label(labels, "partial", "MEDICAL"))


@pytest.mark.parametrize("fault", ["wrong_quote", "unretrieved_clause", "drop_citations"])
def test_invalid_citation_twice_is_refer_citation_invalid(partial_claim, ctx, fault):
    provider = FakeProvider(faults={1: fault, 2: fault})
    claim = run_coverage(partial_claim, provider, ctx)
    rec = claim.recommendation
    assert rec.decision.value == "REFER"
    assert "CITATION_INVALID" in rec.reason_codes
    assert rec.payable_amount == Decimal("0.00")
    assert rec.cited_clauses == []
    assert provider.calls == 2
    assert claim.trace[-1].status == "failed" and claim.trace[-1].provider_calls == 2


def test_invalid_citation_then_valid_retry(partial_claim, ctx):
    provider = FakeProvider(faults={1: "wrong_quote"})
    claim = run_coverage(partial_claim, provider, ctx)
    assert claim.recommendation.decision.value == "PARTIALLY_COVERED"
    assert "CITATION_INVALID" not in claim.recommendation.reason_codes
    assert provider.calls == 2
    assert "quote is not an exact substring" in claim.trace[-1].error
    assert claim.trace[-1].status == "ok"


@pytest.mark.parametrize(
    "fault,code",
    [("malformed_json", "COVERAGE_OUTPUT_INVALID"), ("missing_field", "COVERAGE_OUTPUT_INVALID"),
     ("provider_error", "PROVIDER_ERROR")],
)  # fmt: skip
def test_provider_failures_refer_without_crashing(partial_claim, ctx, fault, code):
    claim = run_coverage(partial_claim, FakeProvider(faults={1: fault, 2: fault}), ctx)
    assert claim.recommendation.decision.value == "REFER"
    assert code in claim.recommendation.reason_codes
    assert claim.recommendation.payable_amount == Decimal("0.00")


def test_failing_real_provider_is_not_replaced_by_fake(partial_claim, ctx):
    class Broken:
        name = "broken"
        calls = 0

        def complete(self, prompt):
            self.calls += 1
            raise ProviderError("down")

    broken = Broken()
    rec = run_coverage(partial_claim, broken, ctx).recommendation
    assert broken.calls == 2 and rec.decision.value == "REFER" and rec.cited_clauses == []


def test_model_payable_is_ignored(partial_claim, ctx, labels):
    class Overpays(FakeProvider):
        def complete(self, prompt):
            answer = json.loads(super().complete(prompt))
            answer["payable_amount"] = "999999.00"
            return json.dumps(answer)

    # An extra key is a schema error, so the model cannot even supply a payable.
    rec = run_coverage(partial_claim, Overpays(), ctx).recommendation
    assert rec.decision.value == "REFER" and rec.payable_amount == Decimal("0.00")


@pytest.mark.parametrize("scenario", ["cert_not_found", "missing_document"])
def test_bypass_codes_refer_without_provider_call(dataset, register, ctx, labels, scenario):
    label = next(lab for lab in labels if lab["scenario"] == scenario)
    claim = intake(dataset, register, label)
    provider = FakeProvider()
    result = run_coverage(claim, provider, ctx)
    assert provider.calls == 0
    assert result.recommendation.decision.value == "REFER"
    assert result.reason_codes == claim.reason_codes  # intake codes preserved
    assert result.intake_status == "flagged"
    step = result.trace[-1]
    assert step.step == "coverage" and step.status == "skipped" and step.provider_calls == 0
    assert "bypassed" in step.error


def test_failed_intake_refers_without_provider_call(dataset, register, ctx, labels):
    label = coverage_label(labels, "covered", "MEDICAL")
    docs = load_packet(dataset, label["claim_id"])
    failing = FakeProvider(faults={i: "malformed_json" for i in range(1, 20)})
    claim = run_intake(label["claim_id"], docs, failing, register)
    assert claim.status == "failed"
    provider = FakeProvider()
    result = run_coverage(claim, provider, ctx)
    assert provider.calls == 0
    assert result.recommendation.decision.value == "REFER"
    assert "INTAKE_FAILED" in result.recommendation.reason_codes


# ---- payable, hand-checked


def medical(rate, days, bill, claimed=None):
    return ClaimFields(
        product="MEDICAL",
        admission_date=date(2026, 3, 1),
        discharge_date=date(2026, 3, 1 + days),
        room_rate_per_day=Decimal(rate),
        bill_total=Decimal(bill),
        amount_claimed=Decimal(claimed or bill),
    )


def test_medical_room_rate_sub_limit():
    # 3 days at RM 400 against a RM 250 limit: (400 - 250) * 3 = 450 not paid.
    # 3000.00 - 450.00 = 2550.00
    fields = medical("400.00", 3, "3000.00")
    assert calculate_payable(fields, MEDICAL_SCHEDULE, Decimal("50000")) == Decimal("2550.00")


def test_medical_room_rate_at_limit_is_paid_in_full_and_same_day_counts_one_day():
    assert calculate_payable(
        medical("250.00", 2, "1800.00"), MEDICAL_SCHEDULE, Decimal("50000")
    ) == Decimal("1800.00")
    # Same-day discharge counts as one day: (300 - 250) * 1 = 50.
    assert calculate_payable(
        medical("300.00", 0, "900.00"), MEDICAL_SCHEDULE, Decimal("50000")
    ) == Decimal("850.00")


def test_medical_limit_boundaries():
    # The lower of annual limit and sum covered caps the payable.
    big = medical("200.00", 2, "120000.00")
    assert calculate_payable(big, MEDICAL_SCHEDULE, Decimal("150000")) == Decimal("100000.00")
    assert calculate_payable(big, MEDICAL_SCHEDULE, Decimal("50000")) == Decimal("50000.00")
    exact = medical("200.00", 2, "50000.00")
    assert calculate_payable(exact, MEDICAL_SCHEDULE, Decimal("50000")) == Decimal("50000.00")


def motor(repair, cause, claimed=None):
    return ClaimFields(
        product="MOTOR",
        repair_total=Decimal(repair),
        cause_of_loss=cause,
        amount_claimed=Decimal(claimed or repair),
    )


def test_motor_excess():
    # RM 2,345.60 collision repair less RM 500.00 excess = RM 1,845.60.
    fields = motor("2345.60", "collision with another vehicle at a junction")
    assert calculate_payable(fields, MOTOR_SCHEDULE, Decimal("30000")) == Decimal("1845.60")
    flood = motor("2345.60", "flood damage during heavy rain")
    assert calculate_payable(flood, MOTOR_SCHEDULE, Decimal("30000")) == Decimal("2345.60")


def test_motor_never_negative_and_capped_by_sum_covered():
    assert calculate_payable(
        motor("300.00", "hit a pole"), MOTOR_SCHEDULE, Decimal("30000")
    ) == Decimal("0.00")
    assert calculate_payable(
        motor("40500.00", "hit a pole"), MOTOR_SCHEDULE, Decimal("30000")
    ) == Decimal("30000.00")


def test_missing_payable_inputs_return_none():
    fields = ClaimFields(product="MEDICAL", bill_total=Decimal("100"))
    assert calculate_payable(fields, MEDICAL_SCHEDULE, Decimal("50000")) is None


def test_not_covered_and_refer_pay_zero(dataset, register, ctx, labels):
    for variant in ("outside_period", "waiting_period", "excluded", "refer"):
        for lab in [x for x in labels if x["coverage_scenario"] == variant][:2]:
            rec = run_coverage(intake(dataset, register, lab), FakeProvider(), ctx).recommendation
            assert rec.decision.value in ("NOT_COVERED", "REFER")
            assert rec.payable_amount == Decimal("0.00")
