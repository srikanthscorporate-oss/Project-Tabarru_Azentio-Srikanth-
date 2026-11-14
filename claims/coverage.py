"""Coverage step: retrieve clauses, ask the provider for a cited decision, compute payable.

    run_coverage(claim, provider, context) -> ClaimDetail (status pending_adjuster)

Guardrails, all in code:
- the answer must parse and match CoverageOutput;
- every cited clause_id must be one of the retrieved clauses, and every quote a
  non-empty exact substring of that clause's text (and IC-free);
- NOT_COVERED and PARTIALLY_COVERED need at least one citation.
A violation is retried once with a short description of the problem (no
quoted model text). A second violation gives REFER with CITATION_INVALID.
A schema failure twice gives REFER with COVERAGE_OUTPUT_INVALID, a provider
exception twice gives REFER with PROVIDER_ERROR. There is no fallback to the
fake provider.

payable_amount is computed here from the schedule and the extracted fields.
The model never supplies it.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from pydantic import ValidationError

from claims.normalize import contains_ic, redact_ics
from claims.pipeline import _Step, describe_error, run_intake
from claims.providers import LLMProvider, build_coverage_prompt
from claims.retrieval import DEFAULT_K, Clause, ClauseIndex, build_query
from claims.schemas import (
    ClaimDetail,
    ClaimFields,
    CoverageOutput,
    Decision,
    InputDocument,
    Recommendation,
)

log = logging.getLogger("claims.coverage")

MAX_ATTEMPTS = 2
CENT = Decimal("0.01")
ZERO = Decimal("0.00")
# Intake codes that send a claim straight to REFER without a provider call.
BYPASS_CODES = ("CERT_NOT_FOUND", "MISSING_DOCUMENT")
SPECIAL_PERILS = ("flood", "tree")  # MTR-5.2: no excess when damage is only from these


@dataclass
class CoverageContext:
    index: ClauseIndex
    schedules: dict[str, dict]
    register: dict[str, dict]
    k: int = DEFAULT_K

    @classmethod
    def load(cls, data_dir: Path | str, register: dict[str, dict]) -> CoverageContext:
        data_dir = Path(data_dir)
        schedules = {
            p: json.loads((data_dir / "schedules" / f"{p}.json").read_text(encoding="utf-8"))
            for p in ("MEDICAL", "MOTOR")
        }
        return cls(ClauseIndex.load(data_dir / "certificates"), schedules, register)


class CitationError(ValueError):
    pass


def check_citations(output: CoverageOutput, retrieved: list[Clause]) -> None:
    """Raise CitationError with a safe message if any citation rule is broken."""
    texts = {c.clause_id: c.text for c in retrieved}
    if output.decision in (Decision.NOT_COVERED, Decision.PARTIALLY_COVERED) and not (
        output.cited_clauses
    ):
        raise CitationError(f"{output.decision.value} must cite at least one clause")
    for i, cited in enumerate(output.cited_clauses, start=1):
        if cited.clause_id not in texts:
            raise CitationError(f"citation {i}: clause_id is not one of the retrieved clauses")
        if not cited.quote.strip() or cited.quote not in texts[cited.clause_id]:
            raise CitationError(f"citation {i}: quote is not an exact substring of its clause")
        if contains_ic(cited.quote):
            raise CitationError(f"citation {i}: quote contains an IC number")


def _money(value: Decimal) -> Decimal:
    return max(value, ZERO).quantize(CENT, rounding=ROUND_HALF_UP)


def calculate_payable(
    fields: ClaimFields, schedule: dict, sum_covered: Decimal | None
) -> Decimal | None:
    """Payable for a COVERED or PARTIALLY_COVERED claim; None when an input is missing.

    MEDICAL: days = max(discharge - admission in days, 1) (MED-4.3)
             room_excess = max(room_rate - room_limit, 0) * days (MED-4.1)
             payable = min(bill_total - room_excess, annual_limit, sum_covered, amount_claimed)
    MOTOR:   excess = schedule excess unless the cause is only flood or a falling tree
             payable = min(repair_total - excess, sum_covered, amount_claimed)
    Deductions come first, then the caps; the result is floored at 0.00 and
    rounded half up to cents.
    """
    caps = [c for c in (sum_covered, fields.amount_claimed) if c is not None]
    if sum_covered is None:
        return None
    if fields.product is not None and fields.product.value == "MEDICAL":
        needed = (fields.bill_total, fields.room_rate_per_day)
        if None in needed or fields.admission_date is None or fields.discharge_date is None:
            return None
        days = max((fields.discharge_date - fields.admission_date).days, 1)
        limit = Decimal(schedule["room_board_per_day"])
        room_excess = max(fields.room_rate_per_day - limit, ZERO) * days
        gross = fields.bill_total - room_excess
        caps.append(Decimal(schedule["annual_limit"]))
    elif fields.product is not None and fields.product.value == "MOTOR":
        if fields.repair_total is None or fields.cause_of_loss is None:
            return None
        cause = fields.cause_of_loss.lower()
        special = any(word in cause for word in SPECIAL_PERILS)
        gross = fields.repair_total - (ZERO if special else Decimal(schedule["excess"]))
    else:
        return None
    return _money(min([gross, *caps]))


def _facts(fields: ClaimFields, cert: dict | None) -> dict:
    """Claim facts sent to the provider. The IC is left out: coverage does not need it."""
    facts = fields.model_dump(mode="json", exclude={"participant_ic"})
    facts["cert_start"] = cert.get("start_date") if cert else None
    facts["cert_end"] = cert.get("end_date") if cert else None
    return facts


def _refer(codes: list[str], rationale: str, cited=None) -> Recommendation:
    return Recommendation(
        decision=Decision.REFER,
        cited_clauses=cited or [],
        rationale=rationale,
        payable_amount=ZERO,
        reason_codes=codes,
    )


def _assess(
    claim: ClaimDetail, provider: LLMProvider, ctx: CoverageContext, step: _Step
) -> Recommendation:
    fields = claim.fields
    cert = ctx.register.get(fields.certificate_no or "")
    start = date.fromisoformat(cert["start_date"]) if cert else None
    product = fields.product.value
    retrieved = ctx.index.retrieve(build_query(fields, start), product, ctx.k)
    step.clause_ids = [c.clause_id for c in retrieved]
    evidence = [{"clause_id": c.clause_id, "text": c.text} for c in retrieved]
    facts = _facts(fields, cert)

    error, kind, output = None, None, None
    for _ in range(MAX_ATTEMPTS):
        step.calls += 1
        try:
            raw = provider.complete(build_coverage_prompt(facts, evidence, error))
            candidate = CoverageOutput.model_validate(json.loads(raw))
            check_citations(candidate, retrieved)
            output = candidate
            break
        except CitationError as exc:
            error, kind = str(exc), "CITATION_INVALID"
        except (json.JSONDecodeError, ValidationError) as exc:
            error, kind = describe_error(exc), "COVERAGE_OUTPUT_INVALID"
        except Exception as exc:  # a real provider failing; never fall back to fake answers
            error, kind = describe_error(exc), "PROVIDER_ERROR"
        step.errors.append(f"attempt {step.calls}: {error}")
    if output is None:
        return _refer(list(claim.reason_codes) + [kind], f"coverage not assessed: {error}")

    rationale = redact_ics(output.rationale)
    if output.decision in (Decision.NOT_COVERED, Decision.REFER):
        payable = ZERO
    else:
        sum_covered = Decimal(cert["sum_covered"]) if cert else None
        payable = calculate_payable(fields, ctx.schedules[product], sum_covered)
        if payable is None:
            step.errors.append("payable inputs missing")
            return _refer(
                list(claim.reason_codes) + ["CALC_INPUT_MISSING"],
                rationale,
                output.cited_clauses,
            )
    return Recommendation(
        decision=output.decision,
        cited_clauses=output.cited_clauses,
        rationale=rationale,
        payable_amount=payable,
        reason_codes=list(claim.reason_codes),
    )


def run_coverage(claim: ClaimDetail, provider: LLMProvider, ctx: CoverageContext) -> ClaimDetail:
    """Add the coverage recommendation and move the claim to pending_adjuster.

    Bypass (REFER, no provider call): intake failed, any intake reason code
    (CERT_NOT_FOUND and MISSING_DOCUMENT as the assignment requires, and also
    DATE_ORDER_INVALID, AMOUNT_MISMATCH, FIELDS_INCOMPLETE, because the facts
    the decision and payable depend on are unreliable), a failed document, or
    an unknown product.
    """
    step = _Step("coverage", [d.doc_id for d in claim.documents])
    step.clause_ids = []
    codes = list(claim.reason_codes)
    if claim.status != "intake_complete" or claim.fields.product is None:
        if claim.status == "failed":
            codes.append("INTAKE_FAILED")
        elif not codes:
            codes.append("INTAKE_INCOMPLETE")
        rec = _refer(codes, "referred without coverage assessment: intake did not complete cleanly")
        bypass = [c for c in codes if c in BYPASS_CODES or c.startswith("INTAKE_")]
        step.errors.append("bypassed: " + ", ".join(bypass or codes))
        entry = step.finish("skipped")
    else:
        rec = _assess(claim, provider, ctx, step)
        entry = step.finish("failed" if step.errors and rec.decision == Decision.REFER else "ok")
    entry.clause_ids = step.clause_ids
    log.info(
        "claim %s: decision=%s payable=%s codes=%s",
        claim.claim_id,
        rec.decision.value,
        rec.payable_amount,
        rec.reason_codes,
    )
    return claim.model_copy(
        update={
            "intake_status": claim.status,
            "status": "pending_adjuster",
            "recommendation": rec,
            "trace": [*claim.trace, entry],
        }
    )


def assess_claim(
    claim_id: str,
    documents: list[InputDocument],
    provider: LLMProvider,
    ctx: CoverageContext,
) -> ClaimDetail:
    """Intake then coverage: what POST /v1/claims and the evaluator run."""
    return run_coverage(run_intake(claim_id, documents, provider, ctx.register), provider, ctx)
