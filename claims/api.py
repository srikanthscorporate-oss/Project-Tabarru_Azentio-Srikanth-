"""FastAPI app.

    uvicorn --factory claims.api:create_app

POST /v1/claims   201 with the claim; 409 if the claim_id already exists;
                  422 for an invalid body (the error never echoes the input, and
                  IC-shaped text in error locations is masked).
Error messages never echo the claim_id.
GET  /v1/claims/{claim_id}   200 with the claim and its trace; 404 if unknown.
GET  /v1/claims/{claim_id}/audit  200 with the claim's audit events; 404 unknown.
GET  /v1/certificates/{product}/clauses/{clause_id}
                  200 with the clause from the current wording file; 404 unknown.
GET  /metrics     Prometheus text format (see claims/metrics.py).
GET  /            the adjuster view (static files in claims/web/).
GET  /v1/adjuster/queue      200 with pending_adjuster claims and recommendations.
POST /v1/claims/{claim_id}/adjuster-decision
                  200 with the decided claim; 404 unknown; 409 not pending
                  (already decided); 422 invalid or inconsistent decision.
POST runs intake then coverage; every claim ends pending_adjuster.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from claims import metrics
from claims.classifier import ClassifierConfig, get_classifier_config
from claims.coverage import CoverageContext, assess_claim
from claims.normalize import redact_ics
from claims.providers import LLMProvider, get_provider
from claims.schemas import (
    AdjusterDecisionRecord,
    AdjusterDecisionRequest,
    AuditEvent,
    ClaimDetail,
    ClaimRequest,
    ClaimResponse,
    ClauseSource,
    Decision,
    Product,
    QueueItem,
)
from claims.storage import AlreadyDecidedError, ClaimNotFoundError, ClaimStore, DuplicateClaimError
from claims.validation import load_register

WEB_DIR = Path(__file__).parent / "web"


class RedactIcFilter(logging.Filter):
    """Masks IC-shaped text in log arguments, such as the URL path in uvicorn's access log."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.args, tuple):
            record.args = tuple(redact_ics(a) if isinstance(a, str) else a for a in record.args)
        return True


def _redact_access_log() -> None:
    access = logging.getLogger("uvicorn.access")
    if not any(isinstance(f, RedactIcFilter) for f in access.filters):
        access.addFilter(RedactIcFilter())


def create_app(
    db_path: Path | str | None = None,
    register_path: Path | str | None = None,
    provider: LLMProvider | None = None,
    data_dir: Path | str | None = None,
    classifier: ClassifierConfig | None = None,
) -> FastAPI:
    store = ClaimStore(db_path or os.environ.get("CLAIMS_DB", "data/claims.db"))
    register_file = Path(register_path or os.environ.get("CLAIMS_REGISTER", "data/policies.jsonl"))
    register = load_register(register_file)
    # Certificates and schedules sit next to the register unless CLAIMS_DATA says otherwise.
    ctx = CoverageContext.load(
        data_dir or os.environ.get("CLAIMS_DATA") or register_file.parent, register
    )
    llm = provider or get_provider()
    classifier = classifier or get_classifier_config()  # fails fast on a bad model or mode
    _redact_access_log()
    app = FastAPI(title="Project Tabarru claim intake", version="0.1.0")

    @app.exception_handler(RequestValidationError)
    async def _invalid_body(_: Request, exc: RequestValidationError) -> JSONResponse:
        # FastAPI's default 422 echoes the submitted input, which could contain
        # a full IC number inside document text. Report locations and types only,
        # and redact the locations too: an unexpected key name is user input.
        detail = [
            {
                "loc": [redact_ics(p) if isinstance(p, str) else p for p in err["loc"]],
                "type": err["type"],
                "msg": redact_ics(err["msg"]),
            }
            for err in exc.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": detail})

    @app.post("/v1/claims", response_model=ClaimResponse, status_code=201)
    def submit_claim(request: ClaimRequest) -> ClaimResponse:
        if store.exists(request.claim_id):
            raise HTTPException(409, "claim already exists")
        claim = assess_claim(request.claim_id, request.documents, llm, ctx, classifier)
        try:
            store.save(claim, request.documents)
        except DuplicateClaimError:
            raise HTTPException(409, "claim already exists") from None
        product = claim.fields.product.value if claim.fields.product else None
        metrics.record_claim(claim.trace, product, claim.recommendation.reason_codes)
        return claim

    @app.get("/v1/claims/{claim_id}", response_model=ClaimDetail)
    def read_claim(claim_id: str) -> ClaimDetail:
        claim = store.get(claim_id)
        if claim is None:
            raise HTTPException(404, "claim not found")
        return claim

    @app.get("/v1/claims/{claim_id}/audit", response_model=list[AuditEvent])
    def claim_audit(claim_id: str) -> list[dict]:
        if not store.exists(claim_id):
            raise HTTPException(404, "claim not found")
        return store.audit(claim_id)

    clauses = {
        (product, c.clause_id): c.text for product, cs in ctx.index.clauses.items() for c in cs
    }

    @app.get("/v1/certificates/{product}/clauses/{clause_id}", response_model=ClauseSource)
    def clause_source(product: Product, clause_id: str) -> ClauseSource:
        # Dictionary lookup only: the path never reaches the filesystem.
        text = clauses.get((product.value, clause_id))
        if text is None:
            raise HTTPException(404, "clause not found")
        return ClauseSource(product=product, clause_id=clause_id, text=text)

    @app.get("/metrics", response_class=PlainTextResponse)
    def prometheus_metrics() -> PlainTextResponse:
        return PlainTextResponse(
            metrics.render(store.status_counts()), media_type="text/plain; version=0.0.4"
        )

    @app.get("/", include_in_schema=False)
    def adjuster_view() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @app.get("/v1/adjuster/queue", response_model=list[QueueItem])
    def adjuster_queue() -> list[dict]:
        return store.queue()

    @app.post("/v1/claims/{claim_id}/adjuster-decision", response_model=ClaimDetail)
    def adjuster_decision(claim_id: str, body: AdjusterDecisionRequest) -> ClaimDetail:
        claim = store.get(claim_id)
        if claim is None:
            raise HTTPException(404, "claim not found")
        if claim.status != "pending_adjuster" or claim.recommendation is None:
            raise HTTPException(409, "claim is not pending an adjuster decision")
        record = _decision_record(body, claim)
        try:
            store.decide(claim_id, record)
        except ClaimNotFoundError:
            raise HTTPException(404, "claim not found") from None
        except AlreadyDecidedError:
            raise HTTPException(409, "claim is not pending an adjuster decision") from None
        return store.get(claim_id)

    return app


def _decision_record(body: AdjusterDecisionRequest, claim: ClaimDetail) -> AdjusterDecisionRecord:
    """Check the decision against the recommendation; HTTP 422 when inconsistent."""
    rec = claim.recommendation
    reason = redact_ics(body.reason.strip()) if body.reason and body.reason.strip() else None
    if body.action == "approve":
        decision = body.final_decision or rec.decision
        payable = rec.payable_amount if body.final_payable is None else body.final_payable
        if decision != rec.decision or payable != rec.payable_amount:
            raise HTTPException(422, "approve must match the recommendation; use override")
    else:
        if reason is None:
            raise HTTPException(422, "override requires a non-empty reason")
        if body.final_decision is None or body.final_payable is None:
            raise HTTPException(422, "override requires final_decision and final_payable")
        decision, payable = body.final_decision, body.final_payable
    if decision in (Decision.NOT_COVERED, Decision.REFER) and payable != 0:
        raise HTTPException(422, "final_payable must be 0.00 for NOT_COVERED and REFER")
    # Zero-payable policy: PARTIALLY_COVERED may carry 0.00 (for example an excess
    # that absorbs the whole loss), so approving that recommendation is allowed.
    # COVERED always needs a payable above 0.00.
    if decision is Decision.COVERED and payable <= 0:
        raise HTTPException(422, "final_payable must be above 0.00 for COVERED")
    if decision is Decision.PARTIALLY_COVERED and payable < 0:
        raise HTTPException(422, "final_payable must not be negative")
    return AdjusterDecisionRecord(
        adjuster=redact_ics(body.adjuster),
        action=body.action,
        final_decision=decision,
        final_payable=payable,
        reason=reason,
    )
