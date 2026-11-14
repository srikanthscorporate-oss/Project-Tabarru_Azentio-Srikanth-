"""FastAPI app.

    uvicorn --factory claims.api:create_app

POST /v1/claims   201 with the claim; 409 if the claim_id already exists;
                  422 for an invalid body (the error never echoes the input).
GET  /v1/claims/{claim_id}   200 with the claim and its trace; 404 if unknown.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from claims.pipeline import run_intake
from claims.providers import LLMProvider, get_provider
from claims.schemas import ClaimDetail, ClaimRequest, ClaimResponse
from claims.storage import ClaimStore, DuplicateClaimError
from claims.validation import load_register


def create_app(
    db_path: Path | str | None = None,
    register_path: Path | str | None = None,
    provider: LLMProvider | None = None,
) -> FastAPI:
    store = ClaimStore(db_path or os.environ.get("CLAIMS_DB", "data/claims.db"))
    register = load_register(
        Path(register_path or os.environ.get("CLAIMS_REGISTER", "data/policies.jsonl"))
    )
    llm = provider or get_provider()
    app = FastAPI(title="Project Tabarru claim intake", version="0.1.0")

    @app.exception_handler(RequestValidationError)
    async def _invalid_body(_: Request, exc: RequestValidationError) -> JSONResponse:
        # FastAPI's default 422 echoes the submitted input, which could contain
        # a full IC number inside document text. Report locations and types only.
        detail = [
            {"loc": list(err["loc"]), "type": err["type"], "msg": err["msg"]}
            for err in exc.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": detail})

    @app.post("/v1/claims", response_model=ClaimResponse, status_code=201)
    def submit_claim(request: ClaimRequest) -> ClaimResponse:
        if store.exists(request.claim_id):
            raise HTTPException(409, f"claim {request.claim_id} already exists")
        claim = run_intake(request.claim_id, request.documents, llm, register)
        try:
            store.save(claim, request.documents)
        except DuplicateClaimError:
            raise HTTPException(409, f"claim {request.claim_id} already exists") from None
        return claim

    @app.get("/v1/claims/{claim_id}", response_model=ClaimDetail)
    def read_claim(claim_id: str) -> ClaimDetail:
        claim = store.get(claim_id)
        if claim is None:
            raise HTTPException(404, f"claim {claim_id} not found")
        return claim

    return app
