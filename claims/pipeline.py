"""The intake agent: an explicit four-step pipeline with a trace entry per step.

    classify_documents -> extract_fields -> validate -> assemble_claim

Model output is never trusted: each answer is parsed as JSON and validated
against a Pydantic schema. A bad answer is retried once with the (sanitised)
validation error in the prompt; a second bad answer marks that document
failed and the pipeline carries on with the rest of the packet.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field

from pydantic import BaseModel, ValidationError

from claims.normalize import mask_ic, redact_ics
from claims.providers import LLMProvider, Task, build_prompt
from claims.schemas import (
    EXTRACTION_SCHEMAS,
    ClaimDetail,
    ClaimFields,
    ClassifyOutput,
    DocType,
    DocumentResult,
    InputDocument,
    TraceEntry,
)
from claims.validation import Extraction, validate

log = logging.getLogger("claims.pipeline")

MAX_ATTEMPTS = 2  # first try plus one retry


@dataclass
class DocState:
    doc_id: str
    text: str
    doc_type: DocType | None = None
    values: dict | None = None
    error: str | None = None

    @property
    def processed(self) -> bool:
        return self.values is not None


@dataclass
class _Step:
    name: str
    doc_ids: list[str]
    started: float = field(default_factory=time.perf_counter)
    calls: int = 0
    errors: list[str] = field(default_factory=list)

    def finish(self, status: str) -> TraceEntry:
        return TraceEntry(
            step=self.name,
            doc_ids=self.doc_ids,
            status=status,
            duration_ms=round((time.perf_counter() - self.started) * 1000, 3),
            provider_calls=self.calls,
            error=redact_ics("; ".join(self.errors)) if self.errors else None,
        )


def describe_error(exc: Exception) -> str:
    """Summarise a parse or schema error without echoing any input values."""
    if isinstance(exc, json.JSONDecodeError):
        return f"output is not valid JSON ({exc.msg} at char {exc.pos})"
    if isinstance(exc, ValidationError):
        problems = [
            f"{'.'.join(str(p) for p in err['loc']) or 'root'}: {err['type']}"
            for err in exc.errors(include_input=False, include_url=False)
        ]
        return "schema validation failed (" + ", ".join(problems) + ")"
    return f"provider error ({type(exc).__name__})"


def call_with_retry(
    provider: LLMProvider,
    step: _Step,
    task: Task,
    schema: type[BaseModel],
    text: str,
    doc_type: DocType | None = None,
) -> tuple[BaseModel | None, str | None]:
    """Ask the provider, validate, retry once. Returns (result, last error)."""
    error = None
    for _ in range(MAX_ATTEMPTS):
        step.calls += 1
        try:
            raw = provider.complete(build_prompt(task, text, doc_type, error))
            return schema.model_validate(json.loads(raw)), None
        except (json.JSONDecodeError, ValidationError) as exc:
            error = describe_error(exc)
        except Exception as exc:  # a real provider failing; never fall back to fake answers
            error = describe_error(exc)
    return None, error


def classify_documents(docs: list[DocState], provider: LLMProvider) -> TraceEntry:
    step = _Step("classify_documents", [d.doc_id for d in docs])
    for doc in docs:
        result, error = call_with_retry(
            provider, step, "classify_document", ClassifyOutput, doc.text
        )
        if result is None:
            doc.error = f"classification failed: {error}"
            step.errors.append(f"{doc.doc_id}: {doc.error}")
        else:
            doc.doc_type = result.doc_type
    return step.finish(_status(len(docs), sum(d.doc_type is not None for d in docs)))


def extract_fields(docs: list[DocState], provider: LLMProvider) -> TraceEntry:
    todo = [d for d in docs if d.doc_type is not None]
    step = _Step("extract_fields", [d.doc_id for d in todo])
    for doc in todo:
        schema = EXTRACTION_SCHEMAS[doc.doc_type]
        result, error = call_with_retry(
            provider, step, "extract_fields", schema, doc.text, doc.doc_type
        )
        if result is None:
            doc.error = f"extraction failed: {error}"
            step.errors.append(f"{doc.doc_id}: {doc.error}")
        else:
            doc.values = result.model_dump()
    if not todo:
        return step.finish("skipped")
    return step.finish(_status(len(todo), sum(d.processed for d in todo)))


def _status(attempted: int, succeeded: int) -> str:
    if succeeded == attempted:
        return "ok"
    return "failed" if succeeded == 0 else "partial"


def run_intake(
    claim_id: str,
    documents: list[InputDocument],
    provider: LLMProvider,
    register: dict[str, dict],
) -> ClaimDetail:
    """Process one claim packet. Never raises on bad model output."""
    docs = [DocState(d.doc_id, d.text) for d in documents]
    trace = [classify_documents(docs, provider), extract_fields(docs, provider)]

    processed = [d for d in docs if d.processed]
    step = _Step("validate", [d.doc_id for d in processed])
    if processed:
        extractions = [Extraction(d.doc_id, d.doc_type, d.values) for d in processed]
        classified = {d.doc_type for d in docs if d.doc_type is not None}
        merged, reason_codes = validate(extractions, classified, register)
        trace.append(step.finish("ok"))
    else:
        merged, reason_codes = {}, []
        trace.append(step.finish("skipped"))

    step = _Step("assemble_claim", [d.doc_id for d in docs])
    merged["participant_ic"] = mask_ic(merged.get("participant_ic"))
    fields = ClaimFields(**merged)
    if not processed:
        status = "failed"
    elif reason_codes or len(processed) < len(docs):
        status = "flagged"
    else:
        status = "intake_complete"
    trace.append(step.finish("ok"))

    log.info(
        "claim %s: status=%s reason_codes=%s failed_docs=%s",
        claim_id,
        status,
        reason_codes,
        [d.doc_id for d in docs if not d.processed],
    )
    return ClaimDetail(
        claim_id=claim_id,
        status=status,
        documents=[
            DocumentResult(
                doc_id=d.doc_id,
                doc_type=d.doc_type,
                status="processed" if d.processed else "failed",
            )
            for d in docs
        ],
        fields=fields,
        reason_codes=reason_codes,
        trace=trace,
    )
