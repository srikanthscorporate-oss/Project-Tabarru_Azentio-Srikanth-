"""LLM provider interface, the deterministic FakeProvider and an optional real adapter.

The pipeline only ever calls ``provider.complete(prompt) -> str`` and treats
the answer as untrusted JSON. Prompts are built by ``build_prompt`` with a
fixed layout so a real model and the fake read the same input.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from datetime import date
from typing import Literal, Protocol

from claims.normalize import fix_ocr_digits, fix_ocr_identifier, parse_amount, parse_date
from claims.schemas import EXTRACTION_SCHEMAS, DocType

Task = Literal["classify_document", "extract_fields"]
Fault = Literal[
    "malformed_json",
    "missing_field",
    # coverage faults, used by guardrail tests
    "wrong_quote",
    "unretrieved_clause",
    "drop_citations",
    "provider_error",
]

_DOC_START, _DOC_END = "<<<DOCUMENT", "DOCUMENT>>>"


class LLMProvider(Protocol):
    name: str

    def complete(self, prompt: str) -> str:
        """Return the model's raw text answer for one prompt."""
        ...


class ProviderError(RuntimeError):
    """The provider could not produce an answer (network, HTTP, timeout)."""


def build_prompt(
    task: Task, text: str, doc_type: DocType | None = None, previous_error: str | None = None
) -> str:
    if task == "classify_document":
        types = ", ".join(t.value for t in DocType)
        instructions = (
            f"Classify this OCR'd Takaful claim document as exactly one of: {types}.\n"
            'Reply with JSON only: {"doc_type": "<type>"}'
        )
    else:
        assert doc_type is not None
        keys = ", ".join(EXTRACTION_SCHEMAS[doc_type].model_fields)
        instructions = (
            f"Extract fields from this {doc_type.value} document.\n"
            f"Reply with one JSON object with exactly these keys: {keys}.\n"
            "Use null when a value is not in the document. Dates as YYYY-MM-DD "
            "(numeric dates in the document are day first), amounts as MYR numbers "
            "with two decimals, product as MOTOR or MEDICAL."
        )
    lines = [f"TASK: {task}"]
    if doc_type is not None:
        lines.append(f"DOC_TYPE: {doc_type.value}")
    lines.append(instructions)
    if previous_error:
        lines.append(f"Your previous answer was rejected: {previous_error}. Return corrected JSON.")
    lines += [_DOC_START, text, _DOC_END]
    return "\n".join(lines)


COVERAGE_TASK = "assess_coverage"
_EVIDENCE_START, _EVIDENCE_END = "<<<EVIDENCE", "EVIDENCE>>>"


def build_coverage_prompt(facts: dict, clauses: list[dict], previous_error: str | None) -> str:
    """Prompt for the coverage decision. Evidence is one JSON object between markers."""
    lines = [
        f"TASK: {COVERAGE_TASK}",
        "Decide whether this Takaful claim is covered, using only the certificate clauses given.",
        'Reply with JSON only: {"decision": "COVERED|PARTIALLY_COVERED|NOT_COVERED|REFER", '
        '"cited_clauses": [{"clause_id": "...", "quote": "..."}], "rationale": "..."}.',
        "Each quote must be copied word for word from the cited clause. NOT_COVERED and "
        "PARTIALLY_COVERED must cite at least one clause. Do not state a payable amount.",
    ]
    if previous_error:
        lines.append(f"Your previous answer was rejected: {previous_error}. Return corrected JSON.")
    evidence = json.dumps({"claim": facts, "clauses": clauses}, sort_keys=True)
    lines += [_EVIDENCE_START, evidence, _EVIDENCE_END]
    return "\n".join(lines)


# ---- FakeProvider

# Phrases that identify a document type, weighted. Titles score higher.
_TYPE_SIGNALS: dict[DocType, list[tuple[str, int]]] = {
    DocType.CLAIM_FORM: [
        ("claim form", 5), ("notification of claim", 5), ("declaration", 2),
        ("amount claimed", 2), ("sum claimed", 2),
    ],
    DocType.HOSPITAL_BILL: [
        ("hospital bill", 5), ("statement of account", 5), ("room charges", 2),
        ("room and board", 2), ("total amount due", 2), ("grand total", 2),
    ],
    DocType.DISCHARGE_SUMMARY: [
        ("discharge summary", 5), ("summary on discharge", 5), ("diagnosis", 2),
        ("attending physician", 2),
    ],
    DocType.REPAIR_ESTIMATE: [
        ("repair estimate", 5), ("quotation for repairs", 5), ("workshop", 2), ("labour", 2),
    ],
    DocType.POLICE_REPORT: [
        ("police report", 3), ("laporan polis", 5), ("investigating officer", 3),
        ("balai polis", 2),
    ],
}  # fmt: skip

_SEP = r"[ \t:.|]*"
_DATE = r"([0-9OIl]{1,4}[/-][0-9OIl]{1,2}[/-][0-9OIl]{2,4}|[0-9OIl]{1,2}[ -][A-Za-z]{3,9}[ -][0-9OIl]{4})"
_AMOUNT = r"((?:RM|MYR)?\s?[0-9OIl][0-9OIl,]*(?:\.[0-9OIl]{1,2})?)"
_NAME = r"([A-Z][A-Za-z']+(?: [A-Z][A-Za-z']+)+)"
_TEXT = r"([^\n|]+?)\.?[ \t]*$"
_HOSPITAL = r"((?:Hospital|Pusat Perubatan)(?: [A-Z][a-z]+)+)"


def _label(*labels: str) -> str:
    return r"(?im)^\s*(?:" + "|".join(re.escape(x) for x in labels) + ")" + _SEP


# field -> list of (regex, value kind). The first pattern that matches wins.
_PATTERNS: dict[str, list[tuple[str, str]]] = {
    "certificate_no": [
        (_label("Certificate No", "Takaful certificate") + r"([A-Z]{3}-[0-9OIl]{4}-[0-9OIl]{6})", "id")
    ],
    "product": [(_label("Product", "Plan") + r"(motor|medical)", "product")],
    "participant_name": [
        (_label("Participant Name", "Name of participant", "Patient name", "Patient") + _NAME, "str"),
        (r"(?m)^" + _NAME + r" was admitted", "str"),
    ],
    "participant_ic": [
        (_label("IC No", "NRIC") + r"([0-9OIl]{6}-[0-9OIl]{2}-[0-9OIl]{4})", "ic")
    ],
    "incident_date": [
        (_label("Date of Incident", "Loss date", "Incident date") + _DATE, "date")
    ],
    "amount_claimed": [(_label("Amount Claimed", "Total sum claimed") + _AMOUNT, "amount")],
    "hospital_name": [(_HOSPITAL, "str")],
    "admission_date": [
        (_label("Admission Date", "Admitted", "Date of admission") + _DATE, "date"),
        (r"(?i)admitted on " + _DATE, "date"),
    ],
    "discharge_date": [
        (_label("Discharge Date", "Date of discharge") + _DATE, "date"),
        (r"(?i)discharged(?: on)?" + _SEP + _DATE, "date"),
    ],
    "diagnosis": [(_label("Final diagnosis", "Diagnosis at discharge") + _TEXT, "str")],
    "room_rate_per_day": [
        (_label("Room rate per day", "Room charges per day") + _AMOUNT, "amount")
    ],
    "bill_total": [(_label("TOTAL AMOUNT DUE", "Grand total") + _AMOUNT, "amount")],
    "vehicle_reg_no": [
        (
            _label("Vehicle Reg No", "Vehicle registration", "Vehicle involved", "Vehicle No", "Vehicle")
            + r"([A-Z]{1,3} [0-9OIl]{1,4})\b",
            "id",
        )
    ],
    "cause_of_loss": [
        (_label("Cause of Loss", "Description of loss", "Nature of incident", "Statement") + _TEXT, "str")
    ],
    "repair_total": [(_label("Estimated repair total", "Total quotation") + _AMOUNT, "amount")],
    "police_report_no": [
        (
            _label("Police Report No", "Police report reference", "Report No", "Report reference")
            + r"([A-Z]{2,}/[0-9OIl]{4}/[0-9OIl]{4,6})",
            "id",
        )
    ],
}  # fmt: skip


def _convert(kind: str, raw: str) -> str | None:
    if kind == "date":
        parsed = parse_date(raw)
        return parsed.isoformat() if parsed else None
    if kind == "amount":
        amount = parse_amount(raw)
        return f"{amount:.2f}" if amount is not None else None
    if kind == "id":
        return fix_ocr_identifier(raw)
    if kind == "ic":
        return fix_ocr_digits(raw)
    if kind == "product":
        return raw.upper()
    return raw.strip()


class FakeProvider:
    """Deterministic offline stand-in for an LLM.

    It answers only from the document text inside the prompt, using keyword
    scoring for classification and label patterns for extraction. It has no
    access to file names, labels or generator state.

    ``faults`` maps a 1-based call number to an injected failure, so tests can
    script error paths: {1: "malformed_json"} breaks the first call only.
    """

    name = "fake"

    def __init__(self, faults: dict[int, Fault] | None = None):
        self.faults = dict(faults or {})
        self.calls = 0

    def complete(self, prompt: str) -> str:
        self.calls += 1
        task = re.search(r"^TASK: (\w+)$", prompt, re.M)
        start = prompt.find(_DOC_START)
        end = prompt.rfind(_DOC_END)
        text = prompt[start + len(_DOC_START) : end] if 0 <= start < end else ""
        if task and task.group(1) == COVERAGE_TASK:
            return self._coverage_answer(prompt)
        if task and task.group(1) == "classify_document":
            answer = self._classify(text)
        else:
            doc_type = re.search(r"^DOC_TYPE: (\w+)$", prompt, re.M)
            answer = self._extract(DocType(doc_type.group(1)), text) if doc_type else {}

        fault = self.faults.get(self.calls)
        if fault == "malformed_json":
            return '{"doc_type": "claim_form", oops'
        if fault == "missing_field" and answer:
            answer.pop(next(iter(answer)))
        return json.dumps(answer)

    def _coverage_answer(self, prompt: str) -> str:
        start = prompt.find(_EVIDENCE_START)
        end = prompt.rfind(_EVIDENCE_END)
        evidence = json.loads(prompt[start + len(_EVIDENCE_START) : end])
        answer = assess_coverage(evidence["claim"], evidence["clauses"])
        fault = self.faults.get(self.calls)
        if fault == "provider_error":
            raise ProviderError("fake provider failure (injected)")
        if fault == "malformed_json":
            return '{"decision": "COVERED", oops'
        if fault == "missing_field":
            answer.pop("decision")
        cited = answer.get("cited_clauses", [])
        if fault == "wrong_quote" and cited:
            cited[0]["quote"] = cited[0]["quote"].rstrip(".") + " in all cases."
        if fault == "unretrieved_clause" and cited:
            cited[0]["clause_id"] = "MED-99.9"
        if fault == "drop_citations":
            answer["cited_clauses"] = []
        return json.dumps(answer)

    @staticmethod
    def _classify(text: str) -> dict:
        lowered = text.lower()
        scores = {
            doc_type: sum(weight for phrase, weight in signals if phrase in lowered)
            for doc_type, signals in _TYPE_SIGNALS.items()
        }
        best = max(scores, key=lambda t: scores[t])
        return {"doc_type": best.value if scores[best] > 0 else None}

    @staticmethod
    def _extract(doc_type: DocType, text: str) -> dict:
        result: dict[str, str | None] = {}
        for key in EXTRACTION_SCHEMAS[doc_type].model_fields:
            result[key] = None
            for pattern, kind in _PATTERNS[key]:
                match = re.search(pattern, text)
                if match:
                    result[key] = _convert(kind, match.group(1))
                    break
        return result


def _quote(clause: dict) -> str:
    """The rule sentence of a clause: the sentence after its title, word for word."""
    sentences = re.split(r"(?<=\.) ", clause["text"])
    return sentences[1] if len(sentences) > 1 else sentences[0]


def assess_coverage(claim: dict, clauses: list[dict]) -> dict:
    """Deterministic rule reading of the supplied facts and retrieved clauses.

    Each rule needs its clause among the retrieved evidence; when the deciding
    clause was not retrieved the fake answers REFER rather than cite from memory.
    Thresholds (waiting days, room limit) are read from the clause text.
    """

    def find(phrase: str) -> dict | None:
        return next((c for c in clauses if phrase in c["text"].lower()), None)

    def answer(decision: str, clause: dict | None, why: str) -> dict:
        if clause is None and decision != "COVERED":
            return {
                "decision": "REFER",
                "cited_clauses": [],
                "rationale": why + "; deciding clause not retrieved",
            }
        cited = [{"clause_id": clause["clause_id"], "quote": _quote(clause)}] if clause else []
        return {"decision": decision, "cited_clauses": cited, "rationale": why}

    product = claim.get("product")
    incident, start, end = (claim.get(k) for k in ("incident_date", "cert_start", "cert_end"))
    if incident and start and end and not (start <= incident <= end):
        return answer(
            "NOT_COVERED",
            find("period of cover"),
            "incident date is outside the certificate period",
        )
    text = " ".join(claim.get(k) or "" for k in ("diagnosis", "cause_of_loss")).lower()
    if product == "MEDICAL":
        if "cosmetic" in text:
            return answer(
                "NOT_COVERED", find("cosmetic treatment exclusion"), "treatment is cosmetic"
            )
        waiting = find("waiting period for illness")
        days = re.search(r"waiting period of (\d+) days", waiting["text"]) if waiting else None
        if days and incident and start and "accident" not in text:
            gap = (date.fromisoformat(incident) - date.fromisoformat(start)).days
            if gap < int(days.group(1)):
                return answer("NOT_COVERED", waiting, "illness began inside the waiting period")
        room = find("room and board sub-limit")
        limit = re.search(r"RM ([0-9.]+) per day", room["text"]) if room else None
        rate = claim.get("room_rate_per_day")
        if limit and rate and float(rate) > float(limit.group(1)):
            return answer(
                "PARTIALLY_COVERED", room, "room rate is above the daily room and board limit"
            )
        return answer(
            "COVERED",
            find("hospitalisation benefit"),
            "medically necessary admission inside the period",
        )
    if product == "MOTOR":
        if "racing" in text:
            return answer("NOT_COVERED", find("racing exclusion"), "vehicle was used for racing")
        if "unlicensed" in text:
            return answer(
                "NOT_COVERED", find("unlicensed driver exclusion"), "driver was unlicensed"
            )
        if any(w in text for w in ("collision", "hit ", "impact")):
            return answer(
                "PARTIALLY_COVERED",
                find("collision excess"),
                "collision damage, the excess applies",
            )
        return answer("COVERED", find("own damage"), "accidental own damage inside the period")
    return {"decision": "REFER", "cited_clauses": [], "rationale": "product unknown"}


# ---- Optional real provider


class OpenAICompatibleProvider:
    """Calls any OpenAI-compatible /chat/completions endpoint (Ollama, Groq, OpenRouter).

    Configured only through environment variables; see .env.example. Errors are
    raised as ProviderError and never replaced with fake answers.
    """

    name = "openai"

    def __init__(self, base_url: str, model: str, api_key: str | None, timeout_s: float):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.api_key = api_key
        self.timeout_s = timeout_s

    def complete(self, prompt: str) -> str:
        body = json.dumps(
            {
                "model": self.model,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": "You extract data and reply with JSON only."},
                    {"role": "user", "content": prompt},
                ],
            }
        ).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(self.url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                payload = json.load(response)
            content = payload["choices"][0]["message"]["content"]
        except Exception as exc:  # network, HTTP, decoding: reported by type only
            raise ProviderError(
                f"{self.name} provider call failed ({type(exc).__name__})"
            ) from None
        return _strip_code_fence(content)


def _strip_code_fence(content: str) -> str:
    match = re.fullmatch(r"\s*```(?:json)?\s*(.*?)\s*```\s*", content, re.S)
    return match.group(1) if match else content


def get_provider() -> LLMProvider:
    """Select the provider from CLAIMS_PROVIDER (default: fake)."""
    choice = os.environ.get("CLAIMS_PROVIDER", "fake").strip().lower()
    if choice == "fake":
        return FakeProvider()
    if choice == "openai":
        base_url = os.environ.get("CLAIMS_LLM_BASE_URL")
        model = os.environ.get("CLAIMS_LLM_MODEL")
        if not base_url or not model:
            raise ValueError(
                "CLAIMS_PROVIDER=openai needs CLAIMS_LLM_BASE_URL and CLAIMS_LLM_MODEL"
            )
        return OpenAICompatibleProvider(
            base_url,
            model,
            os.environ.get("CLAIMS_LLM_API_KEY") or None,
            float(os.environ.get("CLAIMS_LLM_TIMEOUT_S", "60")),
        )
    raise ValueError(f"unknown CLAIMS_PROVIDER {choice!r}; use 'fake' or 'openai'")
