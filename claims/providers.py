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
from typing import Literal, Protocol

from claims.normalize import fix_ocr_digits, fix_ocr_identifier, parse_amount, parse_date
from claims.schemas import EXTRACTION_SCHEMAS, DocType

Task = Literal["classify_document", "extract_fields"]
Fault = Literal["malformed_json", "missing_field"]

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
