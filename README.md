# Project Tabarru: claim packet intake agent

A Python package, `claims`, that takes a packet of OCR'd claim documents for a
synthetic Malaysian Takaful operator (motor and medical), classifies each
document, extracts the claim fields through an `LLMProvider`, validates them
against a typed schema, the certificate register and cross-document rules, and
stores the claim with a step-by-step trace. It has a CLI and a FastAPI API.

Status: **Task 1 is implemented and tested. Tasks 2 and 3 are not started**
(no coverage step, clause retrieval, adjuster gate, evaluation report or
trained classifier yet). See "What is next".

All data is synthetic and produced by `claims/generate.py`. Every document
contains the word SPECIMEN. Names, IC numbers (place code `00`, which is not
issued), certificates, hospitals and workshops are invented.

## Setup

Python 3.10+ is required; the repo pins 3.12 for `uv` in `.python-version`.

```bash
uv sync                     # creates .venv with runtime + dev dependencies
```

Without uv: `python -m venv .venv && . .venv/bin/activate && pip install fastapi pydantic uvicorn httpx pytest`.

Runtime dependencies are only `fastapi`, `pydantic` and `uvicorn`. Everything
else (SQLite, HTTP for the optional real provider, date parsing) is stdlib.

## Commands

```bash
# 1. Generate data (byte-identical for the same seed)
uv run python -m claims.generate --claims 40 --seed 7 --out data/

# 2. Process one packet; prints the same JSON as POST /v1/claims
uv run python -m claims.intake data/claims/CLM-0001/
uv run python -m claims.intake data/claims/CLM-0002/ --trace --db data/claims.db

# 3. Serve the API (reads CLAIMS_DB and CLAIMS_REGISTER, defaults under data/)
uv run uvicorn --factory claims.api:create_app --port 8000

# 4. Tests and checks
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

`data/` is git-ignored; run the generator first.

### Generated data

```
data/policies.jsonl              certificate register (certificate_no, participant_name,
                                 participant_ic, product, start_date, end_date, sum_covered)
data/claims/<claim_id>/*.txt     2 to 4 documents, opaque names such as 42c927b9.txt
data/labels.jsonl                claim_id, product, scenario, documents [{doc_id, doc_type,
                                 ocr_noise}], ground-truth fields, expected_reason_codes
```

Five document types with two layouts each (for example "CLAIM FORM" with
`Label: value` lines, and "NOTIFICATION OF CLAIM" with dotted leaders; a bill as
a list or as a `|` table; a discharge summary as fields or as prose). Every
document picks its date format (`03/04/2026`, `3-Apr-2026`, `2026-04-03`,
`3 April 2026`) and amount format (`RM 1,250.00`, `1250`, `MYR 1,250.00`) at
random. About 40% of documents (55 of 124 for seed 7) also get OCR noise: an
O/0 or I/1 swap inside numeric values, an extra scan header and footer, and,
for half of those, shuffled lines.

Claims cycle through scenarios every ten: six clean, and one each of
certificate not in register, missing document, bad date order and amount
mismatch. So seed 7 with 40 claims has clean medical and motor claims and 4
claims for each reason code. Some medical claims are road accidents and carry
a fourth document (a police report).

Labels are written only by the generator and read only by tests. The runtime
pipeline never imports `claims.generate` or opens `labels.jsonl` (a test
deletes both from reach and gets the same result).

### CLI behaviour

* `claim_id` is the folder name, `doc_id` is each file's stem. File names are
  never used to guess document type.
* Exit 0 on success, 1 if the folder has no `.txt` files, the register is
  missing, or a file name is not a valid id. With `--db`, a claim already stored
  exits 2 and nothing is overwritten.
* `--trace` adds the trace (same shape as `GET /v1/claims/{id}`).

### API

`POST /v1/claims`

```json
{"claim_id": "CLM-0001", "documents": [{"doc_id": "42c927b9", "text": "..."}]}
```

returns 201 and

```json
{"claim_id": "CLM-0001", "status": "intake_complete",
 "documents": [{"doc_id": "42c927b9", "doc_type": "claim_form", "status": "processed"}],
 "fields": {"certificate_no": "MED-2026-100011", "product": "MEDICAL",
            "participant_ic": "XXXXXX-XX-0791", "amount_claimed": "1570.00", "...": "..."},
 "reason_codes": []}
```

`GET /v1/claims/{claim_id}` returns the same object plus `trace`, a list of
`{step, doc_ids, status, duration_ms, provider_calls, error}`.

| Case | Response |
| --- | --- |
| Unknown claim_id on GET | 404 |
| claim_id already stored | 409, the stored claim is unchanged |
| Invalid body (missing keys, extra keys, 0 or more than 10 documents, duplicate doc_id, ids outside `[A-Za-z0-9_.-]`, empty or >100k char text) | 422 with `loc`, `type`, `msg` only. The default FastAPI 422 echoes the input, which could contain an IC, so it is replaced. |
| Model output is bad | Never an error: the document is marked failed and the claim flagged |

Amounts are serialised as strings with two decimals (`"1250.00"`) so no float
rounding happens anywhere; dates as `YYYY-MM-DD`.

## Architecture

```
            text documents
                  |
  +---------------v----------------+       +------------------------+
  | pipeline.run_intake            |       | providers.LLMProvider  |
  |  1 classify_documents  --------+------>|  FakeProvider (default)|
  |  2 extract_fields      --------+------>|  OpenAICompatible      |
  |  3 validate  (validation.py)   |       +------------------------+
  |  4 assemble_claim (mask IC)    |
  +---------------+----------------+
                  | ClaimDetail (fields, reason codes, trace)
       +----------+-----------+
       v                      v
  storage.ClaimStore      intake.py (CLI) / api.py (FastAPI)
  SQLite: claims, documents, trace
```

| Module | Responsibility |
| --- | --- |
| `schemas.py` | Pydantic models: per-document-type extraction schemas, `ClaimFields`, trace entry, API request and response |
| `normalize.py` | Date and amount parsing, OCR digit repair, IC masking and redaction |
| `providers.py` | `LLMProvider` protocol, prompt builder, `FakeProvider`, `OpenAICompatibleProvider`, `get_provider()` |
| `pipeline.py` | The four steps, retry, trace |
| `validation.py` | Register loading, field precedence, reason codes |
| `storage.py` | SQLite tables and queries |
| `api.py`, `intake.py` | HTTP and CLI entry points |
| `generate.py` | Synthetic data; not imported by any runtime module |

The pipeline is plain Python, not LangGraph: four functions called in order
are easy to read, test and trace, and Task 2's coverage step can be appended
as a fifth function.

### Step design, retry and trace

1. **classify_documents**: one provider call per document; the answer must
   validate as `{"doc_type": <one of five>}`.
2. **extract_fields**: one call per classified document with that type's
   schema. Every key must be present (a missing key is a schema error); a value
   may be `null` if the document does not show it. Dates must be ISO, amounts
   decimal, product `MOTOR` or `MEDICAL`.
3. **validate**: merge fields by precedence, then run the rules below. No
   provider calls.
4. **assemble_claim**: mask the IC, build `ClaimFields`, decide the status.

For each provider call: parse JSON, validate with the schema. On failure, call
again once with `Your previous answer was rejected: <error>` added to the
prompt. The error text is built only from JSON error positions and Pydantic
error locations and types (for example `doc_type: missing`), never
from input values, so it cannot carry document content back into the prompt,
the trace or logs. After the second failure the document is marked `failed`
and the pipeline continues. A provider that raises (network, HTTP) is treated
the same way and reported by exception type only; there is no fallback to fake
answers.

Every step appends one trace entry. `status` is `ok`, `partial` (some
documents failed), `failed` (all failed) or `skipped` (nothing to do, for
example extraction when no document was classified). `provider_calls` counts
retries, so a retry is visible as calls greater than documents. `error` lists
`doc_id: reason` for each failed document.

### Validation rules and status

| Code | Rule |
| --- | --- |
| `CERT_NOT_FOUND` | certificate_no missing or not in `policies.jsonl` |
| `MISSING_DOCUMENT` | MEDICAL needs claim_form, hospital_bill, discharge_summary; MOTOR needs claim_form, repair_estimate, police_report. A document counts as present once classified, even if its extraction failed. If the product cannot be determined this code is raised too. |
| `DATE_ORDER_INVALID` | discharge_date before admission_date, or admission_date before incident_date |
| `AMOUNT_MISMATCH` | claim form amount differs from a bill_total or repair_total by more than 1% of that total. Exactly 1% is not a mismatch (tested at 1010.00 vs 1010.01 on a 1000.00 bill). Every document's total is compared, not only the merged one. |
| `FIELDS_INCOMPLETE` | Addition beyond the four required codes: a required field for the product is still null after merging. It keeps a claim with missing data from being reported as `intake_complete`. |

Status: `failed` if no document was both classified and extracted;
`flagged` if there is any reason code or any failed document;
otherwise `intake_complete`. A partial packet still gets every field the
working documents provide.

Product is taken from the claim form, then from the register entry for the
certificate, then from the document mix (a bill or summary means MEDICAL, a
repair estimate means MOTOR).

### Field precedence

For each field, document types are tried in a fixed order and the first
non-null value wins; within one type, packet order decides.

| Field | Order |
| --- | --- |
| certificate_no, product, participant_ic, amount_claimed | claim_form |
| participant_name | claim_form, hospital_bill, discharge_summary |
| incident_date | claim_form, police_report |
| admission_date, discharge_date | discharge_summary, hospital_bill |
| diagnosis | discharge_summary |
| hospital_name | hospital_bill, discharge_summary, claim_form |
| room_rate_per_day, bill_total | hospital_bill |
| vehicle_reg_no | claim_form, police_report, repair_estimate |
| cause_of_loss, police_report_no | police_report, claim_form |
| repair_total | repair_estimate |

The rule: what the participant asserts comes from the claim form; what a
hospital, workshop or police station certifies comes from their document.
Conflicts are only detected for amounts (AMOUNT_MISMATCH); a conflicting
date or name between documents is resolved by precedence without a flag.

### Dates and money

Numeric dates are read day first (`03/04/2026` is 3 April 2026), the Malaysian
convention; a four digit first part is read as ISO. Impossible dates
(`31/02/2026`) become null rather than being swapped. Month names are matched
on their first three letters. Amounts drop `RM`/`MYR` and thousands
separators and are held as `Decimal` quantised to 0.01 (half up) from parsing
to storage. OCR swaps O/0 and I/1 are repaired only inside segments that
already contain a digit, so prefixes like `MED` or `PR` are untouched.

### FakeProvider

The default provider, used by every test. It reads only the document text
inside the prompt:

* classification: weighted keyword phrases per type (titles weigh more); no
  match returns `{"doc_type": null}`, which fails validation like a confused
  model would.
* extraction: label patterns for every field across both layouts, followed by
  the same normalisers as above.

Fault injection is by call number: `FakeProvider(faults={1: "malformed_json",
5: "missing_field"})` breaks call 1 with invalid JSON and drops the first key
from call 5's answer. Calls are numbered in pipeline order (classification of
every document, then extraction).

Over the 40 generated claims it reproduces every document type and every
expected reason code (asserted in `test_every_generated_claim_matches_its_labels`).
That is expected, not an achievement: I wrote both the templates and the
patterns. A real model on unseen layouts would not score like this.

### IC masking

* `assemble_claim` masks `participant_ic` to `XXXXXX-XX-1234`;
  `ClaimFields` rejects any unmasked value, so a full IC cannot reach a
  response or the database through the claim record.
* Stored document text has every IC-shaped number redacted (including OCR
  variants like `9OO1O1-00-I234`).
* Trace errors are built without input values and then also redacted.
* Logs contain claim id, status, reason codes and failed doc ids only.
* The custom 422 handler drops `input`.

Tests check the response bodies, trace, logs and raw SQLite bytes for the full
IC on both a successful path and a misbehaving provider that echoes the IC
inside invalid answers.

### Switching to a real provider

Set the variables from `.env.example`:

```bash
export CLAIMS_PROVIDER=openai
export CLAIMS_LLM_BASE_URL=http://localhost:11434/v1     # Ollama; or Groq / OpenRouter base URL
export CLAIMS_LLM_MODEL=llama3.1
export CLAIMS_LLM_API_KEY=...                            # not needed for Ollama
uv run python -m claims.intake data/claims/CLM-0001/
```

The adapter posts to `<base>/chat/completions` with temperature 0 and strips
a Markdown code fence from the answer. It goes through the same schema
validation and retry as the fake. A failing endpoint fails documents; it
never falls back to fake results. I have not run it against a live model in
this round, so its accuracy is unmeasured.

## Tests

`uv run pytest` runs 74 tests in under a second. An autouse fixture removes
every `CLAIMS_*` variable and patches `socket` so any network connection
raises; the real adapter test relies on that to prove it fails cleanly.
Each test that touches SQLite uses its own `tmp_path` database, and the
dataset is generated once per session into a temp folder.

| "Done means" item | Test |
| --- | --- |
| Generator reproducible | `test_generate.py::test_same_seed_gives_byte_identical_output` |
| Complete medical and motor claims, every doc_type | `test_pipeline.py::test_complete_packet[MEDICAL/MOTOR]` |
| Malformed JSON then successful retry | `test_malformed_json_then_successful_retry` |
| Malformed twice: failed document, flagged claim, no exception | `test_malformed_twice_fails_document_and_flags_claim` |
| Missing required field | `test_missing_required_field_is_retried`, `test_missing_field_twice_fails_extraction_only` |
| Each reason code | `test_reason_code_scenarios_are_flagged[...]` (generated packets) and hand-built rule tests, including the 1% boundary |
| All documents failing | `test_all_documents_failing_gives_failed_claim` |
| POST then GET, shape, four trace steps, masked IC | `test_api.py::test_post_then_get` |
| No full IC in logs, trace, errors, DB | `test_privacy.py` |
| Provider ignores labels, generator and file names | `test_fake_works_without_labels_or_generator`, `test_fake_ignores_file_names_and_packet_order` |

No test needs a marker: nothing uses live services.

## What is next

* Task 2: certificate wording generator, BM25 clause retrieval, coverage step
  with citation guard, payable computed in code, adjuster queue and decision
  endpoints with an append-only audit table, `claims.eval`.
* Task 3: trained TF-IDF + logistic regression classifier with held-out
  layouts, `CLAIMS_CLASSIFIER=trained|llm|hybrid` selecting the
  `classify_documents` implementation (the step already isolates
  classification behind one function).
* Cross-document conflict flags for names and dates, not only amounts.
* Check the claim form's product against the register entry's product
  (today a MEDICAL claim on a MOTOR certificate is not flagged).
* Run the OpenAI-compatible adapter against a local Ollama model and record
  its accuracy against `labels.jsonl`.

The adjuster UI (a stretch goal) is the only visual work in scope; nothing
visual is built yet.
