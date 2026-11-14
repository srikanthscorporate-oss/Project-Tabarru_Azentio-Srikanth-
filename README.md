# Project Tabarru: claim intake and cited coverage recommendation

A Python package, `claims`, that takes a packet of OCR'd claim documents for a
synthetic Malaysian Takaful operator (motor and medical), classifies each
document, extracts the claim fields through an `LLMProvider`, validates them
against a typed schema, the certificate register and cross-document rules, and
stores the claim with a step-by-step trace. It then retrieves certificate
clauses, asks the provider for a cited coverage recommendation, computes the
payable in code and holds every claim for a human adjuster. It has a CLI, a
FastAPI API and a small HTML adjuster view.

| Part | Status |
| --- | --- |
| Task 1: intake agent (generator, four traced steps, validation, SQLite, API, CLI) | Implemented and tested |
| Task 2: retrieval, cited recommendation, payable, adjuster gate, audit, evaluation | Implemented and tested |
| Task 3: trained document classifier, llm/trained/hybrid modes, comparison | Implemented and tested |
| Stretch: adjuster view at `/` (trace, cited clauses, decisions, audit) | Implemented, browser-checked |
| Stretch: Prometheus `/metrics` | Implemented and tested |
| Stretch: Dockerfile and `docker compose up` | Built and run locally (see "Docker") |
| Stretch: coverage evaluation with a real model vs fake | Tooling done; measured result in "Real-model comparison" |

All data is synthetic and produced by `claims/generate.py`. Every document
contains the word SPECIMEN. Names, IC numbers (place code `00`, which is not
issued), certificates, hospitals and workshops are invented.

## Setup

Python 3.10+ is required; the repo pins 3.12 for `uv` in `.python-version`.

```bash
uv sync                     # creates .venv with runtime + dev dependencies
```

Without uv: `python -m venv .venv && . .venv/bin/activate && pip install fastapi pydantic uvicorn httpx pytest`.

Runtime dependencies are `fastapi`, `pydantic`, `uvicorn`, plus
`scikit-learn` and `joblib` for the Task 3 classifier. Everything else
(SQLite, HTTP for the optional real provider, date parsing, metrics) is stdlib.
The adjuster view is three static files with no build step and no CDN.

## Quick start

```bash
./start.sh     # installs deps, generates data if missing, seeds data/demo.db, serves :8000
               # open http://localhost:8000/ (adjuster view), /docs (API), /metrics
./stop.sh      # stops that server and, if running, the docker compose stack; data is kept
```

`PORT=9000 ./start.sh` changes the port. `start.sh` refuses to start twice;
it keeps the server pid in `data/.tabarru.pid` for `stop.sh`. The view and the
API are one FastAPI process, so there is no separate frontend server.

## Commands

```bash
# 1. Generate data (byte-identical for the same seed)
uv run python -m claims.generate --claims 40 --seed 7 --out data/

# 2. Process one packet; prints the same JSON as POST /v1/claims
uv run python -m claims.intake data/claims/CLM-0001/
uv run python -m claims.intake data/claims/CLM-0002/ --trace --db data/claims.db

# 3. Serve the API (reads CLAIMS_DB and CLAIMS_REGISTER, defaults under data/)
uv run uvicorn --factory claims.api:create_app --port 8000

# 4. Score coverage on all 40 claims (offline, about 0.1 s)
uv run python -m claims.eval --provider fake --out reports/

# 5. Adjuster view on a demo database (see "Stretch goals")
uv run python -m claims.seed_demo --data data/ --db data/demo.db
CLAIMS_DB=data/demo.db uv run uvicorn --factory claims.api:create_app --port 8000
# open http://localhost:8000/

# 6. Tests and checks
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
random. Exactly ceil(40%) of documents (50 of 124 for seed 7) also get OCR
noise, picked with a seeded sample so even a two-claim run meets the 25%
minimum: an
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
  missing, or the folder name, a file name or a file's text fails the same
  `ClaimRequest` validation as `POST /v1/claims` (so the CLI and API accept
  the same packets and print the same JSON, which a test asserts). With `--db`, a claim already stored
  exits 2 and nothing is overwritten.
* `--trace` adds the trace (same shape as `GET /v1/claims/{id}`).

### API

`POST /v1/claims`

```json
{"claim_id": "CLM-0001", "documents": [{"doc_id": "42c927b9", "text": "..."}]}
```

returns 201 and (abridged, from the current seed 7 data)

```json
{"claim_id": "CLM-0001", "status": "pending_adjuster", "intake_status": "intake_complete",
 "documents": [{"doc_id": "34893498", "doc_type": "claim_form", "status": "processed"}, "..."],
 "fields": {"certificate_no": "MED-2026-...", "product": "MEDICAL",
            "participant_ic": "XXXXXX-XX-....", "amount_claimed": "1320.00", "...": "..."},
 "reason_codes": [],
 "recommendation": {"decision": "COVERED", "payable_amount": "1320.00", "reason_codes": [],
                    "cited_clauses": [{"clause_id": "MED-2.1", "quote": "..."}], "rationale": "..."}}
```

`status` is the workflow status; the Task 1 intake result is in
`intake_status`. `GET /v1/claims/{claim_id}` returns the same object plus
`trace` (`{step, doc_ids, status, duration_ms, provider_calls, error,
clause_ids}`, five steps: the four Task 1 steps and `coverage`) and
`adjuster_decision`.

| Case | Response |
| --- | --- |
| Unknown claim_id on GET | 404 `claim not found` (the id is not echoed) |
| claim_id already stored | 409, the stored claim is unchanged |
| Invalid body (missing keys, extra keys, 0 or more than 10 documents, duplicate doc_id, ids outside `[A-Za-z0-9_.-]` or containing an IC-shaped number, empty or >100k char text) | 422 with `loc`, `type`, `msg` only. The default FastAPI 422 echoes the input, which could contain an IC, so it is replaced. String parts of `loc` and `msg` go through IC redaction, because an unexpected key name is user input. |
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
  |  5 coverage (coverage.py)  ----+------> retrieval.py (BM25 clauses)
  +---------------+----------------+
                  | ClaimDetail (fields, reason codes, trace)
       +----------+-----------+
       v                      v
  storage.ClaimStore      intake.py (CLI) / api.py (FastAPI) -> web/ (adjuster view)
  SQLite: claims, documents,                      metrics.py (/metrics)
          trace, audit
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
* Every other text field (participant_name, diagnosis, cause_of_loss and so
  on) is passed through the same redaction in `assemble_claim`, in case a
  model copies an IC into it.
* Retry feedback and trace errors are built from error locations and types
  only, and the locations are redacted before they enter the prompt, because a
  model can invent a key such as `"900101-00-1234"`.
* `claim_id` and `doc_id` are rejected (422 in the API, exit 1 in the CLI) if
  they contain an IC-shaped number, since ids are echoed in responses, the
  trace and logs. This is a deliberate rejection; ordinary ids such as
  `CLM-0001` or `42c927b9` are unaffected. 404 and 409 messages do not echo the
  id.
* Logs contain claim id, status, reason codes and failed doc ids only. The app
  adds a filter to the `uvicorn.access` logger that masks IC-shaped text in the
  request path.
* The custom 422 handler drops `input` and redacts `loc` and `msg`.

All of this uses one function, `normalize.redact_ics`, so there is a single
definition of what an IC looks like.

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
never falls back to fake results. `python -m claims.models` (see "Real-model
comparison") lists local Ollama and Groq models and runs the comparison.

## Tests

`uv run pytest` runs 214 tests in about two seconds. An autouse fixture removes
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
| No full IC in logs, trace, errors, DB | `test_privacy.py` (including an IC as an unexpected JSON key, IC-shaped ids, an IC in free-text model fields, retry feedback and the access log) |
| At least 25% OCR noise, even for small runs | `test_generate.py::test_noise_share_is_guaranteed_for_small_runs` |
| CLI prints the same JSON as the API | `test_cli_and_providers.py::test_cli_output_equals_api_post_response` |
| Provider ignores labels, generator and file names | `test_fake_works_without_labels_or_generator`, `test_fake_ignores_file_names_and_packet_order` |

No test needs a marker: nothing uses live services.

## Task 2: cited coverage recommendation, adjuster gate and evaluation

### Flow

```
POST /v1/claims
  run_intake   classify_documents -> extract_fields -> validate -> assemble_claim   (Task 1, unchanged)
  run_coverage coverage   (retrieve clauses -> provider decision + citation guard -> payable in code)
  store        claim, documents, trace, recommendation, audit row (one transaction)
  status       pending_adjuster (always)
POST /v1/claims/{id}/adjuster-decision
  status       decided, adjuster decision stored next to the untouched recommendation, audit row
```

Code: `claims/retrieval.py` (BM25), `claims/coverage.py` (step, guardrails,
payable), `claims/providers.py` (`build_coverage_prompt`, `assess_coverage`
for the fake), `claims/storage.py` (audit, migration, `decide`),
`claims/api.py` (endpoints), `claims/eval.py` (metrics). The generator writes
the wording from `claims/wording.py`; the runtime only reads the generated
files under `data/`.

### Generated certificates, schedules and labels

`python -m claims.generate --claims 40 --seed 7 --out data/` now also writes:

* `data/certificates/MEDICAL.md` (14 clauses, `MED-1.1` to `MED-6.2`) and
  `data/certificates/MOTOR.md` (13 clauses, `MTR-1.1` to `MTR-6.1`). One clause
  per line: `<ID> <Title>. <rule sentence>...`. Each file says SPECIMEN.
* `data/schedules/MEDICAL.json`: `room_board_per_day` 250.00,
  `annual_limit` 100000.00, `waiting_period_days` 30.
  `data/schedules/MOTOR.json`: `excess` 500.00. The certificate limit
  (`sum_covered`) is per certificate in `policies.jsonl`. A test checks that
  every schedule value appears in the wording.
* `labels.jsonl` gains `coverage_scenario`, `expected_decision`,
  `expected_clause_ids` and `expected_payable`. These are computed in
  `expected_coverage()` in the generator from its own facts, not by calling
  `claims.coverage`.

The six "clean" slots of each ten-claim cycle become: covered (2), partial
(sub-limit for medical, collision excess for motor), incident outside the
certificate period, illness inside the 30-day waiting period (always
medical), excluded cause (cosmetic treatment; racing or unlicensed driver).
The four anomaly slots keep their Task 1 reason codes and expect REFER. Per 40
claims: covered 8, partial 4, outside period 4, waiting period 4, excluded 4,
missing document 4 (plus 12 other REFER). Task 1 scenarios, OCR noise, opaque
file names and byte-identical output per seed are unchanged.

### Retrieval

`ClauseIndex.retrieve(query, product, k)` returns `Clause(clause_id, text,
score)`. Only the claim's product's clauses are scored, so a motor query can
never return a medical clause. BM25 with k1 = 1.5, b = 0.75, lowercase
alphanumeric tokens, a short stopword list, no stemming, `k = 6`. Ties sort by
clause id.

Query (`build_query`), built from extracted fields only:

1. fixed product terms: MEDICAL `period of cover start date hospitalisation
   benefit room and board per day`; MOTOR `period of cover start date own
   damage repair excess`;
2. the extracted `diagnosis` and `cause_of_loss` text;
3. `waiting period illness` when the product is MEDICAL and the extracted
   incident date is less than 60 days after the register's certificate start.

Labels, scenarios, file names and claim ids are never used.

### Coverage step and guardrails

The provider gets the claim fields (without the IC), the certificate start and
end dates from the register, and the top-k clauses, as one JSON block in the
prompt. It must return `{"decision", "cited_clauses": [{"clause_id",
"quote"}], "rationale"}`; extra keys (for example a `payable_amount`) are a
schema error.

Checked in code (`check_citations`):

* every `clause_id` is one of the retrieved clauses;
* every quote is a non-empty exact substring of that clause's text, and holds
  no IC number;
* NOT_COVERED and PARTIALLY_COVERED cite at least one clause.

A broken rule is retried once. The retry prompt names the rule and the
citation number only (for example `citation 1: quote is not an exact
substring of its clause`), never the model's text. If the second answer also
fails: REFER with `CITATION_INVALID`. Invalid JSON or schema twice gives REFER
with `COVERAGE_OUTPUT_INVALID`; a provider exception twice gives REFER with
`PROVIDER_ERROR`. A failing real provider is never replaced by the fake. Each
attempt's error is kept in the `coverage` trace entry, which also lists the
retrieved `clause_ids`.

Bypass (REFER, no provider call, trace entry `coverage` with status
`skipped` and `bypassed: <codes>`): intake failed (`INTAKE_FAILED`), or the
claim carries any intake reason code. `CERT_NOT_FOUND` and `MISSING_DOCUMENT`
are required by the assignment. I also bypass `DATE_ORDER_INVALID`,
`AMOUNT_MISMATCH` and `FIELDS_INCOMPLETE`, because the dates and amounts the
decision and payable depend on are in doubt; and a packet with a failed
document (`INTAKE_INCOMPLETE`). The intake `status` is kept as
`intake_status`, and the top-level `reason_codes` stay the intake codes; the
recommendation's `reason_codes` are the intake codes plus any coverage code.

FakeProvider coverage (`assess_coverage`) applies the wording's rules to the
supplied facts, in this order: period of cover, exclusions (cosmetic;
racing; unlicensed), waiting period (days read from the clause text, not for
accident diagnoses), room sub-limit (limit read from the clause text) or
collision excess, otherwise covered. It cites the clause whose title matches
the rule, quoting the rule sentence word for word. If that clause was not
retrieved it answers REFER rather than cite from memory. Faults for tests:
`wrong_quote`, `unretrieved_clause`, `drop_citations`, `provider_error`,
`malformed_json`, `missing_field`.

### Payable (code only)

`calculate_payable` uses `Decimal` throughout. It runs only for COVERED and
PARTIALLY_COVERED; NOT_COVERED and REFER are always 0.00.

```
MEDICAL  days         = max(discharge_date - admission_date in days, 1)         (MED-4.3)
         room_excess  = max(room_rate_per_day - room_board_per_day, 0) * days   (MED-4.1)
         payable      = min(bill_total - room_excess, annual_limit, sum_covered, amount_claimed)
MOTOR    excess       = 500.00 when cause_of_loss names a collision or impact
                        (collision, collided, impact, hit as whole words), else 0 (MTR-5.1, MTR-5.2)
         payable      = min(repair_total - excess, sum_covered, amount_claimed)
then     payable      = max(payable, 0.00), rounded half up to cents
```

Deductions come first, caps second. Days count nights, so a same-day
discharge is one day. If an input is missing (for example no room rate), the
claim is REFER with `CALC_INPUT_MISSING`; nothing is guessed.

Hand-checked tests (`tests/test_coverage.py`): 3 days at RM 400 with a
RM 3,000.00 bill pays 3000.00 - 150 x 3 = **2550.00**; a RM 2,345.60 collision
repair pays 2345.60 - 500.00 = **1845.60**; the flood case pays the full
2345.60; theft, fire and "street" causes carry no excess while "hit a tree" does; boundaries at the annual limit and sum covered; a repair below the
excess pays 0.00.

### Workflow and adjuster endpoints

Statuses: intake result in `intake_status` (`intake_complete`, `flagged`,
`failed`); workflow `status` is `pending_adjuster` after POST, `decided` after
the adjuster acts. Nothing is settled automatically, including REFER.

```bash
curl -s localhost:8000/v1/adjuster/queue
curl -s -X POST localhost:8000/v1/claims/CLM-0002/adjuster-decision \
  -H 'content-type: application/json' \
  -d '{"adjuster": "adj.lim", "action": "approve"}'
curl -s -X POST localhost:8000/v1/claims/CLM-0001/adjuster-decision \
  -H 'content-type: application/json' \
  -d '{"adjuster": "adj.lim", "action": "override", "final_decision": "PARTIALLY_COVERED",
       "final_payable": "1200.00", "reason": "pharmacy item not medically necessary"}'
```

* `approve` may omit `final_decision` and `final_payable`; if sent they must
  equal the recommendation, otherwise 422 (no silent override).
* `override` needs a non-blank `reason`, `final_decision` and `final_payable`
  (422 otherwise).
* NOT_COVERED and REFER need `final_payable` 0.00. COVERED needs more than
  0.00.
* Zero-payable policy: the pipeline can recommend PARTIALLY_COVERED with
  0.00 (an excess that absorbs the whole repair cost). Approving that stored
  recommendation is allowed. An adjuster override to PARTIALLY_COVERED needs
  more than 0.00; an adjuster who wants a zero outcome overrides to
  NOT_COVERED. Tests: `test_zero_payable_partial_recommendation_can_be_approved`,
  `test_override_to_zero_payable_partial_is_422`.
* The adjuster view always sends the decision and payable it showed in the
  confirmation, so a stale view gets 422 instead of recording something else.
* Unknown claim 404; a claim that is not `pending_adjuster` (already decided,
  or a Task 1 claim never assessed) 409.
* `GET /v1/claims/{id}` returns `recommendation` (original, never changed)
  and `adjuster_decision` separately.

`ClaimStore.decide` runs `BEGIN IMMEDIATE`, then
`UPDATE ... WHERE status = 'pending_adjuster'`, then the audit insert, then
`COMMIT`. The conditional update means only one of two concurrent requests
can match; a test runs 8 threads and gets exactly one success.

### Audit

Table `audit(id, timestamp, actor, claim_id, event, decision, payable,
clause_ids, reason)`. One row per stored recommendation (actor
`coverage-agent`, written in the same transaction as the claim) and one per
successful adjuster decision (actor = adjuster). Triggers abort any UPDATE or
DELETE on the table. Rejected requests (422, 404, 409) write nothing.

A Task 1 database is migrated on open with `ALTER TABLE ... ADD COLUMN`
(`intake_status`, `recommendation`, `adjuster_decision`, trace `clause_ids`)
and the new `audit` table. Existing rows are untouched; old claims read back
with `recommendation: null`. Duplicate `claim_id` still returns 409.

### Privacy additions

The provider never receives the IC. Rationale, adjuster `reason` and adjuster
name pass through `redact_ics` (the name is rejected with 422 if it is an IC).
A quote holding an IC fails the citation check instead of being edited, so a
stored quote is always an exact substring. Provider errors are reported by
type only, and the 422 handler from Task 1 covers the new endpoint. A test
dumps the whole SQLite file after an override whose reason contains the IC
and finds no full IC.

### Evaluation

`python -m claims.eval --provider fake --out reports/` runs `assess_claim`
(intake plus coverage) on every claim in memory, with no database, and writes
`reports/coverage_eval.json`. Labels are read only to score. Measured on seed
7, 40 claims, k = 6 (two consecutive runs give the same file; about 0.1 s):

| metric | value |
|---|---|
| decision_accuracy | 0.975 |
| payable_exact_match | 1.0 |
| retrieval_recall_at_k | 0.9583 |
| citation_precision | 1.0 |
| citation_recall | 0.9583 |
| citation_invalid_rate | 0.0 |
| attempted_citation_invalid_rate | 0.0 |

| expected \ predicted | COV | PART | NOT | REF |
|---|---|---|---|---|
| COV | 8 | 0 | 0 | 0 |
| PART | 0 | 4 | 0 | 0 |
| NOT | 0 | 0 | 11 | 1 |
| REF | 0 | 0 | 0 | 16 |

Definitions:

* `decision_accuracy`, `payable_exact_match`: over all 40 claims; payable
  matches within MYR 0.01.
* `retrieval_recall_at_k`: expected clause ids found in the top-k / all
  expected clause ids, over the 24 claims that have expected clauses (REFER
  claims expect none). A bypassed claim retrieves nothing, so its expected ids
  count as missed.
* `citation_precision`: final cited ids that are expected / all final cited
  ids (null if nothing is cited). `citation_recall`: expected ids finally
  cited / all expected ids.
* `citation_invalid_rate`: claims whose final recommendation has
  `CITATION_INVALID` / claims where the coverage provider was called (24;
  bypassed claims are excluded). `attempted_citation_invalid_rate` also counts
  claims fixed on the retry.

Error sources seen:

1. **Retrieval miss on short exclusion text (the only error).** CLM-0038,
   diagnosis "Elective cosmetic liposuction", expected NOT_COVERED on MED-5.1.
   The fixed base terms (period, hospitalisation, room and board) outscore
   the single word "cosmetic", so MED-5.1 is not in the top 6. The fake then
   answers REFER instead of citing an unretrieved clause. This one miss is the
   whole gap in decision accuracy, retrieval recall and citation recall.
2. Nothing else failed. That is a property of this synthetic data, not
   evidence of quality: the same author wrote the wording, the generator and
   the fake's rules, the diagnoses and causes come from short fixed lists, and
   OCR noise does not touch the free-text fields the rules read. A real model
   reading real wording would make quote, exclusion and partial-payment
   mistakes that this set cannot show. `citation_invalid_rate` is 0 because
   the fake copies sentences from the clause text; the guard itself is tested
   with injected faults.

## Task 3: trained document classifier and comparison

### Commands

```bash
uv run python -m claims.train_classifier                 # writes models/doc_classifier.joblib (about 0.3 s)
uv run python -m claims.compare_classifiers --out reports/  # writes reports/classifier_comparison.json (about 0.2 s)
CLAIMS_CLASSIFIER=hybrid uv run python -m claims.intake data/claims/CLM-0001/ --trace
```

Measured on this laptop: training 0.8 s and comparison 0.7 s wall clock,
including interpreter start (training itself 0.26 s).

### Data and template separation

* The generator now has three layouts per document type. Template ids
  (`claim_form.t0` ... `police_report.t2`) are in the generator metadata:
  every row of `classifier_dataset` and every document in `labels.jsonl`
  carries `template_id`. Separation is tested on that field, never on file
  names.
* Claim packets still use layouts 0 and 1 only, so the Task 1 and Task 2
  documents for a given seed are byte-identical to before (`labels.jsonl`
  only gains `template_id`). Layout 2 is used only for the classifier test set.
* Training: seed 1301, layouts 0 and 1, 200 documents per class (1000).
  Test: seed 2602, layout 2 only, 100 per class (500). Every held-out
  template (`*.t2`) is absent from training. Training fails if any template
  or any exact text appears in both sets, or if a class has fewer than 150
  unique documents. Product (medical/motor claim form) varies independently
  of layout. OCR noise (40%) and date and amount styles are the same as in the packets.
* The test set is used only by `compare_classifiers`, never for settings or
  the threshold.

### Model

scikit-learn `Pipeline`: `TfidfVectorizer(word 1-2 grams, lowercase,
min_df=2, sublinear_tf)` then `LogisticRegression(C=1.0, max_iter=1000,
random_state=0)`. Defaults, no search. The input is document text only; no
doc type, template, claim id, file name or scenario. TF-IDF vocabulary and
IDF are fitted inside the pipeline on training text only (a test checks that
words found only in the held-out layout are not in the vocabulary).
`predict` returns the type and its probability, read through
`pipeline.classes_`.

The artifact `models/doc_classifier.joblib` (gitignored, rebuilt by the
command above) holds the pipeline plus metadata: classes, seeds, train and
held-out template ids, settings, threshold and rule, Python, scikit-learn and
joblib versions. It is a pickle: load only artifacts you built. No endpoint
accepts a model file.

### Runtime modes

| Variable | Default | Meaning |
|---|---|---|
| `CLAIMS_CLASSIFIER` | `llm` | `llm`, `trained` or `hybrid` |
| `CLAIMS_CLASSIFIER_MODEL` | `models/doc_classifier.joblib` | read only in trained and hybrid |
| `CLAIMS_HYBRID_THRESHOLD` | threshold stored in the model | optional override, in (0, 1] |

* `llm`: unchanged provider path (schema validation, one retry, failure).
* `trained`: the model types every document, zero classification provider calls.
* `hybrid`: the model's answer is kept when its top probability is `>=` the
  threshold (equal counts as accepted); below it the document goes through
  the same `call_with_retry` path as `llm`, so retries and failures behave the same.
* Only `classify_documents` changes. Extraction and coverage always call
  `CLAIMS_PROVIDER`. The step trace is unchanged: `provider_calls` on
  `classify_documents` counts real classification calls including retries,
  so trained shows 0 and hybrid shows only fallback calls.
* The model loads once, when the API app or CLI starts, and only for trained
  or hybrid. An unknown mode, a missing or invalid model, or a bad threshold
  raises `ClassifierConfigError` with the fix (API startup fails, CLI exits
  1). It never switches mode silently and never downloads anything.

### Hybrid threshold

0.40, chosen from training layouts only: fit on layout 0, predict layout 1,
and the reverse. The rule is the smallest 0.05 step above the most confident
wrong validation prediction, with a floor of 0.40 (twice the 0.20 probability
of a uniform guess over five classes). The first version had no floor; after
the product/layout fix there were no validation mistakes, the rule gave
0.05 and hybrid would never have fallen back, so the floor was added. These
validation numbers, not test numbers, drove that change. Logistic regression
on TF-IDF gives low top probabilities here (median about 0.57 on an unseen
layout) even when correct. The probabilities are not calibrated and I did not
measure calibration: a high confidence does not mean the answer is right.

### Results (held-out layout, 500 documents)

From `reports/classifier_comparison.json`:

| Option | Accuracy | Macro-F1 | Failed | LLM calls / 1k docs | Est. cost USD / 1k claims | Measured ms / doc (local) | Est. hosted LLM ms / doc |
|---|---|---|---|---|---|---|---|
| trained | 1.0000 | 1.0000 | 0 | 0.0 | 0.0000 | 0.18 | 0 |
| llm (FakeProvider) | 0.8000 | 0.8000 | 100 | 1200.0 | 0.9207 | 0.011 | 960 |
| hybrid | 0.9020 | 0.9351 | 49 | 220.0 | 0.1688 | 0.18 | 176 |

* FakeProvider has no keywords for the new discharge layout ("ward release
  note", "condition treated"), so all 100 fail after one retry (1200 calls
  per 1000 documents: 1 per document plus a retry for the 100 failures).
  Hybrid sends low-confidence documents to the same fake: 51 of the
  discharge summaries were accepted by the model and 49 fell back and
  failed, although the model had them right.
* Failed documents stay in every denominator and appear in a `failed`
  column of the confusion matrix (labels in `DocType` order).
* Accuracy, F1, confusion matrices and call counts repeat exactly between
  runs (checked twice, and by a test). Timings vary.
* Measured latency is `time.perf_counter_ns` around `classify_documents` for
  one document: model prediction, and for fallbacks the prompt, the
  in-process fake, JSON parse, validation and retry. Model load (1.7 ms) is
  reported separately. The fake's latency says nothing about a hosted LLM.
* Estimates (hypothetical, not verified prices, no real provider was
  called): 3.1 documents per claim (124/40 in the Task 1 data), 450 input
  and 15 output tokens per classification call, USD 0.50 / 1.50 per million
  input / output tokens, 800 ms per call, calls sequential. Cost is the
  classification step only, not extraction or coverage, and uses the observed
  call rate including retries.

### Recommendation

On this data the trained classifier alone is best: correct on every held-out
document, no LLM calls, well under a millisecond. Hybrid is only as good as
its fallback; here the fallback is the keyword fake, which is weaker than the
model on the new layout, so hybrid lost 49 documents the model had right.
For production I would run `trained`, or `hybrid` with a real LLM
fallback once that LLM is measured on labelled documents to beat the model
on low-confidence cases, and would keep the trained label (flagged) when the
fallback fails. Either way, classification cost drops from about one call
per document to the fallback rate.

### Monitoring and retraining

Track on a labelled sample: accuracy and macro-F1, per-class errors and the
confusion matrix, the confidence distribution and hybrid fallback rate (a
rising fallback rate is the first sign of a new layout), failure rate of the
fallback, OCR noise level and new layouts or senders, and latency. Retrain
when a new layout appears or per-class recall drops.

### Limitations

* Synthetic data from one author: three layouts per type, short fixed word
  lists. 100% on the held-out layout shows the model generalises across
  these layouts, not that it will on real scans.
* The LLM row is the keyword FakeProvider, not a real model; real-model
  accuracy, latency and cost were not measured.
* Confidence is uncalibrated; the threshold rests on two small validation folds.
* The trace does not record per-document confidence or whether the model or
  the fallback answered; only `provider_calls` shows it.
* sklearn is imported in every mode (the model file is loaded only when needed).

## Stretch goals

### Adjuster view

A static HTML/CSS/JavaScript page in `claims/web/`, served by the same FastAPI
app at `/` (assets under `/static/`). `/docs` still works. No build step, no
Node, no CDN, no remote fonts: it works offline. It only calls the API below;
every claim, amount, citation, trace entry and decision shown comes from it.

Local start:

```bash
uv run python -m claims.generate --claims 40 --seed 7 --out data/   # if data/ is empty
uv run python -m claims.seed_demo --data data/ --db data/demo.db    # repeatable
CLAIMS_DB=data/demo.db uv run uvicorn --factory claims.api:create_app --port 8000
# open http://localhost:8000/
```

Workflow: pick a claim in the queue, read facts and documents, open the cited
clauses and the trace, then approve or override.

* Queue (`GET /v1/adjuster/queue`): claim id, product, recommendation, payable,
  reason codes. Search by claim id, filter by product and recommendation,
  refresh. The counts line describes the loaded queue only. REFER is shown as
  "Needs review", never as a rejection.
* Claim (`GET /v1/claims/{id}`): workflow and intake status separately,
  certificate, masked IC, facts for the product, documents with classified
  type and status, intake and coverage reason codes (readable label, exact
  code alongside), rationale. Claimed, recommended and final payable are shown
  as three separate amounts, formatted from the backend's decimal strings
  without float conversion. Missing values read "Not available".
* Citations: clause id, exact quote, product, and a "Show source clause"
  control that loads the clause from
  `GET /v1/certificates/{product}/clauses/{clause_id}` and highlights the
  quote. The note under the heading states exactly what the backend checked
  (the quote is a substring of a clause retrieved for this claim). The source
  is labelled as the current wording file, not a historical snapshot.
* Trace: collapsed by default; per step status, duration, provider calls,
  document ids, retrieved clause ids and the (already redacted) error. When
  coverage was skipped, a sentence explains which intake problems caused the
  referral.
* Decision (`POST /v1/claims/{id}/adjuster-decision`): adjuster name (stated
  as not a sign-in), approve or override (final decision, payable and a
  required reason), then a confirmation dialog showing the exact decision and
  payable. Approve sends those values, so the server rejects a stale view.
  The submit button is disabled while the request runs. 422 shows field or
  form errors; 409 reloads the claim and says it is no longer pending; a
  dropped connection re-reads the claim before suggesting a retry. Decided
  claims leave the queue and stay in "Opened this session". The word "paid"
  or "settled" is never used: nothing is paid here.
* Audit (`GET /v1/claims/{id}/audit`): timestamp, actor, event, decision,
  payable, clause ids, reason. Read-only; the note says the SQLite triggers
  block edits but are not cryptographic tamper proofing.

Design (taste-skill house rules applied to an app screen): warm neutral
ground, dark ink, one deep teal accent, system font stack, tabular figures for
money, surfaces separated by tone instead of lines or shadows (only the modal
dialog has one), flat square-ish buttons, no gradients, no icons, no
decorative motion. Recommendation badges always carry words, so colour is
never the only signal. Three columns on desktop, two on tablet, one on a
phone.

Safety and accessibility: all API text is inserted with `textContent` (a test
fails if `innerHTML` or similar appears). Labelled controls, fieldset and
legend for the action, `aria-invalid` with described-by messages, a native
`<dialog>`, visible focus rings, focus moves to the outcome notice after a
decision, one polite live region for "Loaded CLM-..." and outcomes, and
`prefers-reduced-motion` turns transitions off.

### Demo seeding

`python -m claims.seed_demo --data data/ --db data/demo.db` runs every
generated packet through `assess_claim` and `ClaimStore.save`, the same steps
as POST, with FakeProvider fixed (no real model is ever called). Claims already
in the database are skipped, nothing is deleted, and the dataset is only read.
With seed 7 it gives, among others:

| Example | Claim |
| --- | --- |
| Clean covered medical | CLM-0001 |
| Motor partial (collision excess) | CLM-0002 |
| Medical partial (room-rate limit) | CLM-0012 |
| Excluded (motor) | CLM-0004 |
| Missing document, referred | CLM-0005 |

### Metrics

`GET /metrics` returns Prometheus text format (version 0.0.4), written by
`claims/metrics.py` with no extra dependency.

| Metric | Type | Labels | Meaning |
| --- | --- | --- | --- |
| `tabarru_claims` | gauge | `status` | Claims in SQLite by workflow status, read at scrape time |
| `tabarru_provider_retries_total` | counter | `step` | Provider calls that were a second attempt after bad output |
| `tabarru_citation_invalid_total` | counter | `product` | Claims whose citations were still invalid after the retry |
| `tabarru_step_duration_seconds` | histogram | `step` | Step latency in seconds (buckets 1 ms to 30 s) |

* Labels come from fixed lists only (status, step, product). Claim ids,
  names, ICs, document text and error strings are never labels; a test checks.
* Counters and the histogram live in the API process and reset when it
  restarts. With several worker processes each keeps its own values. The
  gauges come from SQLite, so they survive restarts.
* Counters change only when a claim is processed. Scraping `/metrics` (or any
  GET) does not change them; a test compares two scrapes.
* A retry is counted only on the second attempt, never per call. Retries are
  counted while the claim is assessed, so a rare concurrent duplicate that
  then gets 409 is included.
* `seed_demo`, `claims.eval` and the CLI run in their own processes, so their
  work does not appear in the server's counters.

### Docker

```bash
docker compose up --build -d          # serves http://localhost:8000/
docker compose exec api python -m claims.seed_demo --data /data --db /data/claims.db
docker compose down                   # the volume keeps the data
```

* `python:3.12-slim`, uv 0.13.1 (same as the lockfile), `uv sync --frozen
  --no-dev` from `uv.lock`, runs as uid 10001, binds 0.0.0.0:8000.
* `/data` is the named volume `tabarru-data` (dataset and `claims.db`). On
  start the entrypoint generates the 40-claim seed 7 dataset only if
  `/data/policies.jsonl` is missing, so an existing volume is never
  overwritten. Seeding is a separate explicit command and is repeatable.
* `CLAIMS_PROVIDER=fake` is set in `compose.yaml`. `.dockerignore` keeps
  `.git`, `.venv`, `.env*` (except `.env.example`), `data/`, `models/` and
  databases out of the image.
* A bind mount (`./data:/data`) instead of the named volume must be writable
  by uid 10001.
* Pytest does not need Docker.

Checked on this machine with Docker 29.9.2: the image built; the UI, `/docs`
and `/metrics` answered 200; seeding gave 40 created then 40 already present;
an approve returned 200; after `docker compose down` and `up` the decided claim
and the `decided` gauge were still there; `id` inside the container showed uid
10001; the image has no `.env.local`.

### Real-model comparison

Tooling: `python -m claims.models list` prints the models in the local
Ollama server and the Groq account; `python -m claims.models run` asks for a
source and model (or takes `--source`, `--model`), scores that model and
FakeProvider on the same claims, and writes
`reports/real/<source>__<model>/` (`coverage_eval.json` with run metadata,
`fake_baseline.json`, `comparison.md`). `reports/coverage_eval.json` (the
deterministic fake report) is not touched. Live runs are never part of pytest;
`tests/test_models_cli.py` covers the tool offline.

```bash
uv run python -m claims.models list --env-file .env.local
uv run python -m claims.models run --source ollama --model <name> --limit 10 --timeout 300
```

Measured so far (temperature 0, dataset seed 7, top-k 6, one retry):

| Model | Claims | decision_accuracy | citation_invalid_rate | payable_exact_match | Fake on same claims (accuracy) | Elapsed |
| --- | --- | --- | --- | --- | --- | --- |
| Ollama llama3.1 8B (pilot) | first 4 | 0.5 | 0.0 | 1.0 | 1.0 | 72 s |

A run of Ollama gemma-4-12b (QAT GGUF) on the first 10 claims is in progress
and will be added when it finishes. Groq was listed but not run, by choice
(no rate-limited remote runs). Four claims is too few to rank models; the
pilot only shows that the pipeline runs end to end with a real model.

## What is next

* Retrieval: weight claim-specific terms above the fixed base terms (the one
  measured miss), or retrieve once per question (period, exclusion, limit).
* Hybrid: when the LLM fallback fails, keep the trained label flagged for
  review instead of failing the document (see Task 3 results).
* Cross-document conflict flags for names and dates, not only amounts.
* Check the claim form's product against the register entry's product
  (today a MEDICAL claim on a MOTOR certificate is not flagged).
* Authentication for the adjuster view: the adjuster name is typed in and
  recorded, it is not a sign-in.
* Strip `<think>` blocks from reasoning models in the OpenAI-compatible
  adapter, so those models can be compared too.
