# AI usage

## Tools

* **Claude Code** (VS Code extension, Opus model) wrote the code, tests and
  docs in one session, driven by my prompt below. I own the design and
  review the code and test output before submitting.
* A second Claude Code agent (a code-reviewer subagent) reviewed the finished
  package independently, without seeing the main agent's conclusions.
* Locally installed Claude Code skills, used as checklists: `unlazy` (writes
  acceptance gates before the work and checks them before reporting done),
  `modern-python` (uv, ruff, dependency groups), `test-driven-development` and
  `verification-before-completion` (prove that tests can fail, run the real
  commands), `requesting-code-review` / `receiving-code-review` (independent
  review, then judge each finding before acting). `taste-skill` is for visual
  work only and was not used: Task 1 has no UI. It applies later, to the
  adjuster view.

## Key prompts

The main prompt, abridged:

> Implement Task 1 of the Azentio Project Tabarru assessment completely. Read
> the actual PDF, repository CLAUDE.md ... Complete Task 1 only. Read Tasks 2
> and 3 to keep interfaces extensible ... Prefer an explicit plain-Python
> pipeline and stdlib sqlite3. Keep dependencies small and code understandable
> enough for me to explain and extend without AI in the next interview ...
> FakeProvider must ... never read labels.jsonl, generator state, filenames as
> type hints ... Do not silently fall back from a failing real provider to fake
> results ... Always mask participant_ic ... Test failure paths as well as
> successful paths for leaks ... Do not fabricate evaluation numbers or claim
> unimplemented functionality.

The review prompt asked for correctness against the Task 1 spec, IC leak
paths, whether the fake provider could be cheating via labels or filenames,
hidden network use and needless complexity, with a concrete failure scenario
per finding.

## Where the AI was wrong and how it was caught

* **Build config.** The first `pyproject.toml` declared a build backend and a
  `README.md` that did not exist yet, so `uv sync` failed. Caught by the
  command error. Fixed by making the project a non-packaged app
  (`[tool.uv] package = false`) run with `python -m` from the repo root.
* **Interpreter.** The machine's default Python is 3.15, too new for some
  wheels. The project was pinned to 3.12 in `.python-version`
  (`requires-python` stays `>=3.10`).
* **Import path.** The first smoke script failed with
  `ModuleNotFoundError: claims` because the repo root was not on the path.
  Fixed for tests with `pythonpath = ["."]` in the pytest config.
* **Over-complicated test helper.** The first version of the helper that
  overrides fields in hand-built extractions had tangled branching. It was
  rewritten as a plain loop over the three documents before the tests ran.
* **Tests that passed first time.** All 71 tests passed on the first run,
  which says nothing about whether they can fail. The agent then broke the
  code on purpose three times (skip IC masking, disable the retry, change the
  1% rule from `>` to `>=`) and confirm the suite failed each time (4, 3 and 2
  failing tests), then restore the code.
* **IC redaction gaps (found by the review agent).** The redaction regex
  missed ICs written with spaces (`900101 00 1234`) and with a lowercase `o`.
  The generator does not produce those, but a POSTed document could, and its
  text would have been stored unredacted. Fixed, with a test for each spelling.
* **Review findings that were wrong or rejected.**
  * The reviewer said README.md was missing. It was written while the review
    was running.
  * It said documents marked as noisy might contain no noise. They always
    get an extra scan header and footer, so that is not the case.
  * It recommended dropping the extra `FIELDS_INCOMPLETE` reason code because
    the PDF lists four codes. I kept the code on purpose: without it, a claim
    with null required fields would be reported as `intake_complete`. The
    README documents it as an addition.
* **Lint on regex tables.** ruff flagged long lines in the pattern tables,
  which are marked `fmt: skip`. Line length is left to `ruff format` and E501
  is turned off.

## Second review pass (privacy and requirement audit)

A later Claude Code session was asked to re-check Task 1 against the PDF line
by line, treating the README's claims as unproven, and to close a known
privacy gap. Prompt, abridged:

> Review and finish Task 1 ... Treat the PDF as the source of truth ... the
> RequestValidationError handler returns err["loc"] unchanged ... sanitize ...
> audit IC handling across the complete output boundary ... recheck every
> Task 1 requirement ... the generator chooses noise probabilistically, check
> whether it guarantees the minimum ... do not implement Tasks 2 or 3.

What it found, each reproduced with a failing test before the fix:

* **422 echoed an IC used as a JSON key.** `{"900101-00-1234": 1}` came back
  in `detail[].loc`. The first version of the handler (also AI-written) only
  dropped `input` and missed that key names are user input too.
* **Retry feedback copied model-invented keys** into the prompt and trace.
* **IC-shaped `claim_id` / `doc_id`** were accepted and echoed in responses,
  the trace, logs and the 404 message. They are now rejected.
* **An IC in another text field** (name, diagnosis, cause of loss) from a
  model passed straight through to the response and database.
* **uvicorn's access log** printed an IC-shaped request path. This was not in
  the test suite at all; it showed up only when the server was run for real
  and its log grepped. Fixed with a logging filter.
* **The CLI did not validate `claim_id`** (the folder name) while the API did.
  It now builds the same `ClaimRequest` model.
* **Noise share was not guaranteed.** The 40% coin flip gave 1 of 6 noisy
  documents for `--claims 2 --seed 1`. The earlier README said "about 40%"
  and the existing test only checked seed 7, so it passed. Now a seeded
  sample of exactly ceil(40%).

The new tests were run against the old code to confirm they fail (8 privacy
tests, 4 noise tests) before the fixes were applied.

Limits of that pass: the suite was run on Python 3.12 only. Python 3.10
wheels were not in the local cache and nothing was downloaded, so 3.10 was
checked by parsing every module with the 3.10 interpreter, not by running
tests. No live model was called.

## What I would not trust without checking

The FakeProvider recovers every field and reason code on the 40 generated
claims. That is because the same author wrote the templates and the patterns.
It is not a measure of extraction quality. The OpenAI-compatible adapter is
covered by tests only for configuration and failure handling; it has not been
run against a live model.
