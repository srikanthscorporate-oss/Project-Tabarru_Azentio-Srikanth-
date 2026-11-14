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

## What I would not trust without checking

The FakeProvider recovers every field and reason code on the 40 generated
claims. That is because the same author wrote the templates and the patterns.
It is not a measure of extraction quality. The OpenAI-compatible adapter is
covered by tests only for configuration and failure handling; it has not been
run against a live model.
