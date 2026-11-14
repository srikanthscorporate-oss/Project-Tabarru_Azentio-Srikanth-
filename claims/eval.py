"""Evaluate intake plus coverage against the labels.

    python -m claims.eval --provider fake --out reports/ [--data data/]

Runs the real pipeline (assess_claim) on every claim folder, in memory, with
no database. Labels are read here only, after prediction, to score.

Metric definitions (all claims = every labelled claim):
- decision_accuracy: correct decisions / all claims.
- confusion_matrix: expected decision -> predicted decision -> count.
- payable_exact_match: claims with |predicted - expected| <= 0.01 / all claims.
- retrieval_recall_at_k: expected clause ids found in the retrieved top-k /
  all expected clause ids, over claims that have expected clauses. A claim
  whose coverage was skipped retrieved nothing, so its expected ids count as missed.
- citation_precision: final cited ids that are expected / all final cited ids
  (null when nothing was cited).
- citation_recall: expected ids that were finally cited / all expected ids,
  over claims with expected clauses.
- citation_invalid_rate: claims whose final recommendation carries
  CITATION_INVALID / claims where the coverage provider was called. Claims
  bypassed to REFER are not in the denominator. attempted_citation_invalid_rate
  counts claims where any attempt broke a citation rule, including ones fixed
  on retry.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from decimal import Decimal
from pathlib import Path

from claims.coverage import CoverageContext, assess_claim
from claims.providers import FakeProvider, get_provider
from claims.schemas import Decision, InputDocument
from claims.validation import load_register

DECISIONS = [d.value for d in Decision]
_CITATION_ERRORS = ("must cite", "clause_id is not", "quote is not", "quote contains")


def _ratio(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def evaluate(data_dir: Path, provider_name: str) -> dict:
    labels = [
        json.loads(line)
        for line in (data_dir / "labels.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    ctx = CoverageContext.load(data_dir, load_register(data_dir / "policies.jsonl"))
    matrix = {e: dict.fromkeys(DECISIONS, 0) for e in DECISIONS}
    correct = payable_ok = 0
    exp_total = retrieved_hits = cited_hits = cited_total = cited_expected = 0
    called = final_invalid = attempted_invalid = 0
    for label in labels:
        # A fresh fake per claim keeps call numbering independent of claim order.
        provider = FakeProvider() if provider_name == "fake" else get_provider()
        folder = data_dir / "claims" / label["claim_id"]
        docs = [
            InputDocument(doc_id=p.stem, text=p.read_text(encoding="utf-8"))
            for p in sorted(folder.glob("*.txt"))
        ]
        claim = assess_claim(label["claim_id"], docs, provider, ctx)
        rec, coverage = claim.recommendation, claim.trace[-1]

        predicted, expected = rec.decision.value, label["expected_decision"]
        matrix[expected][predicted] += 1
        correct += predicted == expected
        payable_ok += abs(rec.payable_amount - Decimal(label["expected_payable"])) <= Decimal(
            "0.01"
        )

        expected_ids = set(label["expected_clause_ids"])
        cited_ids = {c.clause_id for c in rec.cited_clauses}
        exp_total += len(expected_ids)
        retrieved_hits += len(expected_ids & set(coverage.clause_ids))
        cited_expected += len(expected_ids & cited_ids)
        cited_total += len(cited_ids)
        cited_hits += len(cited_ids & expected_ids)
        if coverage.provider_calls:
            called += 1
            final_invalid += "CITATION_INVALID" in rec.reason_codes
            attempted_invalid += any(e in (coverage.error or "") for e in _CITATION_ERRORS)

    total = len(labels)
    return {
        "provider": provider_name,
        "claims": total,
        "k": ctx.k,
        "decision_accuracy": _ratio(correct, total),
        "confusion_matrix": matrix,
        "payable_exact_match": _ratio(payable_ok, total),
        "retrieval_recall_at_k": _ratio(retrieved_hits, exp_total),
        "citation_precision": _ratio(cited_hits, cited_total),
        "citation_recall": _ratio(cited_expected, exp_total),
        "citation_invalid_rate": _ratio(final_invalid, called),
        "attempted_citation_invalid_rate": _ratio(attempted_invalid, called),
        "coverage_provider_called": called,
    }


def format_table(report: dict) -> str:
    keys = [
        "decision_accuracy",
        "payable_exact_match",
        "retrieval_recall_at_k",
        "citation_precision",
        "citation_recall",
        "citation_invalid_rate",
        "attempted_citation_invalid_rate",
    ]
    lines = ["| metric | value |", "|---|---|"]
    lines += [f"| {k} | {report[k]} |" for k in keys]
    short = {"COVERED": "COV", "PARTIALLY_COVERED": "PART", "NOT_COVERED": "NOT", "REFER": "REF"}
    lines += ["", "| expected \\ predicted | " + " | ".join(short.values()) + " |"]
    lines.append("|---" * (len(DECISIONS) + 1) + "|")
    for e in DECISIONS:
        row = " | ".join(str(report["confusion_matrix"][e][p]) for p in DECISIONS)
        lines.append(f"| {short[e]} | {row} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score coverage recommendations.")
    parser.add_argument("--provider", default="fake", choices=["fake", "env"])
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("reports"))
    args = parser.parse_args(argv)
    started = time.perf_counter()
    report = evaluate(args.data, args.provider)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "coverage_eval.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(format_table(report))
    print(f"\n{report['claims']} claims in {time.perf_counter() - started:.2f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
