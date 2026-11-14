"""Compare trained, LLM (FakeProvider) and hybrid classification on the held-out layouts.

    python -m claims.compare_classifiers --out reports/ [--model models/doc_classifier.joblib]

The test set is rebuilt from the seed and layouts stored in the model metadata,
so every option sees exactly the same documents. Each option runs the real
classify_documents step, one document at a time, with a fresh FakeProvider.
A document whose classification fails stays in every denominator and appears
in the confusion matrix under the "failed" column.

Quality metrics and call counts are deterministic. Latency is wall-clock
(time.perf_counter_ns) and varies between runs; it is reported separately.
Cost and hosted latency are estimates from the stated hypothetical assumptions.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

from sklearn.metrics import confusion_matrix, f1_score

from claims.classifier import (
    DEFAULT_MODEL_PATH,
    ClassifierConfig,
    ClassifierConfigError,
    TrainedModel,
)
from claims.generate import classifier_dataset
from claims.pipeline import DocState, classify_documents
from claims.providers import FakeProvider
from claims.schemas import DocType

LABELS = [t.value for t in DocType]
FAILED = "failed"

# Hypothetical, not verified prices or measurements. Edit to match a real provider.
ASSUMPTIONS = {
    "docs_per_claim": 3.1,  # Task 1 generator: 124 documents over 40 claims
    "input_tokens_per_call": 450,  # instructions plus one OCR document
    "output_tokens_per_call": 15,  # {"doc_type": "..."}
    "usd_per_million_input_tokens": 0.50,
    "usd_per_million_output_tokens": 1.50,
    "hosted_latency_ms_per_call": 800,
    "hosted_calls_sequential": True,
    "scope": "classification step only, not extraction or coverage calls",
}


def evaluate(name: str, rows: list[dict], config: ClassifierConfig) -> dict:
    provider = FakeProvider()  # fresh per option: call counts and faults never leak
    truth, predicted, calls, timings = [], [], 0, []
    for i, row in enumerate(rows):
        doc = DocState(f"d{i}", row["text"])
        started = time.perf_counter_ns()
        trace = classify_documents([doc], provider, config)
        timings.append(time.perf_counter_ns() - started)
        calls += trace.provider_calls
        truth.append(row["doc_type"])
        predicted.append(doc.doc_type.value if doc.doc_type else FAILED)

    n = len(rows)
    columns = [*LABELS, FAILED]
    matrix = confusion_matrix(truth, predicted, labels=columns)[: len(LABELS)]
    call_rate = calls / n
    a = ASSUMPTIONS
    usd_per_call = (
        a["input_tokens_per_call"] * a["usd_per_million_input_tokens"]
        + a["output_tokens_per_call"] * a["usd_per_million_output_tokens"]
    ) / 1e6
    mean_ms = sum(timings) / n / 1e6
    return {
        "quality": {
            "documents": n,
            "accuracy": round(sum(t == p for t, p in zip(truth, predicted, strict=True)) / n, 4),
            # failed documents count as wrong for their true class
            "macro_f1": round(
                float(f1_score(truth, predicted, labels=LABELS, average="macro", zero_division=0)),
                4,
            ),
            "failed_classifications": predicted.count(FAILED),
            "confusion_matrix": {
                "rows_true": LABELS,
                "columns_predicted": columns,
                "matrix": matrix.tolist(),
            },
            "classification_llm_calls": calls,
            "llm_calls_per_1000_docs": round(call_rate * 1000, 1),
            "estimated_cost_usd_per_1000_claims": round(
                call_rate * a["docs_per_claim"] * 1000 * usd_per_call, 4
            ),
        },
        "timing": {
            "measured_mean_ms_per_doc": round(mean_ms, 4),
            "measured_with": "FakeProvider in process; not representative of a hosted LLM",
            "includes": "trained prediction, fallback prompt build, provider call, "
            "JSON parse and schema validation, retries",
            # assumption only: classification LLM calls per doc x assumed per-call latency
            "estimated_hosted_llm_ms_per_doc": round(
                call_rate * a["hosted_latency_ms_per_call"], 1
            ),
        },
        "option": name,
    }


def compare(model_path: Path) -> dict:
    load_started = time.perf_counter_ns()
    model = TrainedModel.load(model_path)
    load_ms = (time.perf_counter_ns() - load_started) / 1e6
    meta = model.metadata
    rows = classifier_dataset(meta["test_seed"], meta["test_per_class"], meta["heldout_layouts"])
    threshold = float(meta["threshold"])
    options = [
        evaluate("trained", rows, ClassifierConfig("trained", model, threshold)),
        evaluate("llm", rows, ClassifierConfig("llm")),
        evaluate("hybrid", rows, ClassifierConfig("hybrid", model, threshold)),
    ]
    return {
        "dataset": {
            "test_documents": len(rows),
            "test_class_distribution": dict(sorted(Counter(r["doc_type"] for r in rows).items())),
            "test_seed": meta["test_seed"],
            "train_seed": meta["train_seed"],
            "train_documents": meta["train_size"],
            "train_template_ids": meta["train_template_ids"],
            "heldout_template_ids": meta["heldout_template_ids"],
        },
        "hybrid_threshold": threshold,  # from the model; CLAIMS_HYBRID_THRESHOLD is not read here
        "threshold_rule": meta.get("threshold_rule"),
        "model": {
            "artifact": str(model_path),
            "settings": meta["settings"],
            "versions": meta["versions"],
            "load_ms": round(load_ms, 2),
        },
        "llm_provider": "FakeProvider (deterministic keyword stand-in; no real LLM was called)",
        "assumptions": ASSUMPTIONS,
        "options": options,
    }


def print_table(report: dict) -> None:
    print(
        f"{'option':<8} {'accuracy':>8} {'macro_f1':>8} {'failed':>6} "
        f"{'calls/1k docs':>13} {'usd/1k claims':>13} {'local ms/doc':>12} {'est LLM ms':>13}"
    )
    for o in report["options"]:
        q, t = o["quality"], o["timing"]
        print(
            f"{o['option']:<8} {q['accuracy']:>8.4f} {q['macro_f1']:>8.4f} "
            f"{q['failed_classifications']:>6} {q['llm_calls_per_1000_docs']:>13.1f} "
            f"{q['estimated_cost_usd_per_1000_claims']:>13.4f} "
            f"{t['measured_mean_ms_per_doc']:>12.4f} {t['estimated_hosted_llm_ms_per_doc']:>13.1f}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare document classifiers.")
    parser.add_argument("--out", type=Path, default=Path("reports"))
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args(argv)
    started = time.perf_counter()
    try:
        report = compare(args.model)
    except ClassifierConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "classifier_comparison.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print_table(report)
    print(f"wrote {path} in {time.perf_counter() - started:.2f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
