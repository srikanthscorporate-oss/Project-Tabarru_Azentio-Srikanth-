"""Train the TF-IDF + logistic regression document classifier.

    python -m claims.train_classifier [--out models/doc_classifier.joblib]

Training documents come from layouts 0 and 1 (seed TRAIN_SEED); the test set
comes from layout 2 only (seed TEST_SEED), so it measures generalisation to a
layout the model never saw. The test set is not used here at all except to
check that no test text also appears in training. Labels and template ids are
used to build and audit the split; the model is fitted on text alone.
"""

from __future__ import annotations

import argparse
import math
import platform
import sys
import time
from pathlib import Path

import joblib
import sklearn

from claims.classifier import (
    DEFAULT_MODEL_PATH,
    MODEL_SETTINGS,
    TrainedModel,
    build_pipeline,
)
from claims.generate import LAYOUTS_PER_TYPE, classifier_dataset, template_id
from claims.schemas import DocType

TRAIN_SEED = 1301
TEST_SEED = 2602
TRAIN_LAYOUTS = [0, 1]
HELDOUT_LAYOUTS = [2]
TRAIN_PER_CLASS = 200
TEST_PER_CLASS = 100


def template_ids(layouts: list[int]) -> list[str]:
    return sorted(template_id(t.value, layout) for t in DocType for layout in layouts)


def train_rows(per_class: int = TRAIN_PER_CLASS, seed: int = TRAIN_SEED) -> list[dict]:
    return classifier_dataset(seed, per_class, TRAIN_LAYOUTS)


def test_rows(per_class: int = TEST_PER_CLASS, seed: int = TEST_SEED) -> list[dict]:
    return classifier_dataset(seed, per_class, HELDOUT_LAYOUTS)


def check_split(train: list[dict], test: list[dict]) -> None:
    """Raise if the split leaks: shared templates or identical texts."""
    shared_templates = {r["template_id"] for r in train} & {r["template_id"] for r in test}
    if shared_templates:
        raise ValueError(f"templates in both train and test: {sorted(shared_templates)}")
    if {r["text"] for r in train} & {r["text"] for r in test}:
        raise ValueError("identical document text in train and test")


def choose_threshold(rows: list[dict]) -> float:
    """Hybrid threshold from training layouts only, never from the test set.

    Leave one training layout out: fit on one layout, predict the other, both
    ways round. This imitates meeting an unseen layout. The threshold is the
    smallest multiple of 0.05 strictly above the most confident wrong
    validation prediction, so every validation mistake would have gone to the LLM.
    """
    worst = 0.0
    for held in TRAIN_LAYOUTS:
        fit = [r for r in rows if r["template_id"] != template_id(r["doc_type"], held)]
        val = [r for r in rows if r["template_id"] == template_id(r["doc_type"], held)]
        model = train(fit, {})
        for row in val:
            doc_type, confidence = model.predict(row["text"])
            if doc_type.value != row["doc_type"]:
                worst = max(worst, confidence)
    return min(round(math.floor(worst / 0.05 + 1) * 0.05, 2), 1.0)


def train(train: list[dict], metadata: dict) -> TrainedModel:
    texts = [r["text"] for r in train]  # text only: no type, template or id features
    labels = [r["doc_type"] for r in train]
    pipeline = build_pipeline().fit(texts, labels)
    return TrainedModel(pipeline, {**metadata, "classes": [str(c) for c in pipeline.classes_]})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the document classifier.")
    parser.add_argument("--out", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--per-class", type=int, default=TRAIN_PER_CLASS)
    args = parser.parse_args(argv)

    started = time.perf_counter()
    train_set = train_rows(args.per_class)
    # Deduplicate within training so a repeated text is not weighted twice.
    train_set = list({r["text"]: r for r in train_set}.values())
    check_split(train_set, test_rows())
    assert len(TRAIN_LAYOUTS) + len(HELDOUT_LAYOUTS) == LAYOUTS_PER_TYPE

    metadata = {
        "train_seed": TRAIN_SEED,
        "test_seed": TEST_SEED,
        "train_per_class": args.per_class,
        "test_per_class": TEST_PER_CLASS,
        "train_size": len(train_set),
        "train_template_ids": template_ids(TRAIN_LAYOUTS),
        "heldout_template_ids": template_ids(HELDOUT_LAYOUTS),
        "settings": {k: {n: list(v) if isinstance(v, tuple) else v for n, v in s.items()}
                     for k, s in MODEL_SETTINGS.items()},
        "threshold": choose_threshold(train_set),
        "threshold_rule": "smallest 0.05 step above the most confident wrong prediction "
        "in leave-one-training-layout-out validation",
        "versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }  # fmt: skip
    model = train(train_set, metadata)
    model.save(args.out)
    print(
        f"trained on {len(train_set)} documents ({len(model.pipeline.classes_)} classes, "
        f"templates {', '.join(metadata['train_template_ids'])}) in "
        f"{time.perf_counter() - started:.2f} s; held out {', '.join(metadata['heldout_template_ids'])}; "
        f"saved {args.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
