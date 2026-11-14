"""Trained document classifier (TF-IDF + logistic regression) and the runtime mode switch.

    CLAIMS_CLASSIFIER=llm|trained|hybrid   (default llm)
    CLAIMS_CLASSIFIER_MODEL=models/doc_classifier.joblib
    CLAIMS_HYBRID_THRESHOLD=0.40            (optional; default is the threshold stored in the model)

llm keeps the provider path in claims.pipeline. trained never calls the provider
for classification. hybrid uses the trained prediction when its top probability
is >= the threshold and otherwise falls back to the provider path, with its
schema validation and single retry.

The model only ever sees document text. joblib files are pickles, so only load
artifacts this repository produced; nothing here accepts an uploaded model.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from claims.schemas import DocType

Mode = Literal["llm", "trained", "hybrid"]
MODES: tuple[str, ...] = ("llm", "trained", "hybrid")
DEFAULT_MODEL_PATH = Path("models/doc_classifier.joblib")


MODEL_SETTINGS = {
    "tfidf": {
        "lowercase": True,
        "analyzer": "word",
        "ngram_range": (1, 2),
        "min_df": 2,
        "sublinear_tf": True,
    },
    "logreg": {"C": 1.0, "max_iter": 1000, "random_state": 0},
}


class ClassifierConfigError(ValueError):
    """CLAIMS_CLASSIFIER, the model path or the threshold is unusable."""


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(**MODEL_SETTINGS["tfidf"])),
            ("logreg", LogisticRegression(**MODEL_SETTINGS["logreg"])),
        ]
    )


@dataclass
class TrainedModel:
    pipeline: Pipeline
    metadata: dict = field(default_factory=dict)

    def predict(self, text: str) -> tuple[DocType, float]:
        """Top document type and its probability, read through the model's own classes_."""
        probabilities = self.pipeline.predict_proba([text])[0]
        best = int(probabilities.argmax())
        return DocType(self.pipeline.classes_[best]), float(probabilities[best])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "metadata": self.metadata}, path)

    @classmethod
    def load(cls, path: Path) -> TrainedModel:
        """Load a trusted local artifact written by claims.train_classifier."""
        if not path.is_file():
            raise ClassifierConfigError(
                f"classifier model not found at {path}; run "
                "`python -m claims.train_classifier` or set CLAIMS_CLASSIFIER_MODEL"
            )
        try:
            bundle = joblib.load(path)
            model = cls(bundle["pipeline"], bundle["metadata"])
            valid = {t.value for t in DocType}
            if not set(model.pipeline.classes_) <= valid:
                raise ValueError("unknown class labels")
            float(model.metadata["threshold"])
        except Exception as exc:
            raise ClassifierConfigError(
                f"classifier model at {path} is not a valid artifact ({type(exc).__name__}); "
                "retrain with `python -m claims.train_classifier`"
            ) from None
        return model


@dataclass
class ClassifierConfig:
    """How classify_documents picks a type. model is None only in llm mode."""

    mode: Mode = "llm"
    model: TrainedModel | None = None
    threshold: float = 0.5  # only read in hybrid mode, where it comes from the model or env

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ClassifierConfigError(
                f"unknown CLAIMS_CLASSIFIER {self.mode!r}; use one of {', '.join(MODES)}"
            )
        if self.mode != "llm" and self.model is None:
            raise ClassifierConfigError(f"CLAIMS_CLASSIFIER={self.mode} needs a trained model")
        if not 0.0 < self.threshold <= 1.0:
            raise ClassifierConfigError("CLAIMS_HYBRID_THRESHOLD must be in (0, 1]")

    def accept(self, confidence: float) -> bool:
        """True when the trained prediction is used without asking the LLM.

        trained always accepts; hybrid accepts at or above the threshold, so a
        probability exactly equal to the threshold does not fall back.
        """
        return self.mode == "trained" or confidence >= self.threshold


def get_classifier_config() -> ClassifierConfig:
    """Read CLAIMS_CLASSIFIER and friends. Loads the model once, and only when needed."""
    mode = os.environ.get("CLAIMS_CLASSIFIER", "llm").strip().lower()
    if mode not in MODES:
        raise ClassifierConfigError(
            f"unknown CLAIMS_CLASSIFIER {mode!r}; use one of {', '.join(MODES)}"
        )
    if mode == "llm":
        return ClassifierConfig()
    model = TrainedModel.load(Path(os.environ.get("CLAIMS_CLASSIFIER_MODEL", DEFAULT_MODEL_PATH)))
    raw = os.environ.get("CLAIMS_HYBRID_THRESHOLD")
    try:
        threshold = float(raw) if raw else float(model.metadata["threshold"])
    except ValueError:
        raise ClassifierConfigError("CLAIMS_HYBRID_THRESHOLD must be a number") from None
    return ClassifierConfig(mode, model, threshold)
