"""Task 3: trained document classifier, CLAIMS_CLASSIFIER modes and the comparison report."""

import json
import re

import joblib
import numpy as np
import pytest

from claims.classifier import (
    ClassifierConfig,
    ClassifierConfigError,
    TrainedModel,
    get_classifier_config,
)
from claims.compare_classifiers import FAILED, LABELS
from claims.compare_classifiers import main as compare_main
from claims.coverage import CoverageContext, assess_claim
from claims.generate import classifier_dataset
from claims.intake import main as intake_main
from claims.pipeline import DocState, classify_documents, run_intake
from claims.providers import FakeProvider
from claims.schemas import DocType
from claims.train_classifier import (
    HELDOUT_LAYOUTS,
    TEST_PER_CLASS,
    TRAIN_LAYOUTS,
    TRAIN_PER_CLASS,
    check_split,
    choose_threshold,
    train,
    train_rows,
)
from claims.train_classifier import main as train_main
from claims.train_classifier import (
    test_rows as heldout_rows,
)
from tests.conftest import first_label, load_packet

SMALL = 40  # documents per class for the fixture model: fast, still five classes


class CountingProvider(FakeProvider):
    """FakeProvider that also counts calls per task, to tell classification from extraction."""

    def __init__(self, faults=None):
        super().__init__(faults)
        self.by_task: dict[str, int] = {}

    def complete(self, prompt: str) -> str:
        task = re.search(r"^TASK: (\w+)$", prompt, re.M).group(1)
        self.by_task[task] = self.by_task.get(task, 0) + 1
        return super().complete(prompt)


class StubModel:
    """Returns a scripted prediction so both hybrid branches are tested deterministically."""

    def __init__(self, doc_type: DocType, confidence: float):
        self.answer = (doc_type, confidence)
        self.metadata = {"threshold": 0.4}

    def predict(self, text: str):
        return self.answer


@pytest.fixture(scope="session")
def model_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("models") / "doc_classifier.joblib"
    assert train_main(["--out", str(path), "--per-class", str(SMALL)]) == 0
    return path


@pytest.fixture(scope="session")
def model(model_path):
    return TrainedModel.load(model_path)


# ---- dataset and split


def test_training_set_size_classes_and_specimen():
    rows = train_rows()
    counts = {t.value: sum(r["doc_type"] == t.value for r in rows) for t in DocType}
    assert set(counts) == {t.value for t in DocType}
    assert all(n >= 150 for n in counts.values()), counts
    assert all("SPECIMEN" in r["text"] for r in rows + heldout_rows())


def test_heldout_templates_absent_from_training_and_no_text_overlap():
    train_set, test_set = train_rows(), heldout_rows()
    train_templates = {r["template_id"] for r in train_set}
    test_templates = {r["template_id"] for r in test_set}
    assert not train_templates & test_templates
    assert {t.rsplit(".t", 1)[1] for t in test_templates} == {str(x) for x in HELDOUT_LAYOUTS}
    assert len(test_templates) == len(DocType)  # one held-out layout for every type
    assert not {r["text"] for r in train_set} & {r["text"] for r in test_set}
    check_split(train_set, test_set)  # the guard used by training agrees


def test_check_split_rejects_leaks():
    rows = classifier_dataset(5, 3, TRAIN_LAYOUTS)
    with pytest.raises(ValueError, match="templates"):
        check_split(rows, rows[:2])
    other = [dict(r, template_id="x.t9") for r in rows[:1]]
    with pytest.raises(ValueError, match="identical"):
        check_split(rows, other)


def test_dataset_is_deterministic_and_noisy():
    assert classifier_dataset(9, 20, [0, 1]) == classifier_dataset(9, 20, [0, 1])
    assert classifier_dataset(9, 20, [0, 1]) != classifier_dataset(10, 20, [0, 1])
    rows = classifier_dataset(9, 50, [0, 1, 2])
    assert 0.2 < sum(r["ocr_noise"] for r in rows) / len(rows) < 0.6
    assert {r["template_id"] for r in rows} >= {f"claim_form.t{i}" for i in range(3)}


def test_packet_labels_record_template_ids(labels):
    docs = [d for label in labels for d in label["documents"]]
    assert all(d["template_id"] == f"{d['doc_type']}.t{d['template_id'][-1]}" for d in docs)
    assert {d["template_id"][-1] for d in docs} <= {"0", "1"}  # packets keep Task 1 layouts


def test_required_counts_are_configured():
    assert TRAIN_PER_CLASS >= 150 and TEST_PER_CLASS > 0


# ---- model artifact


def test_saved_model_loads_and_predicts_valid_types(model, model_path):
    meta = model.metadata
    assert meta["train_seed"] != meta["test_seed"]
    assert set(meta["classes"]) == {t.value for t in DocType}
    assert not set(meta["train_template_ids"]) & set(meta["heldout_template_ids"])
    assert {"settings", "versions", "threshold"} <= set(meta)
    for row in heldout_rows(per_class=4):
        doc_type, confidence = model.predict(row["text"])
        assert isinstance(doc_type, DocType) and 0.0 < confidence <= 1.0


def test_probability_maps_through_classes(model):
    text = heldout_rows(per_class=1)[0]["text"]
    doc_type, confidence = model.predict(text)
    probs = model.pipeline.predict_proba([text])[0]
    index = list(model.pipeline.classes_).index(doc_type.value)
    assert confidence == pytest.approx(probs[index]) == pytest.approx(probs.max())


def test_prediction_uses_classes_order_not_enum_order():
    class Fixed:
        classes_ = np.array(["police_report", "claim_form"])  # not DocType order

        def predict_proba(self, texts):
            return np.array([[0.9, 0.1]])

    assert TrainedModel(Fixed()).predict("x") == (DocType.POLICE_REPORT, 0.9)


def test_model_sees_text_only():
    rows = classifier_dataset(3, 10, [0])
    a = train(rows, {})
    b = train([dict(r, template_id="other", ocr_noise=False) for r in rows], {})
    text = rows[0]["text"]
    assert a.pipeline.predict_proba([text]).tolist() == b.pipeline.predict_proba([text]).tolist()
    vocab = a.pipeline.named_steps["tfidf"].vocabulary_
    assert not any(term == "t0" or "doc_type" in term for term in vocab)


def test_tfidf_is_fitted_on_training_text_only(model):
    vocab = model.pipeline.named_steps["tfidf"].vocabulary_
    # "bodyshop" and "costing" appear only in the held-out repair_estimate layout
    assert "bodyshop" not in vocab and "costing" not in vocab


def test_threshold_comes_from_training_layouts():
    rows = classifier_dataset(4, 20, TRAIN_LAYOUTS)
    first = choose_threshold(rows)
    assert 0.0 < first <= 1.0 and round(first / 0.05, 6).is_integer()
    assert choose_threshold(rows) == first  # deterministic


# ---- configuration


def test_default_mode_is_llm_and_needs_no_model():
    config = get_classifier_config()
    assert config.mode == "llm" and config.model is None


def test_invalid_mode_is_rejected(monkeypatch):
    monkeypatch.setenv("CLAIMS_CLASSIFIER", "neural")
    with pytest.raises(ClassifierConfigError, match="unknown CLAIMS_CLASSIFIER"):
        get_classifier_config()


@pytest.mark.parametrize("mode", ["trained", "hybrid"])
def test_missing_model_fails_clearly(monkeypatch, tmp_path, mode):
    monkeypatch.setenv("CLAIMS_CLASSIFIER", mode)
    monkeypatch.setenv("CLAIMS_CLASSIFIER_MODEL", str(tmp_path / "absent.joblib"))
    with pytest.raises(ClassifierConfigError, match="train_classifier"):
        get_classifier_config()


def test_invalid_model_file_fails_clearly(monkeypatch, tmp_path):
    bad = tmp_path / "bad.joblib"
    joblib.dump({"something": "else"}, bad)
    monkeypatch.setenv("CLAIMS_CLASSIFIER", "trained")
    monkeypatch.setenv("CLAIMS_CLASSIFIER_MODEL", str(bad))
    with pytest.raises(ClassifierConfigError, match="not a valid artifact"):
        get_classifier_config()


def test_env_selects_trained_and_threshold(monkeypatch, model_path):
    monkeypatch.setenv("CLAIMS_CLASSIFIER", "hybrid")
    monkeypatch.setenv("CLAIMS_CLASSIFIER_MODEL", str(model_path))
    assert get_classifier_config().threshold == TrainedModel.load(model_path).metadata["threshold"]
    monkeypatch.setenv("CLAIMS_HYBRID_THRESHOLD", "0.65")
    assert get_classifier_config().threshold == 0.65
    for bad in ("high", "0", "1.5"):
        monkeypatch.setenv("CLAIMS_HYBRID_THRESHOLD", bad)
        with pytest.raises(ClassifierConfigError, match="CLAIMS_HYBRID_THRESHOLD"):
            get_classifier_config()


def test_cli_reports_missing_model(monkeypatch, dataset, tmp_path, capsys):
    monkeypatch.setenv("CLAIMS_CLASSIFIER", "trained")
    monkeypatch.setenv("CLAIMS_CLASSIFIER_MODEL", str(tmp_path / "absent.joblib"))
    code = intake_main(
        [str(dataset / "claims" / "CLM-0001"), "--register", str(dataset / "policies.jsonl")]
    )
    assert code == 1 and "classifier model not found" in capsys.readouterr().err


# ---- classify_documents modes and provider accounting


def classify(config, provider, text="SPECIMEN some document"):
    doc = DocState("d1", text)
    trace = classify_documents([doc], provider, config)
    return doc, trace


def test_trained_mode_makes_no_classification_calls(model):
    provider = CountingProvider()
    doc, trace = classify(
        ClassifierConfig("trained", model, 0.99), provider, heldout_rows(1)[0]["text"]
    )
    assert doc.doc_type is not None
    assert trace.provider_calls == 0 and provider.calls == 0


def test_trained_mode_ignores_low_confidence():
    provider = CountingProvider()
    doc, trace = classify(
        ClassifierConfig("trained", StubModel(DocType.POLICE_REPORT, 0.01)), provider
    )
    assert doc.doc_type == DocType.POLICE_REPORT and trace.provider_calls == 0


def test_hybrid_high_confidence_skips_provider():
    provider = CountingProvider()
    config = ClassifierConfig("hybrid", StubModel(DocType.REPAIR_ESTIMATE, 0.95), 0.4)
    doc, trace = classify(config, provider)
    assert doc.doc_type == DocType.REPAIR_ESTIMATE
    assert trace.provider_calls == 0 and provider.calls == 0


def test_hybrid_low_confidence_calls_provider():
    provider = CountingProvider()
    config = ClassifierConfig("hybrid", StubModel(DocType.REPAIR_ESTIMATE, 0.2), 0.4)
    doc, trace = classify(config, provider, "SPECIMEN POLICE REPORT\nLaporan polis")
    assert doc.doc_type == DocType.POLICE_REPORT  # the provider's answer, not the model's
    assert trace.provider_calls == provider.calls == 1


@pytest.mark.parametrize(("confidence", "calls"), [(0.4, 0), (0.39999, 1)])
def test_threshold_boundary_equal_is_accepted(confidence, calls):
    """At exactly the threshold the trained answer is kept; just below falls back."""
    provider = CountingProvider()
    config = ClassifierConfig("hybrid", StubModel(DocType.CLAIM_FORM, confidence), 0.4)
    _, trace = classify(config, provider, "SPECIMEN CLAIM FORM")
    assert trace.provider_calls == calls


def test_hybrid_fallback_keeps_retry_and_failure_behaviour():
    low = ClassifierConfig("hybrid", StubModel(DocType.CLAIM_FORM, 0.1), 0.4)
    retried = CountingProvider(faults={1: "malformed_json"})
    doc, trace = classify(low, retried, "SPECIMEN CLAIM FORM")
    assert doc.doc_type == DocType.CLAIM_FORM and trace.provider_calls == 2
    assert trace.status == "ok"

    failing = CountingProvider(faults={1: "malformed_json", 2: "malformed_json"})
    doc, trace = classify(low, failing, "SPECIMEN CLAIM FORM")
    assert doc.doc_type is None and doc.error.startswith("classification failed")
    assert trace.provider_calls == 2 and trace.status == "failed"


def test_llm_mode_unchanged():
    provider = CountingProvider()
    doc, trace = classify(ClassifierConfig(), provider, "SPECIMEN CLAIM FORM")
    assert doc.doc_type == DocType.CLAIM_FORM and trace.provider_calls == 1


@pytest.mark.parametrize("mode", ["trained", "hybrid"])
def test_pipeline_end_to_end_with_trained_modes(mode, model, dataset, labels, register):
    label = first_label(labels, "clean", "MOTOR")
    docs = load_packet(dataset, label["claim_id"])
    provider = CountingProvider()
    config = ClassifierConfig(mode, model, 0.0001 if mode == "hybrid" else 0.5)
    ctx = CoverageContext.load(dataset, register)
    claim = assess_claim(label["claim_id"], docs, provider, ctx, config)

    assert [t.step for t in claim.trace][0] == "classify_documents"
    assert claim.trace[0].provider_calls == 0
    assert provider.by_task.get("classify_document", 0) == 0
    assert provider.by_task["extract_fields"] == len(docs)  # extraction still on the provider
    assert provider.by_task["assess_coverage"] == 1  # coverage still on the provider
    assert claim.status == "pending_adjuster" and claim.recommendation is not None
    assert claim.fields.participant_ic.startswith("XXXXXX-XX-")
    expected = {d["doc_id"]: d["doc_type"] for d in label["documents"]}
    assert {d.doc_id: d.doc_type.value for d in claim.documents} == expected


def test_hybrid_fallback_errors_do_not_leak_ic(labels, dataset, register):
    label = first_label(labels, "clean", "MEDICAL")
    ic = label["fields"]["participant_ic"]
    docs = load_packet(dataset, label["claim_id"])
    config = ClassifierConfig("hybrid", StubModel(DocType.CLAIM_FORM, 0.1), 0.4)

    class EchoIc:
        name = "echo"

        def complete(self, prompt):
            return json.dumps({"doc_type": f"x {ic}", ic: 1})

    claim = run_intake(label["claim_id"], docs, EchoIc(), register, config)
    assert claim.trace[0].error and claim.trace[0].provider_calls == 2 * len(docs)
    assert ic not in claim.model_dump_json()


# ---- comparison report


@pytest.fixture(scope="session")
def report(model_path, tmp_path_factory):
    out = tmp_path_factory.mktemp("reports")
    assert compare_main(["--out", str(out), "--model", str(model_path)]) == 0
    return json.loads((out / "classifier_comparison.json").read_text())


def test_report_has_three_options_and_metrics(report):
    assert [o["option"] for o in report["options"]] == ["trained", "llm", "hybrid"]
    for option in report["options"]:
        q = option["quality"]
        assert {"accuracy", "macro_f1", "confusion_matrix", "llm_calls_per_1000_docs",
                "estimated_cost_usd_per_1000_claims"} <= set(q)  # fmt: skip
        assert option["timing"]["measured_mean_ms_per_doc"] >= 0
        assert q["confusion_matrix"]["rows_true"] == LABELS
        assert q["confusion_matrix"]["columns_predicted"] == [*LABELS, FAILED]
    assert report["options"][0]["quality"]["classification_llm_calls"] == 0
    assert report["dataset"]["heldout_template_ids"]
    assert {"docs_per_claim", "usd_per_million_input_tokens"} <= set(report["assumptions"])
    assert "hybrid_threshold" in report


def test_failed_documents_stay_in_denominators(report):
    total = report["dataset"]["test_documents"]
    for option in report["options"]:
        q = option["quality"]
        matrix = q["confusion_matrix"]["matrix"]
        assert sum(map(sum, matrix)) == q["documents"] == total
        assert sum(row[-1] for row in matrix) == q["failed_classifications"]
        correct = sum(matrix[i][i] for i in range(len(LABELS)))
        assert q["accuracy"] == round(correct / total, 4)
    llm = report["options"][1]["quality"]
    assert llm["failed_classifications"] > 0  # FakeProvider has no keywords for one layout


def test_report_quality_is_deterministic(model_path, report, tmp_path):
    assert compare_main(["--out", str(tmp_path), "--model", str(model_path)]) == 0
    again = json.loads((tmp_path / "classifier_comparison.json").read_text())
    assert [o["quality"] for o in again["options"]] == [o["quality"] for o in report["options"]]


def test_compare_without_model_fails_clearly(tmp_path, capsys):
    assert compare_main(["--out", str(tmp_path), "--model", str(tmp_path / "none.joblib")]) == 1
    assert "train_classifier" in capsys.readouterr().err


def test_api_uses_classifier_from_env(monkeypatch, model_path, dataset, labels, tmp_path):
    from fastapi.testclient import TestClient

    from claims.api import create_app

    monkeypatch.setenv("CLAIMS_CLASSIFIER", "trained")
    monkeypatch.setenv("CLAIMS_CLASSIFIER_MODEL", str(model_path))
    app = create_app(tmp_path / "t3.db", dataset / "policies.jsonl", FakeProvider())
    label = first_label(labels, "clean", "MEDICAL")
    docs = [d.model_dump() for d in load_packet(dataset, label["claim_id"])]
    assert (
        TestClient(app)
        .post("/v1/claims", json={"claim_id": label["claim_id"], "documents": docs})
        .status_code
        == 201
    )
    detail = TestClient(app).get(f"/v1/claims/{label['claim_id']}").json()
    assert detail["trace"][0]["step"] == "classify_documents"
    assert detail["trace"][0]["provider_calls"] == 0

    monkeypatch.setenv("CLAIMS_CLASSIFIER", "bogus")
    with pytest.raises(ClassifierConfigError):
        create_app(tmp_path / "t4.db", dataset / "policies.jsonl", FakeProvider())
