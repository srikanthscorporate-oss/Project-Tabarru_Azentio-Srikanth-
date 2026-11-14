import json
import sqlite3
import threading
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient

from claims.api import create_app
from claims.eval import evaluate
from claims.providers import FakeProvider
from claims.schemas import AdjusterDecisionRecord
from claims.storage import AlreadyDecidedError, ClaimStore
from tests.conftest import load_packet


@pytest.fixture
def db(tmp_path):
    return tmp_path / "adj.db"


@pytest.fixture
def client(dataset, db):
    return TestClient(create_app(db, dataset / "policies.jsonl", FakeProvider()))


def label_for(labels, variant, product=None):
    return next(
        lab
        for lab in labels
        if lab["coverage_scenario"] == variant and (product is None or lab["product"] == product)
    )


def submit(client, dataset, label):
    docs = load_packet(dataset, label["claim_id"])
    body = {"claim_id": label["claim_id"], "documents": [d.model_dump() for d in docs]}
    response = client.post("/v1/claims", json=body)
    assert response.status_code == 201
    return response.json()


def test_post_then_get_keeps_recommendation_and_coverage_trace(client, dataset, labels):
    label = label_for(labels, "partial", "MOTOR")
    posted = submit(client, dataset, label)
    rec = posted["recommendation"]
    assert set(rec) == {"decision", "cited_clauses", "rationale", "payable_amount", "reason_codes"}
    assert rec["decision"] == "PARTIALLY_COVERED"
    assert rec["payable_amount"] == label["expected_payable"]
    detail = client.get(f"/v1/claims/{label['claim_id']}").json()
    assert detail["recommendation"] == rec
    assert detail["status"] == "pending_adjuster" and detail["intake_status"] == "intake_complete"
    coverage = detail["trace"][-1]
    assert coverage["step"] == "coverage" and coverage["provider_calls"] == 1
    assert rec["cited_clauses"][0]["clause_id"] in coverage["clause_ids"]


def test_queue_lists_pending_claims_including_referrals(client, dataset, labels):
    covered = label_for(labels, "covered", "MEDICAL")
    refer = next(lab for lab in labels if lab["scenario"] == "missing_document")
    submit(client, dataset, covered)
    submit(client, dataset, refer)
    queue = client.get("/v1/adjuster/queue").json()
    by_id = {item["claim_id"]: item for item in queue}
    assert set(by_id) == {covered["claim_id"], refer["claim_id"]}
    assert by_id[refer["claim_id"]]["recommendation"]["decision"] == "REFER"
    assert "MISSING_DOCUMENT" in by_id[refer["claim_id"]]["recommendation"]["reason_codes"]
    assert by_id[covered["claim_id"]]["recommendation"]["cited_clauses"]


def test_approve_decides_and_writes_audit(client, dataset, labels, db):
    label = label_for(labels, "covered", "MOTOR")
    posted = submit(client, dataset, label)
    response = client.post(
        f"/v1/claims/{label['claim_id']}/adjuster-decision",
        json={"adjuster": "adj.lim", "action": "approve"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "decided"
    assert body["recommendation"] == posted["recommendation"]  # original kept separately
    assert body["adjuster_decision"]["final_payable"] == label["expected_payable"]
    assert client.get("/v1/adjuster/queue").json() == []

    rows = ClaimStore(db).audit(label["claim_id"])
    assert [r["event"] for r in rows] == ["recommendation", "adjuster_approve"]
    assert rows[0]["actor"] == "coverage-agent" and rows[1]["actor"] == "adj.lim"
    assert rows[1]["payable"] == label["expected_payable"]
    assert rows[1]["clause_ids"] == ["MTR-2.1"]
    assert all(r["timestamp"] for r in rows)


def test_override_with_reason(client, dataset, labels, db):
    label = label_for(labels, "covered", "MEDICAL")
    submit(client, dataset, label)
    response = client.post(
        f"/v1/claims/{label['claim_id']}/adjuster-decision",
        json={
            "adjuster": "adj.lim",
            "action": "override",
            "final_decision": "PARTIALLY_COVERED",
            "final_payable": "100.00",
            "reason": "pharmacy item not medically necessary",
        },
    )
    assert response.status_code == 200
    rows = ClaimStore(db).audit(label["claim_id"])
    assert rows[-1]["decision"] == "PARTIALLY_COVERED" and rows[-1]["payable"] == "100.00"
    assert rows[-1]["reason"] == "pharmacy item not medically necessary"


@pytest.mark.parametrize("reason", [None, "", "   "])
def test_override_without_reason_is_422_and_writes_nothing(client, dataset, labels, db, reason):
    label = label_for(labels, "covered", "MEDICAL")
    submit(client, dataset, label)
    body = {"adjuster": "adj.lim", "action": "override", "final_decision": "NOT_COVERED",
            "final_payable": "0.00", "reason": reason}  # fmt: skip
    response = client.post(f"/v1/claims/{label['claim_id']}/adjuster-decision", json=body)
    assert response.status_code == 422
    assert client.get(f"/v1/claims/{label['claim_id']}").json()["status"] == "pending_adjuster"
    assert [r["event"] for r in ClaimStore(db).audit(label["claim_id"])] == ["recommendation"]


@pytest.mark.parametrize(
    "body",
    [
        {"action": "approve", "final_decision": "NOT_COVERED"},  # silent override
        {"action": "approve", "final_payable": "1.00"},
        {"action": "override", "final_decision": "NOT_COVERED", "final_payable": "5.00", "reason": "x"},
        {"action": "override", "final_decision": "COVERED", "final_payable": "0.00", "reason": "x"},
        {"action": "override", "final_decision": "COVERED", "final_payable": "-1", "reason": "x"},
        {"action": "settle"},
    ],
)  # fmt: skip
def test_inconsistent_decisions_are_422(client, dataset, labels, db, body):
    label = label_for(labels, "covered", "MEDICAL")
    submit(client, dataset, label)
    response = client.post(
        f"/v1/claims/{label['claim_id']}/adjuster-decision", json={"adjuster": "a", **body}
    )
    assert response.status_code == 422
    assert len(ClaimStore(db).audit(label["claim_id"])) == 1


def test_second_decision_is_409_and_unknown_is_404(client, dataset, labels, db):
    label = label_for(labels, "covered", "MEDICAL")
    submit(client, dataset, label)
    url = f"/v1/claims/{label['claim_id']}/adjuster-decision"
    assert client.post(url, json={"adjuster": "a", "action": "approve"}).status_code == 200
    assert client.post(url, json={"adjuster": "b", "action": "approve"}).status_code == 409
    assert len(ClaimStore(db).audit(label["claim_id"])) == 2
    missing = client.post(
        "/v1/claims/NOPE-1/adjuster-decision", json={"adjuster": "a", "action": "approve"}
    )
    assert missing.status_code == 404


def test_concurrent_decisions_only_one_wins(client, dataset, labels, db):
    label = label_for(labels, "covered", "MOTOR")
    submit(client, dataset, label)
    record = AdjusterDecisionRecord(
        adjuster="a", action="approve", final_decision="COVERED",
        final_payable=Decimal(label["expected_payable"]), reason=None,
    )  # fmt: skip
    results = []

    def decide():
        try:
            ClaimStore(db).decide(label["claim_id"], record)
            results.append("ok")
        except AlreadyDecidedError:
            results.append("conflict")

    threads = [threading.Thread(target=decide) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(results) == ["conflict"] * 7 + ["ok"]
    assert len(ClaimStore(db).audit(label["claim_id"])) == 2


def test_audit_is_append_only(client, dataset, labels, db):
    label = label_for(labels, "covered", "MEDICAL")
    submit(client, dataset, label)
    before = ClaimStore(db).audit()
    with sqlite3.connect(db) as conn:
        for sql in ("UPDATE audit SET payable = '0.00'", "DELETE FROM audit"):
            with pytest.raises(sqlite3.IntegrityError, match="append-only"):
                conn.execute(sql)
    submit(client, dataset, label_for(labels, "covered", "MOTOR"))
    after = ClaimStore(db).audit()
    assert after[: len(before)] == before and len(after) == len(before) + 1


def test_task1_database_is_migrated_without_data_loss(tmp_path, dataset, labels):
    db = tmp_path / "old.db"
    with sqlite3.connect(db) as conn:  # the Task 1 schema, with one stored claim
        conn.executescript(
            """
            CREATE TABLE claims (claim_id TEXT PRIMARY KEY, status TEXT NOT NULL, product TEXT,
                fields TEXT NOT NULL, reason_codes TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
            CREATE TABLE documents (claim_id TEXT NOT NULL, position INTEGER NOT NULL,
                doc_id TEXT NOT NULL, doc_type TEXT, status TEXT NOT NULL, text TEXT NOT NULL,
                PRIMARY KEY (claim_id, doc_id));
            CREATE TABLE trace (claim_id TEXT NOT NULL, seq INTEGER NOT NULL, step TEXT NOT NULL,
                doc_ids TEXT NOT NULL, status TEXT NOT NULL, duration_ms REAL NOT NULL,
                provider_calls INTEGER NOT NULL, error TEXT, PRIMARY KEY (claim_id, seq));
            INSERT INTO claims (claim_id, status, product, fields, reason_codes)
                VALUES ('OLD-1', 'flagged', 'MOTOR', '{}', '["AMOUNT_MISMATCH"]');
            INSERT INTO trace VALUES ('OLD-1', 0, 'validate', '[]', 'ok', 1.0, 0, NULL);
            """
        )
    client = TestClient(create_app(db, dataset / "policies.jsonl", FakeProvider()))
    old = client.get("/v1/claims/OLD-1").json()
    assert old["status"] == "flagged" and old["reason_codes"] == ["AMOUNT_MISMATCH"]
    assert old["recommendation"] is None and old["trace"][0]["clause_ids"] == []
    assert client.get("/v1/adjuster/queue").json() == []
    decided = client.post(
        "/v1/claims/OLD-1/adjuster-decision", json={"adjuster": "a", "action": "approve"}
    )
    assert decided.status_code == 409  # never assessed, so not pending
    submit(client, dataset, label_for(labels, "covered", "MEDICAL"))  # new claims still work
    body = {"claim_id": "OLD-1", "documents": [{"doc_id": "d1", "text": "x"}]}
    assert client.post("/v1/claims", json=body).status_code == 409  # duplicate check kept


def test_adjuster_input_does_not_leak_ic(client, dataset, labels, db):
    label = label_for(labels, "covered", "MEDICAL")
    ic = label["fields"]["participant_ic"]
    submit(client, dataset, label)
    url = f"/v1/claims/{label['claim_id']}/adjuster-decision"
    bad_name = client.post(url, json={"adjuster": ic, "action": "approve"})
    assert bad_name.status_code == 422 and ic not in bad_name.text
    response = client.post(
        url,
        json={"adjuster": "a", "action": "override", "final_decision": "REFER",
              "final_payable": "0.00", "reason": f"check participant {ic} again"},
    )  # fmt: skip
    assert response.status_code == 200 and ic not in response.text
    with sqlite3.connect(db) as conn:
        dump = "\n".join(conn.iterdump())
    assert ic not in dump


def test_eval_is_reproducible_and_offline(dataset, tmp_path):
    first, second = evaluate(dataset, "fake"), evaluate(dataset, "fake")
    assert first == second
    for key in ("decision_accuracy", "confusion_matrix", "payable_exact_match",
                "retrieval_recall_at_k", "citation_precision", "citation_recall",
                "citation_invalid_rate"):  # fmt: skip
        assert key in first
    assert sum(sum(row.values()) for row in first["confusion_matrix"].values()) == 40
    json.dumps(first)


def test_zero_payable_partial_recommendation_can_be_approved(client, dataset, labels, db):
    # Regression: an excess that absorbs the whole loss gives PARTIALLY_COVERED with
    # 0.00 payable. Approving the stored recommendation must succeed, not 422.
    from claims.api import _decision_record
    from claims.schemas import AdjusterDecisionRequest, Decision

    label = label_for(labels, "covered", "MEDICAL")
    submit(client, dataset, label)
    claim = ClaimStore(db).get(label["claim_id"])
    rec = claim.recommendation.model_copy(
        update={"decision": Decision.PARTIALLY_COVERED, "payable_amount": Decimal("0.00")}
    )
    claim = claim.model_copy(update={"recommendation": rec})
    record = _decision_record(AdjusterDecisionRequest(adjuster="a", action="approve"), claim)
    assert record.final_decision is Decision.PARTIALLY_COVERED
    assert record.final_payable == Decimal("0.00")


def test_override_to_zero_payable_partial_is_422(client, dataset, labels):
    label = label_for(labels, "covered", "MEDICAL")
    submit(client, dataset, label)
    body = {
        "adjuster": "a",
        "action": "override",
        "final_decision": "PARTIALLY_COVERED",
        "final_payable": "0.00",
        "reason": "excess absorbs the loss",
    }
    response = client.post(f"/v1/claims/{label['claim_id']}/adjuster-decision", json=body)
    assert response.status_code == 422


def test_approve_with_stale_payable_is_422(client, dataset, labels):
    # The view sends the decision and payable it showed; a mismatch must not be recorded.
    label = label_for(labels, "covered", "MEDICAL")
    claim = submit(client, dataset, label)
    rec = claim["recommendation"]
    body = {
        "adjuster": "a",
        "action": "approve",
        "final_decision": rec["decision"],
        "final_payable": "1.23",
    }
    response = client.post(f"/v1/claims/{label['claim_id']}/adjuster-decision", json=body)
    assert response.status_code == 422
    body["final_payable"] = rec["payable_amount"]
    response = client.post(f"/v1/claims/{label['claim_id']}/adjuster-decision", json=body)
    assert response.status_code == 200
