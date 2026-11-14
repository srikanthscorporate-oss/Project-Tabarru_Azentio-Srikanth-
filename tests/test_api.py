import pytest
from fastapi.testclient import TestClient

from claims.api import create_app
from claims.providers import FakeProvider
from tests.conftest import first_label, load_packet

STEPS = ["classify_documents", "extract_fields", "validate", "assemble_claim", "coverage"]
TRACE_KEYS = {"step", "doc_ids", "status", "duration_ms", "provider_calls", "error", "clause_ids"}


@pytest.fixture
def client(dataset, tmp_path):
    app = create_app(tmp_path / "test.db", dataset / "policies.jsonl", FakeProvider())
    return TestClient(app)


def body_for(dataset, label):
    docs = load_packet(dataset, label["claim_id"])
    return {"claim_id": label["claim_id"], "documents": [d.model_dump() for d in docs]}


def test_post_then_get(client, dataset, labels):
    label = first_label(labels, "clean", "MEDICAL")
    full_ic = label["fields"]["participant_ic"]

    posted = client.post("/v1/claims", json=body_for(dataset, label))
    assert posted.status_code == 201
    claim = posted.json()
    assert set(claim) == {
        "claim_id",
        "status",
        "documents",
        "fields",
        "reason_codes",
        "intake_status",
        "recommendation",
    }
    # Task 2: workflow status moves on, the intake result is kept separately.
    assert claim["status"] == "pending_adjuster"
    assert claim["intake_status"] == "intake_complete"
    assert claim["recommendation"]["decision"] == label["expected_decision"]
    assert all(set(d) == {"doc_id", "doc_type", "status"} for d in claim["documents"])
    assert claim["fields"]["participant_ic"] == "XXXXXX-XX-" + full_ic[-4:]

    fetched = client.get(f"/v1/claims/{label['claim_id']}")
    assert fetched.status_code == 200
    detail = fetched.json()
    assert {k: v for k, v in detail.items() if k not in ("trace", "adjuster_decision")} == claim
    assert detail["adjuster_decision"] is None
    assert detail["trace"][-1]["clause_ids"]  # coverage retrieved clauses
    assert [t["step"] for t in detail["trace"]] == STEPS
    assert all(set(t) == TRACE_KEYS for t in detail["trace"])
    assert detail["trace"][0]["provider_calls"] == len(claim["documents"])
    assert full_ic not in posted.text and full_ic not in fetched.text


def test_flagged_claim_round_trip(client, dataset, labels):
    label = first_label(labels, "amount_mismatch")
    client.post("/v1/claims", json=body_for(dataset, label))
    detail = client.get(f"/v1/claims/{label['claim_id']}").json()
    assert detail["status"] == "pending_adjuster"
    assert detail["intake_status"] == "flagged"
    assert "AMOUNT_MISMATCH" in detail["reason_codes"]


def test_duplicate_submission_is_409(client, dataset, labels):
    body = body_for(dataset, first_label(labels, "clean", "MOTOR"))
    assert client.post("/v1/claims", json=body).status_code == 201
    assert client.post("/v1/claims", json=body).status_code == 409


def test_unknown_claim_is_404(client):
    assert client.get("/v1/claims/NOPE-1").status_code == 404


@pytest.mark.parametrize(
    "body",
    [
        {"claim_id": "C1"},
        {"claim_id": "C1", "documents": []},
        {"claim_id": "bad id!", "documents": [{"doc_id": "d1", "text": "x"}]},
        {"claim_id": "C1", "documents": [{"doc_id": "d1", "text": "x"}] * 2},
        {"claim_id": "C1", "documents": [{"doc_id": "d1", "text": "x", "extra": 1}]},
    ],
)
def test_invalid_bodies_are_422(client, body):
    assert client.post("/v1/claims", json=body).status_code == 422


def test_422_does_not_echo_document_text(client):
    ic = "880202-00-4321"
    body = {"claim_id": "bad id!", "documents": [{"doc_id": "d1", "text": f"IC No: {ic}"}]}
    response = client.post("/v1/claims", json=body)
    assert response.status_code == 422
    assert ic not in response.text and "IC No" not in response.text
