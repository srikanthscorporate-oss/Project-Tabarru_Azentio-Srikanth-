"""Stretch goals: adjuster view, evidence and audit endpoints, /metrics, demo seeding."""

import re
import sqlite3

import pytest
from fastapi.testclient import TestClient

from claims.api import create_app
from claims.providers import FakeProvider
from claims.seed_demo import seed
from tests.test_adjuster import label_for, submit


@pytest.fixture
def db(tmp_path):
    return tmp_path / "stretch.db"


@pytest.fixture
def client(dataset, db):
    return TestClient(create_app(db, dataset / "policies.jsonl", FakeProvider()))


def metric(text: str, name: str, labels: str) -> float:
    match = re.search(rf"^{re.escape(name)}\{{{re.escape(labels)}\}} (\S+)$", text, re.M)
    assert match, f"{name}{{{labels}}} missing"
    return float(match.group(1))


def test_root_serves_the_adjuster_view_and_docs_still_work(client):
    page = client.get("/")
    assert page.status_code == 200 and "Tabarru adjuster review" in page.text
    assert "Specimen data" in page.text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/docs").status_code == 200
    # No remote scripts, fonts or styles: the view works offline.
    for name in ("index.html", "app.css", "app.js"):
        body = page.text if name == "index.html" else client.get(f"/static/{name}").text
        assert "http://" not in body and "https://" not in body


def test_view_never_uses_innerhtml():
    from claims.api import WEB_DIR

    source = (WEB_DIR / "app.js").read_text()
    assert not re.search(r"\.(innerHTML|outerHTML)\b|insertAdjacentHTML|document\.write", source)


def test_static_mount_refuses_path_traversal(client):
    assert client.get("/static/../api.py").status_code == 404
    assert client.get("/static/%2e%2e/api.py").status_code == 404


def test_cited_clause_source_contains_the_quote(client, dataset, labels):
    label = label_for(labels, "partial", "MOTOR")
    claim = submit(client, dataset, label)
    cited = claim["recommendation"]["cited_clauses"]
    assert cited
    for c in cited:
        response = client.get(f"/v1/certificates/MOTOR/clauses/{c['clause_id']}")
        assert response.status_code == 200
        body = response.json()
        assert body["wording"] == "current" and c["quote"] in body["text"]


@pytest.mark.parametrize(
    "path,status",
    [
        ("/v1/certificates/MOTOR/clauses/NOPE-1", 404),
        ("/v1/certificates/LIFE/clauses/MTR-5.1", 422),
        ("/v1/certificates/MEDICAL/clauses/MTR-5.1", 404),  # clause of another product
        ("/v1/certificates/MOTOR/clauses/..%2F..%2Fpolicies.jsonl", 404),
    ],
)
def test_clause_lookup_rejects_unknown_input(client, path, status):
    assert client.get(path).status_code == status


def test_audit_endpoint_lists_assessment_and_decision(client, dataset, labels):
    label = label_for(labels, "covered", "MEDICAL")
    submit(client, dataset, label)
    cid = label["claim_id"]
    client.post(f"/v1/claims/{cid}/adjuster-decision", json={"adjuster": "a1", "action": "approve"})
    events = client.get(f"/v1/claims/{cid}/audit").json()
    assert [e["event"] for e in events][-1] == "adjuster_approve"
    assert events[-1]["actor"] == "a1"
    assert client.get("/v1/claims/NOPE/audit").status_code == 404


def test_audit_endpoint_is_read_only(client, dataset, labels):
    label = label_for(labels, "covered", "MEDICAL")
    submit(client, dataset, label)
    path = f"/v1/claims/{label['claim_id']}/audit"
    assert client.delete(path).status_code == 405
    assert client.put(path, json={}).status_code == 405


def test_metrics_shape_types_and_status_gauges(client, dataset, labels):
    submit(client, dataset, label_for(labels, "covered", "MEDICAL"))
    text = client.get("/metrics").text
    assert "# TYPE tabarru_claims gauge" in text
    assert "# TYPE tabarru_provider_retries_total counter" in text
    assert "# TYPE tabarru_citation_invalid_total counter" in text
    assert "# TYPE tabarru_step_duration_seconds histogram" in text
    assert metric(text, "tabarru_claims", 'status="pending_adjuster"') == 1
    assert metric(text, "tabarru_step_duration_seconds_count", 'step="coverage"') >= 1


def test_metrics_scrape_does_not_change_business_metrics(client, dataset, labels):
    submit(client, dataset, label_for(labels, "covered", "MEDICAL"))
    first = client.get("/metrics").text
    client.get("/v1/adjuster/queue")
    assert client.get("/metrics").text == first


def test_retries_count_second_attempts_only(dataset, db, labels):
    label = label_for(labels, "covered", "MEDICAL")
    # Call 1 (first document's classification) is malformed: exactly one retry.
    provider = FakeProvider(faults={1: "malformed_json"})
    client = TestClient(create_app(db, dataset / "policies.jsonl", provider))
    before = metric(
        client.get("/metrics").text, "tabarru_provider_retries_total", 'step="classify_documents"'
    )
    submit(client, dataset, label)
    text = client.get("/metrics").text
    after = metric(text, "tabarru_provider_retries_total", 'step="classify_documents"')
    assert after - before == 1


def test_metrics_labels_never_carry_ids_or_ics(client, dataset, labels):
    label = label_for(labels, "covered", "MEDICAL")
    submit(client, dataset, label)
    text = client.get("/metrics").text
    assert label["claim_id"] not in text
    assert not re.search(r"\d{6}-?\d{2}-?\d{4}", text)
    allowed = {"status", "step", "product", "le"}
    for key in re.findall(r"[{,](\w+)=", text):
        assert key in allowed


def test_seed_demo_is_repeatable_and_keeps_existing_rows(dataset, tmp_path):
    db = tmp_path / "demo.db"
    first = seed(dataset, db)
    assert first["created"] == len(list((dataset / "claims").iterdir()))
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE claims SET status = 'decided' WHERE claim_id = 'CLM-0001'")
    second = seed(dataset, db)
    assert second == {"created": 0, "skipped": first["created"]}
    with sqlite3.connect(db) as conn:
        status = conn.execute("SELECT status FROM claims WHERE claim_id = 'CLM-0001'").fetchone()
    assert status[0] == "decided"
