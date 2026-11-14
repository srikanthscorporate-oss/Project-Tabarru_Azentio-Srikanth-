import json

import pytest

from claims.intake import main as intake_main
from claims.pipeline import run_intake
from claims.providers import (
    FakeProvider,
    OpenAICompatibleProvider,
    ProviderError,
    _strip_code_fence,
    get_provider,
)
from tests.conftest import first_label, load_packet


def run_cli(dataset, claim_id, *extra):
    return intake_main(
        [str(dataset / "claims" / claim_id), "--register", str(dataset / "policies.jsonl"), *extra]
    )


@pytest.mark.parametrize("product", ["MEDICAL", "MOTOR"])
def test_cli_prints_response_json(dataset, labels, capsys, product):
    label = first_label(labels, "clean", product)
    assert run_cli(dataset, label["claim_id"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert set(out) == {
        "claim_id",
        "status",
        "documents",
        "fields",
        "reason_codes",
        "intake_status",
        "recommendation",
    }
    assert out["claim_id"] == label["claim_id"]
    assert out["fields"]["product"] == product


def test_cli_duplicate_store_exits_2(dataset, labels, tmp_path):
    label = first_label(labels, "clean", "MEDICAL")
    db = str(tmp_path / "x.db")
    assert run_cli(dataset, label["claim_id"], "--db", db) == 0
    assert run_cli(dataset, label["claim_id"], "--db", db) == 2


def test_cli_missing_folder_exits_1(tmp_path):
    assert intake_main([str(tmp_path / "nothing")]) == 1


def test_default_provider_is_fake():
    assert isinstance(get_provider(), FakeProvider)


def test_real_provider_needs_explicit_config(monkeypatch):
    monkeypatch.setenv("CLAIMS_PROVIDER", "openai")
    with pytest.raises(ValueError):
        get_provider()
    monkeypatch.setenv("CLAIMS_LLM_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setenv("CLAIMS_LLM_MODEL", "test-model")
    assert isinstance(get_provider(), OpenAICompatibleProvider)


def test_unknown_provider_rejected(monkeypatch):
    monkeypatch.setenv("CLAIMS_PROVIDER", "magic")
    with pytest.raises(ValueError):
        get_provider()


def test_unreachable_real_provider_fails_documents_without_fallback(dataset, labels, register):
    provider = OpenAICompatibleProvider("http://127.0.0.1:9/v1", "m", None, 1)
    with pytest.raises(ProviderError):
        provider.complete("hello")
    label = first_label(labels, "clean", "MOTOR")
    claim = run_intake(
        label["claim_id"], load_packet(dataset, label["claim_id"]), provider, register
    )
    assert claim.status == "failed"
    assert all(d.doc_type is None for d in claim.documents)


def test_code_fence_is_stripped():
    assert _strip_code_fence('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert _strip_code_fence('{"a": 1}') == '{"a": 1}'


@pytest.mark.parametrize("scenario", ["clean", "amount_mismatch", "missing_document"])
def test_cli_output_equals_api_post_response(dataset, labels, capsys, tmp_path, scenario):
    from fastapi.testclient import TestClient

    from claims.api import create_app

    label = first_label(labels, scenario)
    assert run_cli(dataset, label["claim_id"]) == 0
    cli_out = json.loads(capsys.readouterr().out)

    client = TestClient(create_app(tmp_path / "p.db", dataset / "policies.jsonl"))
    docs = load_packet(dataset, label["claim_id"])
    body = {"claim_id": label["claim_id"], "documents": [d.model_dump() for d in docs]}
    assert client.post("/v1/claims", json=body).json() == cli_out
