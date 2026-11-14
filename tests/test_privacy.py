"""Full IC numbers must not reach responses, traces, errors, logs or the database."""

import json
import logging

import pytest
from fastapi.testclient import TestClient

from claims.api import create_app
from claims.intake import main as intake_main
from claims.pipeline import run_intake
from claims.providers import FakeProvider
from tests.conftest import first_label, load_packet


class EchoingProvider:
    """A misbehaving model that puts the document's IC into invalid answers."""

    name = "echo"

    def __init__(self, ic):
        self.ic = ic

    def complete(self, prompt):
        return json.dumps({"doc_type": f"unknown {self.ic}", "note": self.ic})


def test_failure_paths_do_not_leak_ic(dataset, labels, register, tmp_path, caplog):
    label = first_label(labels, "clean", "MEDICAL")
    ic = label["fields"]["participant_ic"]
    docs = load_packet(dataset, label["claim_id"])
    caplog.set_level(logging.DEBUG)

    claim = run_intake(label["claim_id"], docs, EchoingProvider(ic), register)
    assert claim.status == "failed"
    assert claim.trace[0].error  # an error was recorded ...
    assert ic not in claim.model_dump_json()  # ... without the IC
    assert ic not in caplog.text

    app = create_app(tmp_path / "leak.db", dataset / "policies.jsonl", EchoingProvider(ic))
    client = TestClient(app)
    body = {"claim_id": label["claim_id"], "documents": [d.model_dump() for d in docs]}
    assert ic not in client.post("/v1/claims", json=body).text
    assert ic not in client.get(f"/v1/claims/{label['claim_id']}").text
    assert ic.encode() not in (tmp_path / "leak.db").read_bytes()


def test_successful_path_does_not_store_or_log_ic(dataset, labels, tmp_path, capsys, caplog):
    label = first_label(labels, "clean", "MOTOR")
    ic = label["fields"]["participant_ic"]
    caplog.set_level(logging.DEBUG)
    db = tmp_path / "cli.db"
    packet = dataset / "claims" / label["claim_id"]

    code = intake_main(
        [str(packet), "--register", str(dataset / "policies.jsonl"), "--db", str(db), "--trace"]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert json.loads(out)["fields"]["participant_ic"] == "XXXXXX-XX-" + ic[-4:]
    assert ic not in out and ic not in caplog.text
    assert ic.encode() not in db.read_bytes()


SYNTHETIC_IC = "900101-00-1234"  # place code 00 is never issued


def test_422_does_not_echo_ic_used_as_property_name(dataset, tmp_path):
    client = TestClient(create_app(tmp_path / "t.db", dataset / "policies.jsonl"))
    body = {
        "claim_id": "C1",
        SYNTHETIC_IC: 1,
        "documents": [{"doc_id": "d1", "text": "x", SYNTHETIC_IC: {"nested": SYNTHETIC_IC}}],
    }
    response = client.post("/v1/claims", json=body)
    assert response.status_code == 422
    assert SYNTHETIC_IC not in response.text
    locs = [err["loc"] for err in response.json()["detail"]]
    assert ["body", "XXXXXX-XX-1234"] in locs  # structure is kept, the IC is masked
    assert ["body", "documents", 0, "XXXXXX-XX-1234"] in locs


@pytest.mark.parametrize("field", ["claim_id", "doc_id"])
def test_ic_shaped_ids_are_rejected(dataset, tmp_path, field):
    client = TestClient(create_app(tmp_path / "t.db", dataset / "policies.jsonl"))
    ids = {"claim_id": "C1", "doc_id": "d1", field: SYNTHETIC_IC.replace("-", "")}
    body = {"claim_id": ids["claim_id"], "documents": [{"doc_id": ids["doc_id"], "text": "x"}]}
    response = client.post("/v1/claims", json=body)
    assert response.status_code == 422
    assert SYNTHETIC_IC.replace("-", "") not in response.text


def test_404_does_not_echo_path(dataset, tmp_path):
    client = TestClient(create_app(tmp_path / "t.db", dataset / "policies.jsonl"))
    response = client.get(f"/v1/claims/{SYNTHETIC_IC}")
    assert response.status_code == 404
    assert SYNTHETIC_IC not in response.text


def test_cli_rejects_ic_shaped_folder_name(dataset, labels, tmp_path, capsys):
    label = first_label(labels, "clean", "MOTOR")
    packet = tmp_path / SYNTHETIC_IC
    packet.mkdir()
    for f in (dataset / "claims" / label["claim_id"]).glob("*.txt"):
        (packet / f.name).write_text(f.read_text())
    code = intake_main([str(packet), "--register", str(dataset / "policies.jsonl")])
    captured = capsys.readouterr()
    assert code == 1
    assert SYNTHETIC_IC not in captured.out + captured.err


class KeyEchoProvider(FakeProvider):
    """Answers with an extra key named after an IC, so validation errors carry it."""

    def __init__(self):
        super().__init__()
        self.prompts = []

    def complete(self, prompt):
        self.prompts.append(prompt)
        answer = json.loads(super().complete(prompt))
        answer[SYNTHETIC_IC] = "x"
        return json.dumps(answer)


def test_retry_feedback_masks_ic_in_error_locations(dataset, labels, register, caplog):
    label = first_label(labels, "clean", "MEDICAL")
    caplog.set_level(logging.DEBUG)
    provider = KeyEchoProvider()
    claim = run_intake(
        label["claim_id"], load_packet(dataset, label["claim_id"]), provider, register
    )
    assert claim.status == "failed"
    retry_prompts = [p for p in provider.prompts if "previous answer was rejected" in p]
    assert retry_prompts
    for prompt in retry_prompts:
        feedback = prompt.split("<<<DOCUMENT")[0]  # the document text itself may hold an IC
        assert SYNTHETIC_IC not in feedback
        assert "XXXXXX-XX-1234: extra_forbidden" in feedback
    assert SYNTHETIC_IC not in claim.model_dump_json() and SYNTHETIC_IC not in caplog.text


class IcInTextFieldsProvider(FakeProvider):
    """Valid answers, but the model copied an IC into free-text fields."""

    def complete(self, prompt):
        answer = json.loads(super().complete(prompt))
        for key in ("participant_name", "diagnosis", "cause_of_loss"):
            if key in answer:
                answer[key] = f"see {SYNTHETIC_IC}"
        return json.dumps(answer)


@pytest.mark.parametrize("product", ["MEDICAL", "MOTOR"])
def test_ic_in_other_text_fields_is_redacted(dataset, labels, tmp_path, product):
    label = first_label(labels, "clean", product)
    app = create_app(tmp_path / "t.db", dataset / "policies.jsonl", IcInTextFieldsProvider())
    client = TestClient(app)
    docs = load_packet(dataset, label["claim_id"])
    body = {"claim_id": label["claim_id"], "documents": [d.model_dump() for d in docs]}
    posted = client.post("/v1/claims", json=body)
    fetched = client.get(f"/v1/claims/{label['claim_id']}")
    assert posted.status_code == 201
    assert posted.json()["fields"]["participant_name"] == "see XXXXXX-XX-1234"
    for text in (posted.text, fetched.text):
        assert SYNTHETIC_IC not in text
    assert SYNTHETIC_IC.encode() not in (tmp_path / "t.db").read_bytes()


def test_access_log_filter_masks_ic_in_path(dataset, tmp_path):
    create_app(tmp_path / "t.db", dataset / "policies.jsonl")
    record = logging.LogRecord(
        "uvicorn.access",
        logging.INFO,
        __file__,
        0,
        '%s - "%s %s"',
        ("127.0.0.1", "GET", f"/v1/claims/{SYNTHETIC_IC}"),
        None,
    )
    for f in logging.getLogger("uvicorn.access").filters:
        f.filter(record)
    assert SYNTHETIC_IC not in record.getMessage()
    assert "XXXXXX-XX-1234" in record.getMessage()


@pytest.mark.parametrize(
    "ic",
    [
        "850101.14.5678",
        "850101.145678",
        "850101-14.5678",
        "850101_14_5678",
        "X850101145678",
        "8501 0114 5678",
        "850101 - 14 - 5678",
        "850101/14/5678",
        "850101\u201314\u20135678",
    ],
)
def test_dotted_ic_forms_are_redacted(ic):
    from claims.normalize import contains_ic, redact_ics

    text = f"Rationale mentions participant {ic} in passing."
    assert contains_ic(text)
    out = redact_ics(text)
    assert "XXXXXX-XX-5678" in out and "850101" not in out and "8501" not in out


def test_ic_shaped_claim_id_and_adjuster_are_rejected_or_masked(tmp_path, dataset):
    from fastapi.testclient import TestClient

    from claims.api import create_app
    from claims.providers import FakeProvider

    client = TestClient(create_app(tmp_path / "p.db", dataset / "policies.jsonl", FakeProvider()))
    body = {"claim_id": "CLM850101145678", "documents": [{"doc_id": "d1", "text": "SPECIMEN"}]}
    response = client.post("/v1/claims", json=body)
    assert response.status_code == 422 and "850101" not in response.text
