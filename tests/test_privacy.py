"""Full IC numbers must not reach responses, traces, errors, logs or the database."""

import json
import logging

from fastapi.testclient import TestClient

from claims.api import create_app
from claims.intake import main as intake_main
from claims.pipeline import run_intake
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

    code = intake_main([str(packet), "--register", str(dataset / "policies.jsonl"), "--db", str(db), "--trace"])
    out = capsys.readouterr().out
    assert code == 0
    assert json.loads(out)["fields"]["participant_ic"] == "XXXXXX-XX-" + ic[-4:]
    assert ic not in out and ic not in caplog.text
    assert ic.encode() not in db.read_bytes()
