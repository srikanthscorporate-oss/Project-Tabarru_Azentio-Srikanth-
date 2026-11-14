import json
import os
import socket
from pathlib import Path

import pytest

from claims.generate import write_dataset
from claims.schemas import InputDocument
from claims.validation import load_register


class NetworkBlocked(RuntimeError):
    pass


@pytest.fixture(autouse=True)
def offline_and_clean_env(monkeypatch):
    """Every test runs with no CLAIMS_* variables and no outbound network."""
    for key in list(os.environ):
        if key.startswith("CLAIMS_"):
            monkeypatch.delenv(key)

    def refuse(*args, **kwargs):
        raise NetworkBlocked("network access is disabled in tests")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)


@pytest.fixture(scope="session")
def dataset(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("data")
    write_dataset(40, 7, out)
    return out


@pytest.fixture(scope="session")
def labels(dataset) -> list[dict]:
    return [json.loads(line) for line in (dataset / "labels.jsonl").read_text().splitlines()]


@pytest.fixture(scope="session")
def register(dataset) -> dict:
    return load_register(dataset / "policies.jsonl")


def load_packet(dataset: Path, claim_id: str) -> list[InputDocument]:
    folder = dataset / "claims" / claim_id
    return [InputDocument(doc_id=p.stem, text=p.read_text()) for p in sorted(folder.glob("*.txt"))]


def first_label(labels: list[dict], scenario: str, product: str | None = None) -> dict:
    return next(
        lab
        for lab in labels
        if lab["scenario"] == scenario and (product is None or lab["product"] == product)
    )
