"""SQLite persistence for claims, their documents and the step trace.

Only masked IC values are stored: the claim fields are masked during
assemble_claim, and document text is stored with IC numbers redacted.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path

from claims.normalize import redact_ics
from claims.schemas import ClaimDetail, InputDocument

SCHEMA = """
CREATE TABLE IF NOT EXISTS claims (
    claim_id     TEXT PRIMARY KEY,
    status       TEXT NOT NULL,
    product      TEXT,
    fields       TEXT NOT NULL,
    reason_codes TEXT NOT NULL,
    created_at   TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS documents (
    claim_id   TEXT NOT NULL REFERENCES claims(claim_id),
    position   INTEGER NOT NULL,
    doc_id     TEXT NOT NULL,
    doc_type   TEXT,
    status     TEXT NOT NULL,
    text       TEXT NOT NULL,
    PRIMARY KEY (claim_id, doc_id)
);
CREATE TABLE IF NOT EXISTS trace (
    claim_id       TEXT NOT NULL REFERENCES claims(claim_id),
    seq            INTEGER NOT NULL,
    step           TEXT NOT NULL,
    doc_ids        TEXT NOT NULL,
    status         TEXT NOT NULL,
    duration_ms    REAL NOT NULL,
    provider_calls INTEGER NOT NULL,
    error          TEXT,
    PRIMARY KEY (claim_id, seq)
);
"""


class DuplicateClaimError(Exception):
    pass


class ClaimStore:
    def __init__(self, path: Path | str):
        self.path = str(path)
        with closing(self._connect()) as conn:
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def exists(self, claim_id: str) -> bool:
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT 1 FROM claims WHERE claim_id = ?", (claim_id,)).fetchone()
        return row is not None

    def save(self, claim: ClaimDetail, documents: list[InputDocument]) -> None:
        """Insert a claim, its documents and trace in one transaction."""
        texts = {d.doc_id: d.text for d in documents}
        fields = claim.fields.model_dump(mode="json")
        with closing(self._connect()) as conn:
            try:
                with conn:
                    conn.execute(
                        "INSERT INTO claims (claim_id, status, product, fields, reason_codes)"
                        " VALUES (?, ?, ?, ?, ?)",
                        (
                            claim.claim_id,
                            claim.status,
                            fields["product"],
                            json.dumps(fields),
                            json.dumps(claim.reason_codes),
                        ),
                    )
                    conn.executemany(
                        "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?)",
                        [
                            (
                                claim.claim_id,
                                i,
                                d.doc_id,
                                d.doc_type.value if d.doc_type else None,
                                d.status,
                                redact_ics(texts[d.doc_id]),
                            )
                            for i, d in enumerate(claim.documents)
                        ],
                    )
                    conn.executemany(
                        "INSERT INTO trace VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        [
                            (
                                claim.claim_id,
                                i,
                                t.step,
                                json.dumps(t.doc_ids),
                                t.status,
                                t.duration_ms,
                                t.provider_calls,
                                t.error,
                            )
                            for i, t in enumerate(claim.trace)
                        ],
                    )
            except sqlite3.IntegrityError as exc:
                raise DuplicateClaimError(claim.claim_id) from exc

    def get(self, claim_id: str) -> ClaimDetail | None:
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT * FROM claims WHERE claim_id = ?", (claim_id,)).fetchone()
            if row is None:
                return None
            docs = conn.execute(
                "SELECT doc_id, doc_type, status FROM documents WHERE claim_id = ? ORDER BY position",
                (claim_id,),
            ).fetchall()
            trace = conn.execute(
                "SELECT * FROM trace WHERE claim_id = ? ORDER BY seq", (claim_id,)
            ).fetchall()
        return ClaimDetail(
            claim_id=row["claim_id"],
            status=row["status"],
            documents=[dict(d) for d in docs],
            fields=json.loads(row["fields"]),
            reason_codes=json.loads(row["reason_codes"]),
            trace=[{**dict(t), "doc_ids": json.loads(t["doc_ids"])} for t in trace],
        )
