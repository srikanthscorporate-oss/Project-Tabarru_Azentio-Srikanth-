"""SQLite persistence for claims, documents, the step trace, recommendations and audit.

Only masked IC values are stored: the claim fields are masked during
assemble_claim, and document text is stored with IC numbers redacted.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path

from claims.normalize import redact_ics
from claims.schemas import AdjusterDecisionRecord, ClaimDetail, InputDocument

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
CREATE TABLE IF NOT EXISTS audit (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    actor        TEXT NOT NULL,
    claim_id     TEXT NOT NULL REFERENCES claims(claim_id),
    event        TEXT NOT NULL,
    decision     TEXT NOT NULL,
    payable      TEXT NOT NULL,
    clause_ids   TEXT NOT NULL,
    reason       TEXT
);
CREATE TRIGGER IF NOT EXISTS audit_no_update BEFORE UPDATE ON audit
BEGIN SELECT RAISE(ABORT, 'audit is append-only'); END;
CREATE TRIGGER IF NOT EXISTS audit_no_delete BEFORE DELETE ON audit
BEGIN SELECT RAISE(ABORT, 'audit is append-only'); END;
"""

# Columns added in Task 2. A Task 1 database gets them through ALTER TABLE ADD
# COLUMN, which keeps every existing row; old claims read back with no
# recommendation.
MIGRATIONS = {
    "claims": {
        "intake_status": "TEXT",
        "recommendation": "TEXT",
        "adjuster_decision": "TEXT",
    },
    "trace": {"clause_ids": "TEXT"},
}

RECOMMENDER = "coverage-agent"


class DuplicateClaimError(Exception):
    pass


class ClaimNotFoundError(Exception):
    pass


class AlreadyDecidedError(Exception):
    pass


def _audit_row(actor, claim_id, event, decision, payable, clause_ids, reason=None) -> tuple:
    return (actor, claim_id, event, decision, f"{payable:.2f}", json.dumps(clause_ids), reason)


_AUDIT_INSERT = (
    "INSERT INTO audit (actor, claim_id, event, decision, payable, clause_ids, reason)"
    " VALUES (?, ?, ?, ?, ?, ?, ?)"
)


class ClaimStore:
    def __init__(self, path: Path | str):
        self.path = str(path)
        with closing(self._connect()) as conn:
            conn.executescript(SCHEMA)
            for table, columns in MIGRATIONS.items():
                have = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
                for name, kind in columns.items():
                    if name not in have:
                        conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {kind}")
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10)
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
        rec = claim.recommendation
        with closing(self._connect()) as conn:
            try:
                with conn:
                    conn.execute(
                        "INSERT INTO claims (claim_id, status, product, fields, reason_codes,"
                        " intake_status, recommendation) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            claim.claim_id,
                            claim.status,
                            fields["product"],
                            json.dumps(fields),
                            json.dumps(claim.reason_codes),
                            claim.intake_status,
                            rec.model_dump_json() if rec else None,
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
                        "INSERT INTO trace (claim_id, seq, step, doc_ids, status, duration_ms,"
                        " provider_calls, error, clause_ids) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                                json.dumps(t.clause_ids),
                            )
                            for i, t in enumerate(claim.trace)
                        ],
                    )
                    if rec is not None:
                        conn.execute(
                            _AUDIT_INSERT,
                            _audit_row(
                                RECOMMENDER,
                                claim.claim_id,
                                "recommendation",
                                rec.decision.value,
                                rec.payable_amount,
                                [c.clause_id for c in rec.cited_clauses],
                            ),
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
            intake_status=row["intake_status"],
            recommendation=_load(row["recommendation"]),
            adjuster_decision=_load(row["adjuster_decision"]),
            trace=[
                {
                    **dict(t),
                    "doc_ids": json.loads(t["doc_ids"]),
                    "clause_ids": json.loads(t["clause_ids"] or "[]"),
                }
                for t in trace
            ],
        )

    def queue(self) -> list[dict]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT claim_id, product, recommendation FROM claims"
                " WHERE status = 'pending_adjuster' ORDER BY created_at, claim_id"
            ).fetchall()
        return [
            {
                "claim_id": r["claim_id"],
                "product": r["product"],
                "recommendation": json.loads(r["recommendation"]),
            }
            for r in rows
            if r["recommendation"]
        ]

    def decide(self, claim_id: str, record: AdjusterDecisionRecord) -> None:
        """Close a pending claim and write its audit row in one transaction.

        BEGIN IMMEDIATE takes the write lock first, and the UPDATE only matches
        a pending row, so two concurrent requests cannot both decide a claim.
        """
        conn = self._connect()
        conn.isolation_level = None  # explicit transaction control
        try:
            conn.execute("BEGIN IMMEDIATE")
            updated = conn.execute(
                "UPDATE claims SET status = 'decided', adjuster_decision = ?"
                " WHERE claim_id = ? AND status = 'pending_adjuster'",
                (record.model_dump_json(), claim_id),
            ).rowcount
            if updated == 0:
                exists = conn.execute(
                    "SELECT 1 FROM claims WHERE claim_id = ?", (claim_id,)
                ).fetchone()
                conn.execute("ROLLBACK")
                raise AlreadyDecidedError(claim_id) if exists else ClaimNotFoundError(claim_id)
            rec = json.loads(
                conn.execute(
                    "SELECT recommendation FROM claims WHERE claim_id = ?", (claim_id,)
                ).fetchone()["recommendation"]
            )
            conn.execute(
                _AUDIT_INSERT,
                _audit_row(
                    record.adjuster,
                    claim_id,
                    f"adjuster_{record.action}",
                    record.final_decision.value,
                    record.final_payable,
                    [c["clause_id"] for c in rec["cited_clauses"]],
                    record.reason,
                ),
            )
            conn.execute("COMMIT")
        except (AlreadyDecidedError, ClaimNotFoundError):
            raise
        except Exception:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    def audit(self, claim_id: str | None = None) -> list[dict]:
        query, args = "SELECT * FROM audit", ()
        if claim_id is not None:
            query, args = query + " WHERE claim_id = ?", (claim_id,)
        with closing(self._connect()) as conn:
            rows = conn.execute(query + " ORDER BY id", args).fetchall()
        return [{**dict(r), "clause_ids": json.loads(r["clause_ids"])} for r in rows]


def _load(value: str | None):
    return json.loads(value) if value else None
