"""Clause retrieval over the certificate wording with BM25 (stdlib only).

    retrieve(query, product, k) -> list[Clause]

Clauses are filtered to the product before ranking, so a motor query can never
return a medical clause. Ties are broken by clause id so results are stable.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from claims.schemas import ClaimFields

DEFAULT_K = 6
K1, B = 1.5, 0.75
_CLAUSE_LINE = re.compile(r"^((?:MED|MTR)-\d+\.\d+) (.+)$", re.M)
_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "we",
        "while",
        "with",
        "not",
        "any",
        "our",
        "you",
        "your",
    ]
)


@dataclass(frozen=True)
class Clause:
    clause_id: str
    text: str
    score: float = 0.0


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS]


def parse_certificate(text: str) -> list[Clause]:
    return [Clause(m.group(1), m.group(2).strip()) for m in _CLAUSE_LINE.finditer(text)]


class ClauseIndex:
    def __init__(self, certificates: dict[str, list[Clause]]):
        self.clauses = certificates
        self.tokens = {
            product: [tokenize(c.text) for c in clauses]
            for product, clauses in certificates.items()
        }

    @classmethod
    def load(cls, folder: Path | str) -> ClauseIndex:
        folder = Path(folder)
        return cls(
            {
                product: parse_certificate((folder / f"{product}.md").read_text(encoding="utf-8"))
                for product in ("MEDICAL", "MOTOR")
            }
        )

    def retrieve(self, query: str, product: str, k: int = DEFAULT_K) -> list[Clause]:
        clauses = self.clauses.get(product, [])
        docs = self.tokens.get(product, [])
        if not clauses or k <= 0:
            return []
        avg_len = sum(len(d) for d in docs) / len(docs)
        df = Counter(term for d in docs for term in set(d))
        query_terms = tokenize(query)
        scored = []
        for clause, doc in zip(clauses, docs, strict=True):
            tf = Counter(doc)
            score = 0.0
            for term in query_terms:
                if term not in tf:
                    continue
                idf = math.log(1 + (len(docs) - df[term] + 0.5) / (df[term] + 0.5))
                norm = tf[term] * (K1 + 1) / (tf[term] + K1 * (1 - B + B * len(doc) / avg_len))
                score += idf * norm
            scored.append(Clause(clause.clause_id, clause.text, round(score, 6)))
        scored.sort(key=lambda c: (-c.score, c.clause_id))
        return scored[:k]


# Terms every coverage query carries: they name the questions each product
# always has to answer (is the loss inside the period, which benefit applies,
# which limit or excess reduces it).
BASE_TERMS = {
    "MEDICAL": "period of cover start date hospitalisation benefit room and board per day",
    "MOTOR": "period of cover start date own damage repair excess",
}


def build_query(fields: ClaimFields, cert_start: date | None) -> str:
    """Query from extracted claim fields only: product base terms plus claim text.

    Diagnosis and cause of loss are the free-text facts that point at specific
    exclusions or perils. A medical incident within 60 days of the certificate
    start adds waiting-period terms, because the waiting rule then needs checking.
    """
    product = fields.product.value if fields.product else ""
    parts = [BASE_TERMS.get(product, "")]
    if fields.diagnosis:
        parts.append(fields.diagnosis)
    if fields.cause_of_loss:
        parts.append(fields.cause_of_loss)
    if (
        product == "MEDICAL"
        and cert_start
        and fields.incident_date
        and (fields.incident_date - cert_start).days < 60
    ):
        parts.append("waiting period illness")
    return " ".join(p for p in parts if p)
