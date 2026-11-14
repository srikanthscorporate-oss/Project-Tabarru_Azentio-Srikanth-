"""Fill a demo database by running generated packets through the real pipeline.

    python -m claims.seed_demo --data data/ --db data/demo.db

Every packet in <data>/claims/ goes through the same steps as POST /v1/claims
(assess_claim, then ClaimStore.save) with FakeProvider fixed, so seeding never
calls a real model. Claims already in the database are skipped, so the command
is safe to repeat and never deletes or overwrites existing rows. The dataset itself is only read;
generate it first with python -m claims.generate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from claims.classifier import ClassifierConfig
from claims.coverage import CoverageContext, assess_claim
from claims.providers import FakeProvider
from claims.schemas import InputDocument
from claims.storage import ClaimStore
from claims.validation import load_register


def seed(data_dir: Path, db_path: Path) -> dict[str, int]:
    packets = sorted(p for p in (data_dir / "claims").iterdir() if p.is_dir())
    store = ClaimStore(db_path)
    ctx = CoverageContext.load(data_dir, load_register(data_dir / "policies.jsonl"))
    provider, classifier = FakeProvider(), ClassifierConfig("llm")
    counts = {"created": 0, "skipped": 0}
    for folder in packets:
        if store.exists(folder.name):
            counts["skipped"] += 1
            continue
        docs = [
            InputDocument(doc_id=p.stem, text=p.read_text(encoding="utf-8"))
            for p in sorted(folder.glob("*.txt"))
        ]
        claim = assess_claim(folder.name, docs, provider, ctx, classifier)
        store.save(claim, docs)
        counts["created"] += 1
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--db", type=Path, default=Path("data/demo.db"))
    args = parser.parse_args(argv)
    if not (args.data / "claims").is_dir() or not (args.data / "policies.jsonl").is_file():
        print("error: no dataset found; run python -m claims.generate first", file=sys.stderr)
        return 1
    counts = seed(args.data, args.db)
    print(f"seeded {args.db}: {counts['created']} created, {counts['skipped']} already present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
