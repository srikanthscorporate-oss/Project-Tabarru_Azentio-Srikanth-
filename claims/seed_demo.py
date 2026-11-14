"""Fill a demo database by sending generated packets through the real API.

    python -m claims.seed_demo --data data/ --db data/demo.db

Every packet in <data>/claims/ is POSTed to an in-process app that always uses
FakeProvider, so seeding never calls a real model. Claims already in the
database are skipped (the API answers 409), so the command is safe to repeat
and never deletes or overwrites existing rows. The dataset itself is only read;
generate it first with python -m claims.generate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from claims.api import create_app
from claims.classifier import ClassifierConfig
from claims.providers import FakeProvider


def seed(data_dir: Path, db_path: Path) -> dict[str, int]:
    packets = sorted(p for p in (data_dir / "claims").iterdir() if p.is_dir())
    app = create_app(
        db_path, data_dir / "policies.jsonl", FakeProvider(), data_dir, ClassifierConfig("llm")
    )
    counts = {"created": 0, "skipped": 0}
    with TestClient(app) as client:
        for folder in packets:
            docs = [
                {"doc_id": p.stem, "text": p.read_text(encoding="utf-8")}
                for p in sorted(folder.glob("*.txt"))
            ]
            response = client.post("/v1/claims", json={"claim_id": folder.name, "documents": docs})
            if response.status_code == 201:
                counts["created"] += 1
            elif response.status_code == 409:
                counts["skipped"] += 1
            else:
                raise RuntimeError(f"{folder.name}: HTTP {response.status_code}")
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
