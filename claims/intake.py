"""CLI: run one claim packet folder through the intake pipeline.

    python -m claims.intake data/claims/CLM-0001/ [--register PATH] [--db PATH] [--trace]

The claim_id is the folder name and each *.txt file is one document (doc_id is
the file stem; the name is never used as a type hint). Prints the same JSON as
POST /v1/claims. The folder and files must pass the same validation as
the API request. With --db the claim is also stored; an existing claim_id
exits with status 2. Exit status 1 means the folder could not be read.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from pydantic import ValidationError

from claims.normalize import redact_ics
from claims.pipeline import run_intake
from claims.providers import get_provider
from claims.schemas import ClaimRequest, ClaimResponse
from claims.storage import ClaimStore, DuplicateClaimError
from claims.validation import load_register


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Process one claim packet folder.")
    parser.add_argument("packet", type=Path, help="folder holding the packet's .txt documents")
    parser.add_argument(
        "--register",
        type=Path,
        default=Path(os.environ.get("CLAIMS_REGISTER", "data/policies.jsonl")),
    )
    parser.add_argument("--db", type=Path, help="also store the claim in this SQLite file")
    parser.add_argument("--trace", action="store_true", help="include the step trace")
    args = parser.parse_args(argv)

    files = sorted(args.packet.glob("*.txt")) if args.packet.is_dir() else []
    if not files:
        print(f"error: no .txt documents found in {redact_ics(str(args.packet))}", file=sys.stderr)
        return 1
    try:
        register = load_register(args.register)
    except FileNotFoundError:
        print(
            f"error: certificate register not found: {redact_ics(str(args.register))}",
            file=sys.stderr,
        )
        return 1

    # Same request model as POST /v1/claims, so the CLI accepts exactly what the API accepts.
    try:
        request = ClaimRequest(
            claim_id=args.packet.resolve().name,
            documents=[{"doc_id": f.stem, "text": f.read_text(encoding="utf-8")} for f in files],
        )
    except ValidationError:
        print(
            "error: folder name, file names or contents are not acceptable"
            " (same rules as POST /v1/claims)",
            file=sys.stderr,
        )
        return 1
    claim = run_intake(request.claim_id, request.documents, get_provider(), register)
    if args.db:
        try:
            ClaimStore(args.db).save(claim, request.documents)
        except DuplicateClaimError:
            print("error: claim already stored in the database", file=sys.stderr)
            return 2

    output = claim if args.trace else ClaimResponse.model_validate(claim.model_dump())
    print(json.dumps(output.model_dump(mode="json"), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
