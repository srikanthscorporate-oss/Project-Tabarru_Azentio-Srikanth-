#!/usr/bin/env bash
# Start Project Tabarru locally: one FastAPI process serves the API and the
# adjuster view (frontend) at http://localhost:$PORT/.
#
#   ./start.sh            # port 8000, demo database data/demo.db
#   PORT=9000 ./start.sh
#
# Steps, each skipped when already done (nothing is deleted or overwritten):
#   1. install locked dependencies with uv
#   2. generate the synthetic SPECIMEN dataset (40 claims, seed 7) if data/ is empty
#   3. train the Task 3 classifier only if CLAIMS_CLASSIFIER is trained or hybrid
#      and models/doc_classifier.joblib is missing
#   4. seed the demo database through the real pipeline (FakeProvider, repeatable)
#   5. serve the API and the adjuster view (stop with ./stop.sh or Ctrl+C)
set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-8000}"
DATA_DIR="${CLAIMS_DATA:-data}"
export CLAIMS_DB="${CLAIMS_DB:-$DATA_DIR/demo.db}"
export CLAIMS_REGISTER="${CLAIMS_REGISTER:-$DATA_DIR/policies.jsonl}"
export CLAIMS_DATA="$DATA_DIR"
export CLAIMS_PROVIDER="${CLAIMS_PROVIDER:-fake}"

if ! command -v uv >/dev/null 2>&1; then
    echo "error: uv is required (https://docs.astral.sh/uv/)" >&2
    exit 1
fi

echo "[1/5] installing locked dependencies"
uv sync --frozen --quiet

if [ ! -f "$DATA_DIR/policies.jsonl" ]; then
    echo "[2/5] generating synthetic dataset into $DATA_DIR/"
    uv run python -m claims.generate --claims 40 --seed 7 --out "$DATA_DIR/"
else
    echo "[2/5] dataset found in $DATA_DIR/"
fi

case "${CLAIMS_CLASSIFIER:-llm}" in
    trained|hybrid)
        if [ ! -f "${CLAIMS_CLASSIFIER_MODEL:-models/doc_classifier.joblib}" ]; then
            echo "[3/5] training the document classifier"
            uv run python -m claims.train_classifier
        else
            echo "[3/5] classifier model found"
        fi
        ;;
    *) echo "[3/5] classifier mode llm: no model needed" ;;
esac

echo "[4/5] seeding $CLAIMS_DB (existing claims are kept)"
uv run python -m claims.seed_demo --data "$DATA_DIR" --db "$CLAIMS_DB"

PID_FILE="$DATA_DIR/.tabarru.pid"
if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "error: already running (pid $(cat "$PID_FILE")); run ./stop.sh first" >&2
    exit 1
fi
echo $$ > "$PID_FILE"   # exec below keeps this pid for the server; stop.sh reads it

echo "[5/5] serving on http://localhost:$PORT/  (API docs /docs, metrics /metrics, provider $CLAIMS_PROVIDER)"
exec uv run uvicorn --factory claims.api:create_app --host 127.0.0.1 --port "$PORT"
