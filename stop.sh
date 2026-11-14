#!/usr/bin/env bash
# Stop everything this project starts: the local server from ./start.sh and,
# if it is up, the docker compose stack. Data is kept (the demo database, the
# generated dataset and the Docker volume are not removed).
set -uo pipefail
cd "$(dirname "$0")"

PID_FILE="${CLAIMS_DATA:-data}/.tabarru.pid"
if [ -f "$PID_FILE" ]; then
    pid="$(cat "$PID_FILE")"
    if kill -0 "$pid" 2>/dev/null && ps -p "$pid" -o command= | grep -q "claims.api:create_app"; then
        kill "$pid"
        for _ in $(seq 1 20); do kill -0 "$pid" 2>/dev/null || break; sleep 0.25; done
        kill -0 "$pid" 2>/dev/null && kill -9 "$pid"
        echo "stopped local server (pid $pid)"
    else
        echo "local server not running"
    fi
    rm -f "$PID_FILE"
else
    echo "local server not running"
fi

if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
    if [ -n "$(docker compose ps -q 2>/dev/null)" ]; then
        docker compose down
        echo "stopped docker compose stack (volume kept)"
    else
        echo "docker compose stack not running"
    fi
fi
