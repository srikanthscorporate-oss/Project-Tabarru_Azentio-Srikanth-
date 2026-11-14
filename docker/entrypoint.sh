#!/bin/sh
# Generate the synthetic dataset into the volume only when it is absent, so an
# existing volume (and its database) is never overwritten. Demo seeding is a
# separate explicit command (see README).
set -eu
if [ ! -f /data/policies.jsonl ]; then
    echo "no dataset in /data: generating 40 synthetic SPECIMEN claims (seed 7)"
    python -m claims.generate --claims 40 --seed 7 --out /data
fi
exec "$@"
