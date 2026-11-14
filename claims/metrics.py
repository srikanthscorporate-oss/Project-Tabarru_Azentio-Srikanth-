"""Prometheus text exposition for GET /metrics, with no extra dependency.

Counters and the latency histogram live in this process and reset on restart.
They are updated only when a claim is processed (POST /v1/claims, the eval or
the CLI in their own processes), never by a GET, so scraping is read-only.
Gauges for claims by status are read from SQLite at scrape time.

Labels are fixed vocabularies only (step name, product, status). Claim ids,
names, ICs, document text and error strings are never labels.
"""

from __future__ import annotations

import threading
from collections import defaultdict

STEPS = ("classify_documents", "extract_fields", "validate", "assemble_claim", "coverage")
STATUSES = ("intake_complete", "flagged", "failed", "pending_adjuster", "decided")
PRODUCTS = ("MEDICAL", "MOTOR", "UNKNOWN")
# Seconds. Fake-provider steps take well under a millisecond; real models take seconds.
BUCKETS = (0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 30.0)

_lock = threading.Lock()
_retries: dict[str, int] = defaultdict(int)
_citation_invalid: dict[str, int] = defaultdict(int)
_hist_counts: dict[str, list[int]] = {s: [0] * len(BUCKETS) for s in STEPS}
_hist_sum: dict[str, float] = defaultdict(float)
_hist_total: dict[str, int] = defaultdict(int)


def record_retry(step: str) -> None:
    """One provider call that was a second attempt (a retry), not every call."""
    if step in STEPS:
        with _lock:
            _retries[step] += 1


def record_claim(trace: list, product: str | None, reason_codes: list[str]) -> None:
    """Observe step latencies and the final citation outcome of one processed claim."""
    product = product if product in PRODUCTS else "UNKNOWN"
    with _lock:
        for entry in trace:
            if entry.step not in STEPS:
                continue
            seconds = entry.duration_ms / 1000
            for i, bound in enumerate(BUCKETS):
                if seconds <= bound:
                    _hist_counts[entry.step][i] += 1
            _hist_sum[entry.step] += seconds
            _hist_total[entry.step] += 1
        if "CITATION_INVALID" in reason_codes:
            _citation_invalid[product] += 1


def render(status_counts: dict[str, int]) -> str:
    lines = [
        "# HELP tabarru_claims Claims currently stored, by workflow status.",
        "# TYPE tabarru_claims gauge",
    ]
    lines += [f'tabarru_claims{{status="{s}"}} {status_counts.get(s, 0)}' for s in STATUSES]
    with _lock:
        lines += [
            "# HELP tabarru_provider_retries_total Provider calls that were a retry after"
            " malformed or invalid output, since process start. Counted while a claim is"
            " assessed, so a rare concurrent duplicate later rejected with 409 is included.",
            "# TYPE tabarru_provider_retries_total counter",
        ]
        lines += [f'tabarru_provider_retries_total{{step="{s}"}} {_retries[s]}' for s in STEPS]
        lines += [
            "# HELP tabarru_citation_invalid_total Claims whose coverage citations were still"
            " invalid after the retry (referred), since process start.",
            "# TYPE tabarru_citation_invalid_total counter",
        ]
        lines += [
            f'tabarru_citation_invalid_total{{product="{p}"}} {_citation_invalid[p]}'
            for p in PRODUCTS
        ]
        lines += [
            "# HELP tabarru_step_duration_seconds Pipeline step latency, since process start.",
            "# TYPE tabarru_step_duration_seconds histogram",
        ]
        for s in STEPS:
            for bound, count in zip(BUCKETS, _hist_counts[s], strict=True):
                lines.append(
                    f'tabarru_step_duration_seconds_bucket{{step="{s}",le="{bound}"}} {count}'
                )
            lines.append(
                f'tabarru_step_duration_seconds_bucket{{step="{s}",le="+Inf"}} {_hist_total[s]}'
            )
            lines.append(f'tabarru_step_duration_seconds_sum{{step="{s}"}} {_hist_sum[s]:.6f}')
            lines.append(f'tabarru_step_duration_seconds_count{{step="{s}"}} {_hist_total[s]}')
    return "\n".join(lines) + "\n"
