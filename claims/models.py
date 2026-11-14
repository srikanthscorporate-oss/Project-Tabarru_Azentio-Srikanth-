"""Pick a real model (local Ollama or Groq) and compare it with the fake provider.

    python -m claims.models list [--env-file .env.local]
    python -m claims.models run  [--source ollama|groq] [--model NAME]
                                 [--limit N] [--min-interval S] [--env-file .env.local]

`list` shows the models installed in the local Ollama server and the models
the Groq account can use. `run` asks for a source and model when they are not
given, then runs the coverage evaluation twice on the same claims: once with
the chosen model and once with FakeProvider. It writes
reports/real/<source>__<model>/{coverage_eval.json, fake_baseline.json,
comparison.md}. The deterministic report in reports/coverage_eval.json is not
touched.

Real inference is selected the same way as everywhere else: this command sets
CLAIMS_PROVIDER=openai and the CLAIMS_LLM_* variables for its own process.
A failing model call is recorded as a failure and never replaced with a fake
answer. The Groq key is read from GROQ_API_KEY (the environment, or the file
given with --env-file) and is never printed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from claims.eval import evaluate, format_table
from claims.providers import LLMProvider, get_provider

SOURCES = {
    "ollama": {
        "base_url": os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        "key_var": None,
    },
    "groq": {"base_url": "https://api.groq.com/openai/v1", "key_var": "GROQ_API_KEY"},
}
COMPARED = [
    "decision_accuracy",
    "citation_invalid_rate",
    "attempted_citation_invalid_rate",
    "payable_exact_match",
    "citation_precision",
    "citation_recall",
    "retrieval_recall_at_k",
    "coverage_provider_called",
]


def load_env_file(path: Path) -> None:
    """Read KEY=VALUE lines into os.environ without overriding what is already set."""
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def _api_key(source: str) -> str | None:
    var = SOURCES[source]["key_var"]
    return os.environ.get(var) if var else None


def list_models(source: str, timeout_s: float = 10) -> list[str]:
    """Model ids from the source's OpenAI-compatible /models endpoint."""
    key_var = SOURCES[source]["key_var"]
    headers = {"User-Agent": "tabarru-claims/0.1"}
    if key_var:
        key = _api_key(source)
        if not key:
            raise RuntimeError(f"{key_var} is not set")
        headers["Authorization"] = f"Bearer {key}"
    request = urllib.request.Request(SOURCES[source]["base_url"] + "/models", headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            payload = json.load(response)
    except Exception as exc:  # report the type and status only: messages can echo details
        status = f" {exc.code}" if isinstance(exc, urllib.error.HTTPError) else ""
        raise RuntimeError(
            f"{source} model list unavailable ({type(exc).__name__}{status})"
        ) from None
    return sorted(m["id"] for m in payload.get("data", []))


class Throttled:
    """Waits between calls so a rate-limited API (Groq free tier) is not flooded."""

    def __init__(self, inner: LLMProvider, min_interval_s: float):
        self.inner, self.min_interval_s, self._last = inner, min_interval_s, 0.0
        self.name = inner.name

    def complete(self, prompt: str) -> str:
        wait = self._last + self.min_interval_s - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        try:
            return self.inner.complete(prompt)
        finally:
            self._last = time.monotonic()


def comparison_markdown(real: dict, fake: dict, meta: dict) -> str:
    lines = [
        f"# Coverage evaluation: {meta['source']} {meta['model']} vs fake",
        "",
        f"- provider type: OpenAI-compatible ({meta['source']}), temperature 0",
        f"- model: {meta['model']}",
        f"- dataset: {meta['data']} (generator seed {meta['seed']}), claims: {real['claims']}",
        f"- top-k: {real['k']}, retry policy: one retry with the validation error",
        f"- elapsed: real {meta['real_seconds']:.1f} s, fake {meta['fake_seconds']:.2f} s",
        f"- min interval between calls: {meta['min_interval']} s",
        "",
        "| metric | real | fake |",
        "|---|---|---|",
    ]
    lines += [f"| {k} | {real[k]} | {fake[k]} |" for k in COMPARED]
    return "\n".join(lines) + "\n"


def _choose(prompt: str, options: list[str]) -> str:
    for i, option in enumerate(options, 1):
        print(f"  {i}. {option}")
    while True:
        raw = input(f"{prompt} [1-{len(options)}]: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        print("enter one of the numbers above")


def cmd_list() -> int:
    for source in SOURCES:
        print(f"{source}:")
        try:
            models = list_models(source)
            print("\n".join(f"  {m}" for m in models) if models else "  (none)")
        except RuntimeError as exc:
            print(f"  unavailable: {exc}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    source = args.source or _choose("source", list(SOURCES))
    model = args.model
    if model is None:
        try:
            models = list_models(source)
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        if not models:
            print(f"error: no models available from {source}", file=sys.stderr)
            return 1
        model = _choose("model", models)
    if SOURCES[source]["key_var"] and not _api_key(source):
        print(f"error: {SOURCES[source]['key_var']} is not set", file=sys.stderr)
        return 1

    os.environ.update(
        {
            "CLAIMS_PROVIDER": "openai",
            "CLAIMS_LLM_BASE_URL": SOURCES[source]["base_url"],
            "CLAIMS_LLM_MODEL": model,
            "CLAIMS_LLM_TIMEOUT_S": str(args.timeout),
        }
    )
    key = _api_key(source)
    if key:
        os.environ["CLAIMS_LLM_API_KEY"] = key
    else:
        os.environ.pop("CLAIMS_LLM_API_KEY", None)

    shared = Throttled(get_provider(), args.min_interval)
    print(f"running {source} {model} on {args.limit or 'all'} claims...", file=sys.stderr)
    started = time.perf_counter()
    real = evaluate(args.data, f"{source}:{model}", lambda: shared, args.limit)
    real_seconds = time.perf_counter() - started
    started = time.perf_counter()
    fake = evaluate(args.data, "fake", limit=args.limit)
    fake_seconds = time.perf_counter() - started

    out = args.out / "real" / f"{source}__{model.replace('/', '_').replace(':', '_')}"
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "source": source,
        "model": model,
        "data": str(args.data),
        "seed": args.seed,
        "min_interval": args.min_interval,
        "real_seconds": real_seconds,
        "fake_seconds": fake_seconds,
    }
    (out / "coverage_eval.json").write_text(json.dumps({**real, "run": meta}, indent=2) + "\n")
    (out / "fake_baseline.json").write_text(json.dumps(fake, indent=2) + "\n")
    comparison = comparison_markdown(real, fake, meta)
    (out / "comparison.md").write_text(comparison)
    print(comparison)
    print(format_table(real))
    print(f"\nreports written to {out}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="List real models and compare one with fake.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("list", "run"):
        p = sub.add_parser(name)
        p.add_argument("--env-file", type=Path, help="read GROQ_API_KEY etc. from this file")
    run = sub.choices["run"]
    run.add_argument("--source", choices=list(SOURCES))
    run.add_argument("--model")
    run.add_argument("--limit", type=int, help="first N claims only (same N for both runs)")
    run.add_argument("--min-interval", type=float, default=0.0, help="seconds between calls")
    run.add_argument("--timeout", type=float, default=60.0, help="seconds per model call")
    run.add_argument("--data", type=Path, default=Path("data"))
    run.add_argument("--out", type=Path, default=Path("reports"))
    run.add_argument("--seed", type=int, default=7, help="generator seed of --data, for the report")
    args = parser.parse_args(argv)
    if args.env_file:
        load_env_file(args.env_file)
    return cmd_list() if args.command == "list" else cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
