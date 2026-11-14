"""Real-model selection CLI, tested offline: the HTTP layer is replaced."""

import io
import json

import pytest

from claims import models
from claims.providers import FakeProvider


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_list_models_reads_openai_style_ids_and_sends_key(monkeypatch):
    seen = {}

    def fake_urlopen(request, timeout):
        seen["url"], seen["auth"] = request.full_url, request.headers.get("Authorization")
        return _Response(json.dumps({"data": [{"id": "b-model"}, {"id": "a-model"}]}).encode())

    monkeypatch.setattr(models.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    assert models.list_models("groq") == ["a-model", "b-model"]
    assert seen["url"].endswith("/models") and seen["auth"] == "Bearer test-key"


def test_list_models_without_key_or_server_fails_without_echoing_details(monkeypatch):
    with pytest.raises(RuntimeError, match="GROQ_API_KEY is not set"):
        models.list_models("groq")

    def refuse(request, timeout):
        raise OSError("connection refused to http://secret-host")

    monkeypatch.setattr(models.urllib.request, "urlopen", refuse)
    with pytest.raises(RuntimeError) as exc:
        models.list_models("ollama")
    assert "secret-host" not in str(exc.value)


def test_env_file_does_not_override_existing_values(tmp_path, monkeypatch):
    env = tmp_path / ".env.local"
    env.write_text("# comment\nGROQ_API_KEY='from-file'\nOTHER=x\n")
    monkeypatch.setenv("OTHER", "kept")
    models.load_env_file(env)
    import os

    assert os.environ["GROQ_API_KEY"] == "from-file" and os.environ["OTHER"] == "kept"
    monkeypatch.delenv("GROQ_API_KEY")


def test_run_writes_separate_reports_and_comparison(dataset, tmp_path, monkeypatch):
    # The "real" provider is a FakeProvider here, so the test stays offline.
    monkeypatch.setattr(models, "get_provider", FakeProvider)
    code = models.main(
        ["run", "--source", "ollama", "--model", "m:1", "--limit", "3"]
        + ["--data", str(dataset), "--out", str(tmp_path)]
    )
    assert code == 0
    out = tmp_path / "real" / "ollama__m_1"
    real = json.loads((out / "coverage_eval.json").read_text())
    assert real["claims"] == 3 and real["run"]["model"] == "m:1"
    assert json.loads((out / "fake_baseline.json").read_text())["provider"] == "fake"
    table = (out / "comparison.md").read_text()
    assert "decision_accuracy" in table and "citation_invalid_rate" in table
    assert not (tmp_path / "coverage_eval.json").exists()  # deterministic report untouched


def test_throttle_spaces_calls(monkeypatch):
    clock = {"t": 100.0}
    slept = []
    monkeypatch.setattr(models.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(models.time, "sleep", lambda s: slept.append(s))
    throttled = models.Throttled(FakeProvider(), 2.0)
    inner = FakeProvider()
    throttled.inner = type("P", (), {"name": "x", "complete": lambda self, p: inner.name})()
    throttled.complete("a")
    throttled.complete("b")
    assert slept == [2.0]
