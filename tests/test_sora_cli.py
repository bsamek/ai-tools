import argparse
import builtins
import datetime
import io
import json
import os
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class _DummyOpenAI:
    def __init__(self, api_key: str):
        self.api_key = api_key


class _DummyOpenAIError(Exception):
    pass


fake_openai = types.ModuleType("openai")
fake_openai.OpenAI = _DummyOpenAI
fake_openai.OpenAIError = _DummyOpenAIError
sys.modules.setdefault("openai", fake_openai)

import sora_cli


def make_args(**overrides: object) -> argparse.Namespace:
    defaults = {
        "prompt": None,
        "file": None,
        "auto_approve": False,
        "env_file": Path(".env"),
        "skip_refinement": False,
        "refine_only": False,
        "output": None,
        "duration": sora_cli.DEFAULT_DURATION,
        "size": sora_cli.DEFAULT_SIZE,
        "timeout": 60,
        "poll_interval": 1.0,
        "dry_run": False,
        "sora_model": sora_cli.DEFAULT_SORA_MODEL,
        "refinement_model": sora_cli.DEFAULT_REASONING_MODEL,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_ensure_api_key_returns_value(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    assert sora_cli.ensure_api_key() == "secret"


def test_ensure_api_key_missing(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(SystemExit) as exc:
        sora_cli.ensure_api_key()
    assert "OPENAI_API_KEY" in str(exc.value)


def test_populate_env_from_file(tmp_path, monkeypatch):
    env_file = tmp_path / "sample.env"
    env_file.write_text(
        """
        # comment line
        OPENAI_API_KEY = from_file
        OTHER=value
        INVALID_LINE
        EMPTY_KEY= 
        QUOTED="quoted value"
        """.strip(),
        encoding="utf-8",
    )

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OTHER", raising=False)
    monkeypatch.delenv("QUOTED", raising=False)

    sora_cli.populate_env_from_file(env_file)

    assert os.environ["OPENAI_API_KEY"] == "from_file"
    assert os.environ["OTHER"] == "value"
    assert os.environ["QUOTED"] == "quoted value"


def test_load_prompt_prefers_inline(monkeypatch):
    args = make_args(prompt="  inline prompt  ", file=None)
    monkeypatch.setattr(sys, "stdin", io.StringIO("should not be used"))

    prompt = sora_cli.load_prompt_from_args(args)

    assert prompt == "inline prompt"


def test_load_prompt_reads_file(tmp_path):
    file_path = tmp_path / "prompt.txt"
    file_path.write_text("  from file  ", encoding="utf-8")
    args = make_args(prompt=None, file=file_path)

    prompt = sora_cli.load_prompt_from_args(args)

    assert prompt == "from file"


def test_load_prompt_missing_file(tmp_path):
    missing = tmp_path / "missing.txt"
    args = make_args(prompt=None, file=missing)

    with pytest.raises(SystemExit) as exc:
        sora_cli.load_prompt_from_args(args)

    assert str(missing) in str(exc.value)


def test_load_prompt_from_stdin(monkeypatch):
    args = make_args(prompt=None, file=None)
    monkeypatch.setattr(sys, "stdin", io.StringIO("  typed prompt  "))

    prompt = sora_cli.load_prompt_from_args(args)

    assert prompt == "typed prompt"


class _Response:
    def __init__(self, payload):
        self.output = [{"content": [{"type": "output_text", "text": payload}]}]


def test_parse_prompt_review_success():
    payload = json.dumps(
        {
            "improved_prompt": "Refined",
            "analysis": ["line one", "line two"],
            "tips": {"tip": "value"},
        }
    )
    response = _Response(payload)
    improved, analysis, tips = sora_cli.parse_prompt_review(response, "fallback")

    assert improved == "Refined"
    assert analysis == "line one\nline two"
    assert json.loads(tips) == {"tip": "value"}


def test_parse_prompt_review_failure(capsys):
    response = _Response("not json")
    improved, analysis, tips = sora_cli.parse_prompt_review(response, "fallback")

    captured = capsys.readouterr()
    assert "Failed to parse" in captured.err or "Failed to parse" in captured.out
    assert improved == "fallback"
    assert analysis == ""
    assert tips == ""


def test_build_video_request_payload_defaults():
    args = make_args(duration=8, size="1280x720")
    payload = sora_cli.build_video_request_payload("Prompt", args)

    assert payload == {
        "prompt": "Prompt",
        "model": sora_cli.DEFAULT_SORA_MODEL,
        "seconds": str(sora_cli.DEFAULT_DURATION),
        "size": sora_cli.DEFAULT_SIZE,
    }


def test_build_video_request_payload_custom():
    args = make_args(duration=12, size="1792x1024", sora_model="sora-2-pro")
    payload = sora_cli.build_video_request_payload("Prompt", args)

    assert payload["model"] == "sora-2-pro"
    assert payload["seconds"] == "12"
    assert payload["size"] == "1792x1024"


def test_parse_args_invalid_model():
    with pytest.raises(SystemExit):
        sora_cli.parse_args(["--model", "not-supported", "--prompt", "hi"])


def test_normalize_duration_invalid(capsys):
    assert sora_cli.normalize_duration(99) == str(sora_cli.DEFAULT_DURATION)
    captured = capsys.readouterr()
    assert "Unsupported duration" in captured.out


def test_normalize_size_invalid(capsys):
    assert sora_cli.normalize_size("999x999") == sora_cli.DEFAULT_SIZE
    captured = capsys.readouterr()
    assert "Unsupported size" in captured.out


@pytest.mark.parametrize(
    "usage, expected",
    [
        ({"total_credits": 5, "total_tokens": 42, "total_cost_usd": 1.25}, "credits: 5 | tokens: 42 | ≈ $1.2500 USD"),
        ({"total_cost_usd": 0.0}, "≈ $0.0000 USD"),
        ({}, None),
    ],
)
def test_summarize_cost_dict_input(usage, expected):
    job = {"usage": usage}
    assert sora_cli.summarize_cost(job) == expected


def test_summarize_cost_iterable():
    class Usage:
        def __iter__(self):
            yield from [("total_credits", 1), ("total_tokens", 2), ("total_cost_usd", 3.0)]

    job = type("Job", (), {"usage": Usage()})()
    summary = sora_cli.summarize_cost(job)
    assert summary == "credits: 1 | tokens: 2 | ≈ $3.0000 USD"


def test_choose_prompt_auto_approve(capsys):
    review = sora_cli.PromptReview(original="orig", improved="better")
    final = sora_cli.choose_prompt(review, auto_approve=True)
    assert final == "better"
    captured = capsys.readouterr()
    assert "Using refined prompt." in captured.out


def test_choose_prompt_interactive(monkeypatch, capsys):
    review = sora_cli.PromptReview(original="orig", improved="better")
    responses = iter(["4", "2"])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(responses))

    final = sora_cli.choose_prompt(review, auto_approve=False)

    assert final == "better"
    captured = capsys.readouterr()
    assert "Invalid selection" in captured.out


def test_get_manual_prompt_collects_lines(monkeypatch):
    responses = iter(["line 1", "line 2", " ", "ignored"])
    monkeypatch.setattr(builtins, "input", lambda: next(responses))

    prompt = sora_cli.get_manual_prompt()

    assert prompt == "line 1\nline 2"


def test_get_manual_prompt_empty(monkeypatch):
    responses = iter(["   "])
    monkeypatch.setattr(builtins, "input", lambda: next(responses))

    with pytest.raises(SystemExit) as exc:
        sora_cli.get_manual_prompt()

    assert "Manual prompt" in str(exc.value)


def test_display_prompt_review_outputs(capsys):
    review = sora_cli.PromptReview(
        original="orig",
        improved="better",
        analysis="analysis",
        tips="tips",
    )

    sora_cli.display_prompt_review(review)

    captured = capsys.readouterr()
    assert "Prompt Comparison" in captured.out
    assert "analysis" in captured.out
    assert "tips" in captured.out


def test_improve_prompt_success():
    payload = json.dumps(
        {
            "improved_prompt": "Improved",
            "analysis": ["line one", "line two"],
            "tips": "keep it up",
        }
    )

    class DummyResponses:
        def create(self, **kwargs):
            return types.SimpleNamespace(
                output=[{"content": [{"type": "output_text", "text": payload}]}]
            )

    client = types.SimpleNamespace(responses=DummyResponses())

    review = sora_cli.improve_prompt(client, "Original", model="test-model")

    assert review.original == "Original"
    assert review.improved == "Improved"
    assert "line one" in review.analysis
    assert review.tips == "keep it up"


def test_improve_prompt_openai_error(capsys):
    class DummyResponses:
        def create(self, **kwargs):
            raise sora_cli.OpenAIError("nope")

    client = types.SimpleNamespace(responses=DummyResponses())

    review = sora_cli.improve_prompt(client, "Original")

    assert review.original == "Original"
    assert review.improved == "Original"
    captured = capsys.readouterr()
    assert "Prompt refinement failed" in captured.out


def test_coerce_to_text_variants():
    assert sora_cli._coerce_to_text(None) == ""
    assert sora_cli._coerce_to_text(True) == "True"
    assert sora_cli._coerce_to_text([" a ", 2]) == "a\n2"
    assert json.loads(sora_cli._coerce_to_text({"k": "v"})) == {"k": "v"}


def test_normalize_duration_none():
    assert sora_cli.normalize_duration(None) is None


def test_normalize_size_none():
    assert sora_cli.normalize_size(None) is None


def test_create_sora_job_error():
    class DummyVideos:
        def create(self, **kwargs):
            raise sora_cli.OpenAIError("boom")

    client = types.SimpleNamespace(videos=DummyVideos())

    with pytest.raises(SystemExit) as exc:
        sora_cli.create_sora_job(client, {})

    assert "Failed to create" in str(exc.value)


def test_poll_sora_job_success(monkeypatch):
    class DummyVideos:
        def __init__(self):
            self.calls = []

        def retrieve(self, job_id):
            self.calls.append(job_id)
            return types.SimpleNamespace(id=job_id, status="completed", progress=100)

    videos = DummyVideos()
    client = types.SimpleNamespace(videos=videos)
    initial = types.SimpleNamespace(id="job-1", status="processing", progress=10)

    times = iter([0.0, 1.0])
    monkeypatch.setattr(sora_cli.time, "sleep", lambda _: None)
    monkeypatch.setattr(sora_cli.time, "monotonic", lambda: next(times))

    result = sora_cli.poll_sora_job(client, initial_job=initial, poll_interval=0.1, timeout=10)

    assert result.status == "completed"
    assert videos.calls == ["job-1"]


def test_poll_sora_job_timeout(monkeypatch):
    class DummyVideos:
        def retrieve(self, job_id):
            return types.SimpleNamespace(id=job_id, status="processing", progress=50)

    client = types.SimpleNamespace(videos=DummyVideos())
    initial = types.SimpleNamespace(id="job-9", status="processing", progress=20)

    times = iter([0.0, 100.0])
    monkeypatch.setattr(sora_cli.time, "sleep", lambda _: None)
    monkeypatch.setattr(sora_cli.time, "monotonic", lambda: next(times))

    with pytest.raises(SystemExit) as exc:
        sora_cli.poll_sora_job(client, initial_job=initial, poll_interval=0.1, timeout=1)

    assert "Timed out" in str(exc.value)


def test_resolve_output_path_default(monkeypatch):
    class FixedDateTime:
        @staticmethod
        def now(tz=None):
            assert tz == sora_cli.timezone.utc
            return datetime.datetime(2024, 2, 3, 4, 5, 6, tzinfo=tz)

    monkeypatch.setattr(sora_cli, "datetime", FixedDateTime)

    path = sora_cli.resolve_output_path(make_args(output=None))

    assert path.name == "sora_result_20240203-040506.mp4"


def test_download_video_asset_success(tmp_path):
    destination = tmp_path / "video.mp4"

    class DummyStream:
        def iter_bytes(self):
            yield from (b"abc", b"", b"def")

    class DummyVideos:
        def download_content(self, video_id, variant):
            assert video_id == "vid"
            assert variant == "video"
            return DummyStream()

    client = types.SimpleNamespace(videos=DummyVideos())

    sora_cli.download_video_asset(client, "vid", destination)

    assert destination.read_bytes() == b"abcdef"


def test_download_video_asset_error(tmp_path):
    class DummyVideos:
        def download_content(self, video_id, variant):
            raise sora_cli.OpenAIError("nope")

    client = types.SimpleNamespace(videos=DummyVideos())

    with pytest.raises(SystemExit) as exc:
        sora_cli.download_video_asset(client, "vid", tmp_path / "video.mp4")

    assert "Failed to download" in str(exc.value)


def test_summarize_cost_to_dict_recursive():
    class Usage:
        def to_dict_recursive(self):
            return {"total_credits": 2, "total_tokens": 5, "total_cost_usd": 1.5}

    job = types.SimpleNamespace(usage=Usage())

    summary = sora_cli.summarize_cost(job)

    assert "credits: 2" in summary
    assert "tokens: 5" in summary
    assert "$1.5000" in summary


def test_main_refine_only_flow(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "token")

    review = sora_cli.PromptReview(original="Prompt", improved="Better", analysis="A", tips="B")

    dummy_client = object()

    def fake_instantiate(api_key: str):
        assert api_key == "token"
        return dummy_client

    def fake_improve(client, prompt: str, model: str):
        assert client is dummy_client
        assert prompt == "Prompt"
        assert model == sora_cli.DEFAULT_REASONING_MODEL
        return review

    seen = {}

    def fake_display(review_arg):
        seen["review"] = review_arg

    monkeypatch.setattr(sora_cli, "instantiate_client", fake_instantiate)
    monkeypatch.setattr(sora_cli, "improve_prompt", fake_improve)
    monkeypatch.setattr(sora_cli, "display_prompt_review", fake_display)

    sora_cli.main(["--prompt", "Prompt", "--refine-only"])

    assert seen["review"] == review
    captured = capsys.readouterr()
    assert "Refined prompt ready to copy" in captured.out


def test_main_dry_run(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    args = [
        "--prompt",
        "Scene",
        "--dry-run",
        "--auto-approve",
        "--duration",
        "8",
    ]

    sora_cli.main(args)

    captured = capsys.readouterr()
    assert "[dry-run]" in captured.out
    assert "Scene" in captured.out

