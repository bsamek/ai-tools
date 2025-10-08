import argparse
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


def mock_read_text_error(*args, **kwargs):
    raise OSError("Permission denied")


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


def test_populate_env_from_file_handles_read_error(tmp_path, monkeypatch, capsys):
    env_file = tmp_path / "protected.env"
    env_file.touch()
    monkeypatch.setattr(Path, "read_text", mock_read_text_error)

    sora_cli.populate_env_from_file(env_file)

    captured = capsys.readouterr()
    assert "[warn] Failed to read env file" in captured.out


def test_load_prompt_from_args_inline():
    args = make_args(prompt="  inline prompt  ")
    assert sora_cli.load_prompt_from_args(args) == "inline prompt"


def test_load_prompt_from_args_file(tmp_path):
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("  file prompt  ")
    args = make_args(file=prompt_file)
    assert sora_cli.load_prompt_from_args(args) == "file prompt"


def test_load_prompt_from_args_file_error(tmp_path, monkeypatch):
    prompt_file = tmp_path / "unreadable.txt"
    prompt_file.touch()
    args = make_args(file=prompt_file)
    monkeypatch.setattr(Path, "read_text", mock_read_text_error)

    with pytest.raises(SystemExit) as exc:
        sora_cli.load_prompt_from_args(args)
    assert "Failed to read prompt file" in str(exc.value)


def test_load_prompt_from_stdin(monkeypatch):
    args = make_args()
    monkeypatch.setattr(sys, "stdin", io.StringIO("  stdin prompt  "))
    assert sora_cli.load_prompt_from_args(args) == "stdin prompt"


def test_load_prompt_from_stdin_empty(monkeypatch):
    args = make_args()
    monkeypatch.setattr(sys, "stdin", io.StringIO("\n"))
    with pytest.raises(SystemExit) as exc:
        sora_cli.load_prompt_from_args(args)
    assert "No prompt provided" in str(exc.value)


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


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, ""),
        ("text", "text"),
        (123, "123"),
        (True, "True"),
        (["a", "b", 3], "a\nb\n3"),
        ({"key": "value"}, '{\n  "key": "value"\n}'),
    ],
)
def test_coerce_to_text(value, expected):
    assert sora_cli._coerce_to_text(value) == expected


def test_improve_prompt_api_error(capsys):
    class ErrorClient:
        def __init__(self, *args, **kwargs):
            self.responses = self

        def create(self, *args, **kwargs):
            raise sora_cli.OpenAIError("API is down")

    client = ErrorClient()
    review = sora_cli.improve_prompt(client, "original prompt")
    assert review.original == "original prompt"
    assert review.improved == "original prompt"

    captured = capsys.readouterr()
    assert "[warn] Prompt refinement failed" in captured.out


def test_parse_prompt_review_handles_missing_keys():
    payload = json.dumps({"wrong_key": "value"})
    response = _Response(payload)
    improved, _, _ = sora_cli.parse_prompt_review(response, "fallback")
    assert improved == "fallback"


def test_choose_prompt_select_original(monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO("1\n"))
    review = sora_cli.PromptReview(original="orig", improved="better")
    final = sora_cli.choose_prompt(review, auto_approve=False)
    assert final == "orig"


def test_choose_prompt_select_refined(monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO("2\n"))
    review = sora_cli.PromptReview(original="orig", improved="better")
    final = sora_cli.choose_prompt(review, auto_approve=False)
    assert final == "better"


def test_choose_prompt_reenter_manually(monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO("3\noverride\n\n"))
    review = sora_cli.PromptReview(original="orig", improved="better")
    final = sora_cli.choose_prompt(review, auto_approve=False)
    assert final == "override"


def test_get_manual_prompt_empty_input(monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO("\n"))
    with pytest.raises(SystemExit) as exc:
        sora_cli.get_manual_prompt()
    assert "cannot be empty" in str(exc.value)


def test_choose_prompt_auto_approve(capsys):
    review = sora_cli.PromptReview(original="orig", improved="better")
    final = sora_cli.choose_prompt(review, auto_approve=True)
    assert final == "better"
    captured = capsys.readouterr()
    assert "Using refined prompt." in captured.out


def test_main_dry_run(capsys):
    args = ["--prompt", "test", "--dry-run"]
    sora_cli.main(args)
    captured = capsys.readouterr()
    assert "[dry-run] Payload to send:" in captured.out
    assert '"prompt": "test"' in captured.out


def test_main_refine_only(capsys, monkeypatch):
    class MockClient:
        def __init__(self, *args, **kwargs):
            self.responses = self
        def create(self, *args, **kwargs):
            return _Response(json.dumps({"improved_prompt": "refined"}))

    monkeypatch.setenv("OPENAI_API_KEY", "fake")
    monkeypatch.setattr(sora_cli, "instantiate_client", MockClient)
    args = ["--prompt", "test", "--refine-only"]
    sora_cli.main(args)
    captured = capsys.readouterr()
    assert "Refined prompt ready to copy" in captured.out
    assert "refined" in captured.out


def test_resolve_output_path_generates_default(monkeypatch):
    class MockDatetime:
        @classmethod
        def now(cls, *args, **kwargs):
            return types.SimpleNamespace(strftime=lambda fmt: "20240101-120000")

    monkeypatch.setattr(sora_cli, "datetime", MockDatetime)
    args = make_args()
    path = sora_cli.resolve_output_path(args)
    assert path == Path("sora_result_20240101-120000.mp4")


def test_main_job_fails(capsys, monkeypatch):
    class MockJob:
        id = "job_123"
        status = "failed"
        error = "Something went wrong"
        usage = None

    class MockClient:
        def __init__(self, *args, **kwargs):
            self.videos = self
        def create(self, **kwargs):
            return MockJob()
        def retrieve(self, *args, **kwargs):
            return MockJob()

    monkeypatch.setenv("OPENAI_API_KEY", "fake")
    monkeypatch.setattr(sora_cli, "instantiate_client", MockClient)
    args = ["--prompt", "test", "--skip-refinement"]
    with pytest.raises(SystemExit) as exc:
        sora_cli.main(args)
    assert "Something went wrong" in str(exc.value)


def test_main_job_timeout(capsys, monkeypatch):
    class MockJob:
        id = "job_123"
        status = "processing"
        progress = 50
        usage = None

    class MockClient:
        def __init__(self, *args, **kwargs):
            self.videos = self
        def create(self, **kwargs):
            return MockJob()
        def retrieve(self, *args, **kwargs):
            return MockJob()

    monkeypatch.setenv("OPENAI_API_KEY", "fake")
    monkeypatch.setattr(sora_cli, "instantiate_client", MockClient)
    args = ["--prompt", "test", "--skip-refinement", "--timeout", "0", "--poll-interval", "0.05"]
    with pytest.raises(SystemExit) as exc:
        sora_cli.main(args)
    assert "Timed out" in str(exc.value)


def test_main_download_fails(capsys, monkeypatch):
    class MockJob:
        id = "job_123"
        status = "completed"
        usage = None

    class MockClient:
        def __init__(self, *args, **kwargs):
            self.videos = self
        def create(self, **kwargs):
            return MockJob()
        def retrieve(self, *args, **kwargs):
            return MockJob()
        def download_content(self, *args, **kwargs):
            raise sora_cli.OpenAIError("Download failed")

    monkeypatch.setenv("OPENAI_API_KEY", "fake")
    monkeypatch.setattr(sora_cli, "instantiate_client", MockClient)
    args = ["--prompt", "test", "--skip-refinement"]
    with pytest.raises(SystemExit) as exc:
        sora_cli.main(args)
    assert "Failed to download video" in str(exc.value)
