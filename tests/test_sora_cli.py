import argparse
import builtins
import io
import json
import os
import sys
import types
from pathlib import Path

import importlib.machinery
import importlib.util
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


def test_populate_env_from_file_handles_oserror(tmp_path, monkeypatch, capsys):
    env_file = tmp_path / "broken.env"
    env_file.write_text("OPENAI_API_KEY=value", encoding="utf-8")

    original_read_text = Path.read_text

    def fake_read_text(self, *args, **kwargs):
        if self == env_file:
            raise OSError("boom")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fake_read_text)

    sora_cli.populate_env_from_file(env_file)

    captured = capsys.readouterr()
    assert "Failed to read env file" in captured.out


def test_parse_args_defaults():
    args = sora_cli.parse_args([])

    assert args.duration == sora_cli.DEFAULT_DURATION
    assert args.size == sora_cli.DEFAULT_SIZE
    assert args.dry_run is False


def test_coerce_to_text_various():
    assert sora_cli._coerce_to_text(None) == ""
    assert sora_cli._coerce_to_text("hello") == "hello"
    assert sora_cli._coerce_to_text(3) == "3"
    assert sora_cli._coerce_to_text(["a", 2]) == "a\n2"
    assert sora_cli._coerce_to_text({"k": "v"}) == json.dumps({"k": "v"}, indent=2)


def test_instantiate_client_returns_openai():
    client = sora_cli.instantiate_client("token")

    assert isinstance(client, sora_cli.OpenAI)


def test_improve_prompt_success():
    payload = json.dumps(
        {
            "improved_prompt": "Refined",
            "analysis": "Analysis",
            "tips": "Tips",
        }
    )

    class DummyClient:
        def __init__(self):
            self.responses = types.SimpleNamespace(create=self.create)
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            return types.SimpleNamespace(
                output=[{"content": [{"type": "output_text", "text": payload}]}]
            )

    client = DummyClient()

    review = sora_cli.improve_prompt(client, "Original", model="model-x")

    assert review.improved == "Refined"
    assert client.calls[0]["model"] == "model-x"


def test_improve_prompt_failure(capsys):
    class DummyClient:
        def __init__(self):
            self.responses = types.SimpleNamespace(create=self.create)

        def create(self, **kwargs):
            raise sora_cli.OpenAIError("nope")

    client = DummyClient()

    review = sora_cli.improve_prompt(client, "Original", model="model-x")

    captured = capsys.readouterr()
    assert "Prompt refinement failed" in captured.out
    assert review.original == review.improved == "Original"


def test_load_prompt_from_args_file_failure(tmp_path):
    directory = tmp_path / "nested"
    directory.mkdir()
    args = make_args(file=directory)

    with pytest.raises(SystemExit) as exc:
        sora_cli.load_prompt_from_args(args)

    assert "Failed to read prompt" in str(exc.value)


def test_load_prompt_keyboard_interrupt(monkeypatch):
    args = make_args()

    class Boom:
        def read(self):
            raise KeyboardInterrupt

    monkeypatch.setattr(sys, "stdin", Boom())

    with pytest.raises(SystemExit) as exc:
        sora_cli.load_prompt_from_args(args)

    assert "Prompt entry cancelled" in str(exc.value)


def test_load_prompt_no_input(monkeypatch):
    args = make_args()
    monkeypatch.setattr(sys, "stdin", io.StringIO("   "))

    with pytest.raises(SystemExit) as exc:
        sora_cli.load_prompt_from_args(args)

    assert "No prompt" in str(exc.value)


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
    args = make_args(duration=12, size="1792x1024", sora_model="sora-x")
    payload = sora_cli.build_video_request_payload("Prompt", args)

    assert payload["model"] == "sora-x"
    assert payload["seconds"] == "12"
    assert payload["size"] == "1792x1024"


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


def test_summarize_cost_to_dict_recursive():
    class Usage:
        def to_dict_recursive(self):
            return {"total_credits": 9}

    job = type("Job", (), {"usage": Usage()})()

    assert sora_cli.summarize_cost(job) == "credits: 9"


def test_summarize_cost_none_usage():
    assert sora_cli.summarize_cost({}) is None


def test_choose_prompt_auto_approve(capsys):
    review = sora_cli.PromptReview(original="orig", improved="better")
    final = sora_cli.choose_prompt(review, auto_approve=True)
    assert final == "better"
    captured = capsys.readouterr()
    assert "Using refined prompt." in captured.out


def test_choose_prompt_interactive_selection(monkeypatch):
    review = sora_cli.PromptReview(original="orig", improved="better")

    inputs = iter(["", "2"])

    def fake_input(prompt: str = "") -> str:
        return next(inputs)

    monkeypatch.setattr(sora_cli, "display_prompt_review", lambda review: None)
    monkeypatch.setattr("builtins.input", fake_input)

    final = sora_cli.choose_prompt(review, auto_approve=False)

    assert final == "better"


def test_get_manual_prompt(monkeypatch):
    responses = iter(["First", "Second", ""])  # blank line terminates

    monkeypatch.setattr("builtins.input", lambda prompt="": next(responses))

    assert sora_cli.get_manual_prompt() == "First\nSecond"


def test_get_manual_prompt_empty(monkeypatch):
    responses = iter(["   ", ""])

    monkeypatch.setattr("builtins.input", lambda prompt="": next(responses))

    with pytest.raises(SystemExit) as exc:
        sora_cli.get_manual_prompt()

    assert "Manual prompt cannot be empty" in str(exc.value)


def test_display_prompt_review_outputs(capsys):
    review = sora_cli.PromptReview(
        original="orig",
        improved="better",
        analysis="analysis text",
        tips="tip text",
    )

    sora_cli.display_prompt_review(review)

    captured = capsys.readouterr()
    assert "Prompt Comparison" in captured.out
    assert "analysis text" in captured.out
    assert "tip text" in captured.out


def test_normalize_duration_valid(capsys):
    assert sora_cli.normalize_duration(4) == "4"
    captured = capsys.readouterr()
    assert captured.out == ""


def test_normalize_size_valid(capsys):
    assert sora_cli.normalize_size("1280x720") == "1280x720"
    captured = capsys.readouterr()
    assert captured.out == ""


def test_create_sora_job_success():
    class Videos:
        def __init__(self):
            self.created = None

        def create(self, **kwargs):
            self.created = kwargs
            return {"job": "ok"}

    client = types.SimpleNamespace(videos=Videos())
    payload = {"prompt": "test"}

    result = sora_cli.create_sora_job(client, payload)

    assert result == {"job": "ok"}
    assert client.videos.created == payload


def test_poll_sora_job_completes(monkeypatch, capsys):
    job_sequence = iter(
        [
            types.SimpleNamespace(id="job", status="processing", progress=10),
            types.SimpleNamespace(id="job", status="completed", progress=100),
        ]
    )

    def fake_retrieve(job_id: str):
        return next(job_sequence)

    client = types.SimpleNamespace(videos=types.SimpleNamespace(retrieve=fake_retrieve))
    initial_job = types.SimpleNamespace(id="job", status="queued", progress=0)

    counter = {"value": 0}

    def fake_monotonic():
        counter["value"] += 0.1
        return counter["value"]

    monkeypatch.setattr(sora_cli.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(sora_cli.time, "monotonic", fake_monotonic)

    final_job = sora_cli.poll_sora_job(client, initial_job, poll_interval=0, timeout=5)

    assert final_job.status == "completed"
    captured = capsys.readouterr()
    assert "status" in captured.out


def test_poll_sora_job_timeout(monkeypatch):
    client = types.SimpleNamespace(
        videos=types.SimpleNamespace(retrieve=lambda job_id: types.SimpleNamespace(id=job_id, status="processing", progress=10))
    )
    initial_job = types.SimpleNamespace(id="job", status="processing", progress=0)

    times = iter([0.0, 10.0])

    monkeypatch.setattr(sora_cli.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(sora_cli.time, "monotonic", lambda: next(times))

    with pytest.raises(SystemExit) as exc:
        sora_cli.poll_sora_job(client, initial_job, poll_interval=0, timeout=1)

    assert "Timed out" in str(exc.value)


def test_poll_sora_job_retrieve_failure(monkeypatch):
    class Videos:
        def retrieve(self, job_id):
            raise sora_cli.OpenAIError("boom")

    client = types.SimpleNamespace(videos=Videos())
    initial_job = types.SimpleNamespace(id="job", status="processing", progress=0)

    calls = iter([0.0, 0.1, 0.2])

    monkeypatch.setattr(sora_cli.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(sora_cli.time, "monotonic", lambda: next(calls))

    with pytest.raises(SystemExit) as exc:
        sora_cli.poll_sora_job(client, initial_job, poll_interval=0, timeout=5)

    assert "Polling failed" in str(exc.value)


def test_resolve_output_path_uses_timestamp(monkeypatch):
    class _DummyNow:
        def strftime(self, fmt: str) -> str:
            return "20240101-120000"

    class _DummyDatetime:
        @classmethod
        def now(cls, tz=None):
            return _DummyNow()

    monkeypatch.setattr(sora_cli, "datetime", _DummyDatetime)

    args = make_args(output=None)

    path = sora_cli.resolve_output_path(args)

    assert path.name == "sora_result_20240101-120000.mp4"


def test_download_video_asset_writes(tmp_path):
    destination = tmp_path / "video.mp4"

    class Stream:
        def iter_bytes(self):
            yield b"hello"
            yield b""
            yield b" world"

    client = types.SimpleNamespace(videos=types.SimpleNamespace(download_content=lambda video_id, variant: Stream()))

    sora_cli.download_video_asset(client, "vid", destination)

    assert destination.read_bytes() == b"hello world"


def test_download_video_asset_error(monkeypatch, tmp_path):
    destination = tmp_path / "video.mp4"

    class Videos:
        def download_content(self, video_id, variant):
            raise sora_cli.OpenAIError("fail")

    client = types.SimpleNamespace(videos=Videos())

    with pytest.raises(SystemExit) as exc:
        sora_cli.download_video_asset(client, "vid", destination)

    assert "Failed to download" in str(exc.value)


def test_main_dry_run(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)

    args = [
        "--prompt",
        "Scene",
        "--dry-run",
        "--auto-approve",
        "--duration",
        "8",
        "--size",
        "1280x720",
    ]

    sora_cli.main(args)

    captured = capsys.readouterr()
    assert "[dry-run] Payload" in captured.out


def test_main_refine_only(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)

    args = [
        "--prompt",
        "Scene",
        "--skip-refinement",
        "--refine-only",
    ]

    sora_cli.main(args)

    captured = capsys.readouterr()
    assert "Refined prompt ready" in captured.out


def test_main_generates_video(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)

    def fake_ensure():
        return "token"

    class DummyClient:
        pass

    def fake_instantiate(api_key):
        assert api_key == "token"
        return DummyClient()

    def fake_improve(client, prompt, model):
        assert prompt == "Scene"
        return sora_cli.PromptReview(original=prompt, improved=f"{prompt} refined")

    def fake_build(prompt, args):
        return {"prompt": prompt, "model": args.sora_model}

    initial_job = types.SimpleNamespace(id="job-1", status="queued")
    final_job = types.SimpleNamespace(id="job-1", status="completed", usage={"total_cost_usd": 1.5})

    def fake_create(client, payload):
        assert payload["prompt"].endswith("refined")
        return initial_job

    def fake_poll(client, initial_job, poll_interval, timeout):
        return final_job

    saved = {}

    def fake_download(client, video_id, destination):
        saved["path"] = destination
        destination.write_bytes(b"video")

    def fake_resolve(args, suffix=".mp4"):
        return tmp_path / "output.mp4"

    monkeypatch.setattr(sora_cli, "ensure_api_key", fake_ensure)
    monkeypatch.setattr(sora_cli, "instantiate_client", fake_instantiate)
    monkeypatch.setattr(sora_cli, "improve_prompt", fake_improve)
    monkeypatch.setattr(sora_cli, "build_video_request_payload", fake_build)
    monkeypatch.setattr(sora_cli, "create_sora_job", fake_create)
    monkeypatch.setattr(sora_cli, "poll_sora_job", fake_poll)
    monkeypatch.setattr(sora_cli, "download_video_asset", fake_download)
    monkeypatch.setattr(sora_cli, "resolve_output_path", fake_resolve)

    args = [
        "--prompt",
        "Scene",
        "--auto-approve",
    ]

    sora_cli.main(args)

    captured = capsys.readouterr()
    assert "Submitting Sora video generation job" in captured.out
    assert saved["path"].name == "output.mp4"


def test_main_job_missing_id(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(sora_cli, "ensure_api_key", lambda: "token")
    monkeypatch.setattr(sora_cli, "instantiate_client", lambda api_key: object())
    monkeypatch.setattr(
        sora_cli,
        "improve_prompt",
        lambda client, prompt, model: sora_cli.PromptReview(original=prompt, improved=prompt),
    )
    monkeypatch.setattr(sora_cli, "build_video_request_payload", lambda prompt, args: {})
    monkeypatch.setattr(sora_cli, "create_sora_job", lambda client, payload: types.SimpleNamespace(status="queued"))

    args = ["--prompt", "Scene", "--auto-approve"]

    with pytest.raises(SystemExit) as exc:
        sora_cli.main(args)

    assert "missing video ID" in str(exc.value)


def test_main_job_failure_status(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(sora_cli, "ensure_api_key", lambda: "token")
    monkeypatch.setattr(sora_cli, "instantiate_client", lambda api_key: object())
    monkeypatch.setattr(
        sora_cli,
        "improve_prompt",
        lambda client, prompt, model: sora_cli.PromptReview(original=prompt, improved=prompt),
    )
    monkeypatch.setattr(sora_cli, "build_video_request_payload", lambda prompt, args: {})
    monkeypatch.setattr(sora_cli, "create_sora_job", lambda client, payload: types.SimpleNamespace(id="job", status="queued"))
    monkeypatch.setattr(
        sora_cli,
        "poll_sora_job",
        lambda client, initial_job, poll_interval, timeout: types.SimpleNamespace(id="job", status="failed", error="oops"),
    )

    args = ["--prompt", "Scene", "--auto-approve"]

    with pytest.raises(SystemExit) as exc:
        sora_cli.main(args)

    assert "Job ended with status" in str(exc.value)


def test_sora_cli_import_error(monkeypatch):
    monkeypatch.delitem(sys.modules, "openai", raising=False)

    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "openai":
            raise ImportError("missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    module_name = "sora_cli_import_test"
    loader = importlib.machinery.SourceFileLoader(module_name, str(ROOT / "sora_cli.py"))
    spec = importlib.util.spec_from_loader(module_name, loader)
    module = importlib.util.module_from_spec(spec)

    with pytest.raises(SystemExit) as exc:
        loader.exec_module(module)

    assert "OpenAI Python SDK" in str(exc.value)

