import argparse
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


def test_choose_prompt_auto_approve(capsys):
    review = sora_cli.PromptReview(original="orig", improved="better")
    final = sora_cli.choose_prompt(review, auto_approve=True)
    assert final == "better"
    captured = capsys.readouterr()
    assert "Using refined prompt." in captured.out

