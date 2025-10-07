import argparse
import base64
import io
import os
import sys
import types
from pathlib import Path

import pytest


# Provide a minimal stub for the openai module if it is not installed.
if "openai" not in sys.modules:
    openai_stub = types.ModuleType("openai")

    class _OpenAIError(Exception):
        pass

    class _OpenAI:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    openai_stub.OpenAI = _OpenAI
    openai_stub.OpenAIError = _OpenAIError
    sys.modules["openai"] = openai_stub


import importlib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

image_cli = importlib.import_module("image_cli")


def make_args(**overrides: object) -> argparse.Namespace:
    defaults = {
        "prompt": None,
        "file": None,
        "env_file": Path(".env"),
        "output": None,
        "model": image_cli.DEFAULT_MODEL,
        "dry_run": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_parse_args_prompt_only():
    args = image_cli.parse_args(["--prompt", "hello", "--output", "img.png"])
    assert args.prompt == "hello"
    assert args.output == Path("img.png")
    assert args.model == image_cli.DEFAULT_MODEL


def test_populate_env_from_file_sets_missing(tmp_path, monkeypatch):
    env_file = tmp_path / "test.env"
    env_file.write_text("OPENAI_API_KEY=abc123\nOTHER=value\n", encoding="utf-8")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OTHER", raising=False)

    image_cli.populate_env_from_file(env_file)

    assert os.environ["OPENAI_API_KEY"] == "abc123"
    assert os.environ["OTHER"] == "value"


def test_ensure_api_key_requires_env(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(SystemExit) as exc:
        image_cli.ensure_api_key()
    assert "required" in str(exc.value)


def test_load_prompt_from_args_file(tmp_path):
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("  Draw a cat  \n", encoding="utf-8")
    args = make_args(file=prompt_file)

    prompt = image_cli.load_prompt_from_args(args)

    assert prompt == "Draw a cat"


def test_load_prompt_from_stdin(monkeypatch):
    args = make_args()
    monkeypatch.setattr(sys, "stdin", io.StringIO("  A scenic landscape  \n"))

    prompt = image_cli.load_prompt_from_args(args)

    assert prompt == "A scenic landscape"


class DummyResponseOutput:
    def __init__(self, type_: str, result: str):
        self.type = type_
        self.result = result


class DummyClient:
    def __init__(self, outputs):
        self._outputs = outputs
        self.responses = types.SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        return types.SimpleNamespace(output=self._outputs)


def test_generate_image_success():
    outputs = [
        DummyResponseOutput("text", "ignored"),
        DummyResponseOutput("image_generation_call", "ZmFrZV9pbWFnZQ=="),
    ]
    client = DummyClient(outputs)

    image_data = image_cli.generate_image(client, "prompt", "model")

    assert image_data == "ZmFrZV9pbWFnZQ=="


def test_generate_image_no_image_data():
    client = DummyClient([DummyResponseOutput("text", "nothing")])

    with pytest.raises(SystemExit) as exc:
        image_cli.generate_image(client, "prompt", "model")

    assert "No image" in str(exc.value)


def test_generate_image_handles_openai_error():
    class ErrorClient:
        def __init__(self):
            self.responses = types.SimpleNamespace(create=self._create)

        def _create(self, **kwargs):
            raise image_cli.OpenAIError("boom")

    client = ErrorClient()

    with pytest.raises(SystemExit) as exc:
        image_cli.generate_image(client, "prompt", "model")

    assert "Failed to generate image" in str(exc.value)


def test_resolve_output_path_returns_explicit_path():
    target = Path("custom.png")
    args = make_args(output=target)

    assert image_cli.resolve_output_path(args) == target


def test_save_image_writes_file(tmp_path):
    destination = tmp_path / "image.png"
    encoded = base64.b64encode(b"image-bytes").decode("ascii")

    image_cli.save_image(encoded, destination)

    assert destination.read_bytes() == b"image-bytes"


def test_main_dry_run(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    args = ["--prompt", "A castle", "--dry-run", "--output", "result.png"]

    image_cli.main(args)

    captured = capsys.readouterr()
    assert "[dry-run]" in captured.out
    assert "A castle" in captured.out
