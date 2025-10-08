import argparse
import base64
import datetime
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


def test_load_prompt_from_args_prefers_inline(monkeypatch):
    args = make_args(prompt=" inline text ")
    monkeypatch.setattr(sys, "stdin", io.StringIO("should not be read"))

    prompt = image_cli.load_prompt_from_args(args)

    assert prompt == "inline text"


def test_load_prompt_from_args_missing_file(tmp_path):
    missing = tmp_path / "missing.txt"
    args = make_args(file=missing)

    with pytest.raises(SystemExit) as exc:
        image_cli.load_prompt_from_args(args)

    assert str(missing) in str(exc.value)


def test_ensure_api_key_returns_value(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "secret")

    assert image_cli.ensure_api_key() == "secret"


class DummyImageData:
    def __init__(self, b64_json: str | None):
        self.b64_json = b64_json


class DummyImagesResource:
    def __init__(self, data: list[DummyImageData]):
        self._data = data

    def generate(self, **kwargs):  # noqa: D401 - mimic OpenAI client interface
        return types.SimpleNamespace(data=self._data)


class DummyClient:
    def __init__(self, data: list[DummyImageData]):
        self.images = DummyImagesResource(data)


def test_generate_image_success():
    data = [DummyImageData("ZmFrZV9pbWFnZQ==")]
    client = DummyClient(data)

    image_data = image_cli.generate_image(client, "prompt", "model")

    assert image_data == "ZmFrZV9pbWFnZQ=="


def test_generate_image_no_image_data():
    client = DummyClient([DummyImageData(None)])

    with pytest.raises(SystemExit) as exc:
        image_cli.generate_image(client, "prompt", "model")

    assert "Image payload missing" in str(exc.value)


def test_generate_image_handles_openai_error():
    class ErrorImagesResource:
        def generate(self, **kwargs):
            raise image_cli.OpenAIError("boom")

    class ErrorClient:
        def __init__(self):
            self.images = ErrorImagesResource()

    client = ErrorClient()

    with pytest.raises(SystemExit) as exc:
        image_cli.generate_image(client, "prompt", "model")

    assert "Failed to generate image" in str(exc.value)


def test_resolve_output_path_returns_explicit_path():
    target = Path("custom.png")
    args = make_args(output=target)

    assert image_cli.resolve_output_path(args) == target


def test_resolve_output_path_generates_timestamp(monkeypatch):
    class FixedDateTime:
        @staticmethod
        def now(tz=None):
            assert tz == image_cli.timezone.utc
            return datetime.datetime(2024, 1, 2, 3, 4, 5, tzinfo=tz)

    monkeypatch.setattr(image_cli, "datetime", FixedDateTime)

    path = image_cli.resolve_output_path(make_args(output=None))

    assert path.name == "image_result_20240102-030405.png"


def test_save_image_writes_file(tmp_path):
    destination = tmp_path / "image.png"
    encoded = base64.b64encode(b"image-bytes").decode("ascii")

    image_cli.save_image(encoded, destination)

    assert destination.read_bytes() == b"image-bytes"


def test_save_image_invalid_base64(tmp_path):
    destination = tmp_path / "bad.png"

    with pytest.raises(SystemExit) as exc:
        image_cli.save_image("not-base64!!", destination)

    assert "Failed to save image" in str(exc.value)


def test_main_dry_run(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    args = ["--prompt", "A castle", "--dry-run", "--output", "result.png"]

    image_cli.main(args)

    captured = capsys.readouterr()
    assert "[dry-run]" in captured.out
    assert "A castle" in captured.out


def test_main_success_flow(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "token")

    dummy_client = object()
    encoded = base64.b64encode(b"generated").decode("ascii")

    def fake_instantiate(api_key: str):
        assert api_key == "token"
        return dummy_client

    def fake_generate(client, prompt: str, model: str) -> str:
        assert client is dummy_client
        assert prompt == "Sunset"
        assert model == "gpt-image-1"
        return encoded

    monkeypatch.setattr(image_cli, "instantiate_client", fake_instantiate)
    monkeypatch.setattr(image_cli, "generate_image", fake_generate)

    args = ["--prompt", "Sunset", "--output", "final.png"]

    image_cli.main(args)

    saved = tmp_path / "final.png"
    assert saved.read_bytes() == b"generated"

    captured = capsys.readouterr()
    assert "Generating image" in captured.out
    assert "Image saved to" in captured.out
