import argparse
import base64
import builtins
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
import importlib.machinery
import importlib.util

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


def test_populate_env_from_file_handles_oserror(tmp_path, monkeypatch, capsys):
    env_file = tmp_path / "broken.env"
    env_file.write_text("OPENAI_API_KEY=abc\n", encoding="utf-8")

    original_read_text = Path.read_text

    def fake_read_text(self, *args, **kwargs):
        if self == env_file:
            raise OSError("boom")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fake_read_text)

    image_cli.populate_env_from_file(env_file)

    captured = capsys.readouterr()
    assert "Failed to read env file" in captured.out


def test_ensure_api_key_requires_env(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(SystemExit) as exc:
        image_cli.ensure_api_key()
    assert "required" in str(exc.value)


def test_ensure_api_key_returns_value(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "secret")

    assert image_cli.ensure_api_key() == "secret"


def test_load_prompt_from_args_file(tmp_path):
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("  Draw a cat  \n", encoding="utf-8")
    args = make_args(file=prompt_file)

    prompt = image_cli.load_prompt_from_args(args)

    assert prompt == "Draw a cat"


def test_load_prompt_from_args_file_failure(tmp_path):
    directory = tmp_path / "nested"
    directory.mkdir()
    args = make_args(file=directory)

    with pytest.raises(SystemExit) as exc:
        image_cli.load_prompt_from_args(args)

    assert "Failed to read prompt file" in str(exc.value)


def test_load_prompt_from_stdin(monkeypatch):
    args = make_args()
    monkeypatch.setattr(sys, "stdin", io.StringIO("  A scenic landscape  \n"))

    prompt = image_cli.load_prompt_from_args(args)

    assert prompt == "A scenic landscape"


def test_load_prompt_keyboard_interrupt(monkeypatch):
    args = make_args()

    class Boom:
        def read(self):
            raise KeyboardInterrupt

    monkeypatch.setattr(sys, "stdin", Boom())

    with pytest.raises(SystemExit) as exc:
        image_cli.load_prompt_from_args(args)

    assert "Prompt entry cancelled" in str(exc.value)


def test_load_prompt_no_input(monkeypatch):
    args = make_args()
    monkeypatch.setattr(sys, "stdin", io.StringIO("   \n   "))

    with pytest.raises(SystemExit) as exc:
        image_cli.load_prompt_from_args(args)

    assert "No prompt" in str(exc.value)


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


def test_generate_image_no_data_entries():
    client = DummyClient([])

    with pytest.raises(SystemExit) as exc:
        image_cli.generate_image(client, "prompt", "model")

    assert "No image data" in str(exc.value)


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


def test_instantiate_client_returns_openai(monkeypatch):
    client = image_cli.instantiate_client("token")

    assert isinstance(client, image_cli.OpenAI)


def test_resolve_output_path_returns_explicit_path():
    target = Path("custom.png")
    args = make_args(output=target)

    assert image_cli.resolve_output_path(args) == target


def test_resolve_output_path_uses_timestamp(monkeypatch):
    class _DummyNow:
        def strftime(self, fmt: str) -> str:
            return "20240101-120000"

    class _DummyDatetime:
        @classmethod
        def now(cls, tz=None):
            return _DummyNow()

    monkeypatch.setattr(image_cli, "datetime", _DummyDatetime)

    args = make_args(output=None)

    path = image_cli.resolve_output_path(args)

    assert path.name == "image_result_20240101-120000.png"


def test_save_image_error(monkeypatch, tmp_path):
    destination = tmp_path / "image.png"

    original_open = Path.open

    def fake_open(self, *args, **kwargs):
        if self == destination:
            raise OSError("disk full")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fake_open)

    encoded = base64.b64encode(b"data").decode("ascii")

    with pytest.raises(SystemExit) as exc:
        image_cli.save_image(encoded, destination)

    assert "Failed to save image" in str(exc.value)


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


def test_main_generates_image(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "token")

    created = {}

    def fake_instantiate(api_key: str):
        created["api_key"] = api_key
        return object()

    def fake_generate(client, prompt: str, model: str) -> str:
        created["generate"] = (client, prompt, model)
        return base64.b64encode(b"pixels").decode("ascii")

    output_path = tmp_path / "result.png"

    def fake_resolve(args, suffix=".png"):
        created["resolved"] = True
        return output_path

    saved = {}

    def fake_save_image(data: str, destination: Path) -> None:
        saved["data"] = base64.b64decode(data)
        saved["destination"] = destination

    monkeypatch.setattr(image_cli, "instantiate_client", fake_instantiate)
    monkeypatch.setattr(image_cli, "generate_image", fake_generate)
    monkeypatch.setattr(image_cli, "resolve_output_path", fake_resolve)
    monkeypatch.setattr(image_cli, "save_image", fake_save_image)

    image_cli.main(["--prompt", "Sunset"])

    assert created["api_key"] == "token"
    assert created["generate"][1] == "Sunset"
    assert saved["destination"] == output_path
    assert saved["data"] == b"pixels"


def test_image_cli_import_error(monkeypatch):
    monkeypatch.delitem(sys.modules, "openai", raising=False)

    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "openai":
            raise ImportError("missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    module_name = "image_cli_import_test"
    loader = importlib.machinery.SourceFileLoader(module_name, str(PROJECT_ROOT / "image_cli.py"))
    spec = importlib.util.spec_from_loader(module_name, loader)
    module = importlib.util.module_from_spec(spec)

    with pytest.raises(SystemExit) as exc:
        loader.exec_module(module)

    assert "OpenAI Python SDK" in str(exc.value)
