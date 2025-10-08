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


def mock_read_text_error(*args, **kwargs):
    raise OSError("Permission denied")


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


def test_populate_env_from_file_does_not_overwrite(tmp_path, monkeypatch):
    env_file = tmp_path / "test.env"
    env_file.write_text("OPENAI_API_KEY=file_key", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "env_key")

    image_cli.populate_env_from_file(env_file)

    assert os.environ["OPENAI_API_KEY"] == "env_key"


def test_populate_env_from_file_ignores_malformed_lines(tmp_path, monkeypatch):
    env_file = tmp_path / "test.env"
    env_file.write_text(
        "# Comment\n\n=no_key\nno_value=\n'spaced-key' = 'spaced-value'\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("spaced-key", raising=False)

    image_cli.populate_env_from_file(env_file)

    assert "OPENAI_API_KEY" not in os.environ
    assert os.environ["spaced-key"] == "spaced-value"


def test_populate_env_from_file_handles_missing_file(tmp_path):
    # This should not raise an error.
    image_cli.populate_env_from_file(tmp_path / "nonexistent.env")


def test_populate_env_from_file_handles_read_error(tmp_path, monkeypatch, capsys):
    env_file = tmp_path / "protected.env"
    env_file.touch()
    monkeypatch.setattr(Path, "read_text", mock_read_text_error)

    image_cli.populate_env_from_file(env_file)

    captured = capsys.readouterr()
    assert "[warn] Failed to read env file" in captured.out


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


def test_load_prompt_from_args_file_read_error(tmp_path, monkeypatch):
    prompt_file = tmp_path / "unreadable.txt"
    prompt_file.touch()
    args = make_args(file=prompt_file)
    monkeypatch.setattr(Path, "read_text", mock_read_text_error)

    with pytest.raises(SystemExit) as exc:
        image_cli.load_prompt_from_args(args)
    assert "Failed to read prompt file" in str(exc.value)


def test_load_prompt_from_stdin_empty_input(monkeypatch):
    args = make_args()
    monkeypatch.setattr(sys, "stdin", io.StringIO("\n"))

    with pytest.raises(SystemExit) as exc:
        image_cli.load_prompt_from_args(args)
    assert "No prompt provided" in str(exc.value)


def test_load_prompt_from_stdin(monkeypatch):
    args = make_args()
    monkeypatch.setattr(sys, "stdin", io.StringIO("  A scenic landscape  \n"))

    prompt = image_cli.load_prompt_from_args(args)

    assert prompt == "A scenic landscape"


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


def test_generate_image_no_response_data():
    client = DummyClient([])

    with pytest.raises(SystemExit) as exc:
        image_cli.generate_image(client, "prompt", "model")

    assert "No image data returned" in str(exc.value)


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


def test_save_image_writes_file(tmp_path):
    destination = tmp_path / "image.png"
    encoded = base64.b64encode(b"image-bytes").decode("ascii")

    image_cli.save_image(encoded, destination)

    assert destination.read_bytes() == b"image-bytes"


def test_resolve_output_path_generates_default(monkeypatch):
    class MockDatetime:
        @classmethod
        def now(cls, *args, **kwargs):
            # A bit of a hack to mock both datetime.now() and the strftime call
            return types.SimpleNamespace(strftime=lambda fmt: "20240101-120000")

    monkeypatch.setattr(image_cli, "datetime", MockDatetime)
    args = make_args()

    path = image_cli.resolve_output_path(args)

    assert path == Path("image_result_20240101-120000.png")


def test_save_image_handles_write_error(tmp_path, monkeypatch):
    destination = tmp_path / "protected" / "image.png"
    encoded = base64.b64encode(b"image-bytes").decode("ascii")

    def mock_open_error(*args, **kwargs):
        raise OSError("Disk full")

    monkeypatch.setattr(Path, "open", mock_open_error)

    with pytest.raises(SystemExit) as exc:
        image_cli.save_image(encoded, destination)
    assert "Failed to save image" in str(exc.value)


def test_main_dry_run(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    args = ["--prompt", "A castle", "--dry-run", "--output", "result.png"]

    image_cli.main(args)

    captured = capsys.readouterr()
    assert "[dry-run]" in captured.out
    assert "A castle" in captured.out


def test_main_success_flow(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "fake_key")
    monkeypatch.chdir(tmp_path)

    # Mock client and API call
    mock_client = DummyClient([DummyImageData(base64.b64encode(b"fake_image_bytes").decode("ascii"))])
    monkeypatch.setattr(image_cli, "instantiate_client", lambda key: mock_client)

    # Mock output path to be predictable
    output_path = tmp_path / "output.png"
    monkeypatch.setattr(image_cli, "resolve_output_path", lambda args: output_path)

    args = ["--prompt", "A real image"]
    image_cli.main(args)

    captured = capsys.readouterr()
    assert "Generating image" in captured.out
    assert f"Saving image to {output_path}" in captured.out
    assert "Image saved" in captured.out
    assert output_path.read_bytes() == b"fake_image_bytes"
