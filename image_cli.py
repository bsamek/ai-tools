#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "openai>=1.0.0",
# ]
# ///
"""
Command-line helper for generating images via the OpenAI API using gpt-image-1.

Workflow:
1. Collect a user prompt via CLI flags or interactive input.
2. Generate an image using gpt-image-1 through the Responses API.
3. Save the image to a file.

The script is intended to be executed with `uv run image_cli.py [OPTIONS]`.
"""

from __future__ import annotations

import argparse
import base64
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

try:
    from openai import OpenAI, OpenAIError  # type: ignore
except ImportError as exc:  # pragma: no cover - runtime failure path
    raise SystemExit(
        "The OpenAI Python SDK is required. Install with `uv pip install openai`."
    ) from exc


DEFAULT_MODEL = "gpt-5-mini"


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate images with gpt-image-1 via the Responses API.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--prompt", help="Inline prompt text.")
    parser.add_argument("--file", type=Path, help="Path to a file containing the prompt.")
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Optional dotenv-style file to populate OPENAI_API_KEY if the environment is unset.",
    )
    parser.add_argument("--output", type=Path, help="Destination filename for the image.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model to use for image generation (e.g., gpt-4.1-mini, gpt-5-mini).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the request details instead of calling the API.",
    )
    return parser.parse_args(argv)


def ensure_api_key() -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "OPENAI_API_KEY environment variable is required to call the OpenAI API."
        )
    return api_key


def populate_env_from_file(env_path: Path) -> None:
    if not env_path.exists() or not env_path.is_file():
        return
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            key = key.strip()
            if not key:
                continue
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)
    except OSError as exc:
        print(f"[warn] Failed to read env file {env_path}: {exc}")


def load_prompt_from_args(args: argparse.Namespace) -> str:
    if args.prompt:
        return args.prompt.strip()

    if args.file:
        try:
            return args.file.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise SystemExit(f"Failed to read prompt file {args.file}: {exc}") from exc

    print("Enter your image prompt. End with Ctrl-D (Unix) or Ctrl-Z (Windows).")
    try:
        user_input = sys.stdin.read()
    except KeyboardInterrupt:  # pragma: no cover - interactive path
        raise SystemExit("\nPrompt entry cancelled by user.")

    prompt = user_input.strip()
    if not prompt:
        raise SystemExit("No prompt provided.")
    return prompt


def instantiate_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)


def generate_image(client: OpenAI, prompt: str, model: str) -> str:
    """
    Generate an image using the Responses API with image_generation tool.
    Returns the base64-encoded image data.
    """
    try:
        response = client.responses.create(
            model=model,
            input=prompt,
            tools=[{"type": "image_generation"}],
        )
    except OpenAIError as exc:
        raise SystemExit(f"Failed to generate image: {exc}") from exc

    # Extract image data from response
    image_data = [
        output.result
        for output in response.output
        if output.type == "image_generation_call"
    ]

    if not image_data:
        raise SystemExit("No image was generated in the response.")

    return image_data[0]


def resolve_output_path(args: argparse.Namespace, suffix: str = ".png") -> Path:
    if args.output:
        return args.output
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return Path(f"image_result_{timestamp}{suffix}")


def save_image(image_base64: str, destination: Path) -> None:
    """Save base64-encoded image data to a file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("wb") as f:
            f.write(base64.b64decode(image_base64))
    except Exception as exc:
        raise SystemExit(f"Failed to save image to {destination}: {exc}") from exc


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)

    if args.env_file:
        populate_env_from_file(args.env_file)

    prompt = load_prompt_from_args(args)

    if args.dry_run:
        print("[dry-run] Image generation request:")
        print(f"  Model: {args.model}")
        print(f"  Prompt: {prompt}")
        print(f"  Tools: [image_generation]")
        return

    api_key = ensure_api_key()
    client = instantiate_client(api_key)

    print(f"\nGenerating image using {args.model}...\n")
    image_base64 = generate_image(client, prompt, args.model)

    target_path = resolve_output_path(args)
    print(f"Saving image to {target_path}...")
    save_image(image_base64, target_path)

    print(f"Image saved to {target_path.resolve()}")


if __name__ == "__main__":
    main()
