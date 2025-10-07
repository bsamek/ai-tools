#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "openai>=1.0.0",
# ]
# ///
"""
Command-line helper for creating Sora videos via the OpenAI API.

Workflow:
1. Collect a user prompt via CLI flags or interactive input.
2. Ask GPT-5 to refine the prompt following Sora best practices.
3. Let the user choose the original or refined prompt (auto-approval optional).
4. Create a Sora video generation job, poll until completion, then download the video.
5. Print a summary, including estimated cost.

The script is intended to be executed with `uv run sora_cli.py [OPTIONS]`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

try:
    from openai import OpenAI, OpenAIError  # type: ignore
except ImportError as exc:  # pragma: no cover - runtime failure path
    raise SystemExit(
        "The OpenAI Python SDK is required. Install with `uv pip install openai`."
    ) from exc


DEFAULT_DURATION = 8
DEFAULT_SIZE = "1280x720"
DEFAULT_SORA_MODEL = "sora-2"
DEFAULT_REASONING_MODEL = "gpt-5"

ALLOWED_VIDEO_SIZES = {
    "1280x720",
    "720x1280",
    "1792x1024",
    "1024x1792",
}

ALLOWED_VIDEO_SECONDS = {4, 8, 12}


PROMPT_GUIDANCE = textwrap.dedent(
    """
    Focus on concrete visual descriptions covering:
    - Subject(s) with appearance, clothing, and behavior details.
    - Environment or setting, including time of day and atmosphere.
    - Camera perspective, motion, framing, and transitions.
    - Lighting, color palette, and mood.
    - Style (photorealistic, cinematic, animation) and post-processing notes.
    - Aspect ratio and output format hints.
    - Do not mention duration; it is configured separately via the API parameters.
    """
).strip()


@dataclass
class PromptReview:
    original: str
    improved: str
    analysis: str = ""
    tips: str = ""


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Sora videos with prompt refinement assistance.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--prompt", help="Inline prompt text.")
    parser.add_argument("--file", type=Path, help="Path to a file containing the prompt.")
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="Skip confirmation and use the refined prompt automatically.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Optional dotenv-style file to populate OPENAI_API_KEY if the environment is unset.",
    )
    parser.add_argument(
        "--skip-refinement",
        action="store_true",
        help="Bypass GPT-5 prompt refinement and send the prompt as-is.",
    )
    parser.add_argument(
        "--refine-only",
        action="store_true",
        help="Run the prompt refinement workflow and exit without calling Sora.",
    )
    parser.add_argument("--output", type=Path, help="Destination filename for the video.")
    parser.add_argument(
        "--duration",
        type=int,
        default=DEFAULT_DURATION,
        help="Target duration in seconds (allowed values: 4, 8, 12).",
    )
    parser.add_argument(
        "--size",
        choices=sorted(ALLOWED_VIDEO_SIZES),
        default=DEFAULT_SIZE,
        help="Predefined output resolution.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=15 * 60,
        help="Maximum time to wait for job completion (seconds).",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=5.0,
        help="Polling interval in seconds.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the request payload instead of calling Sora.",
    )
    parser.add_argument(
        "--sora-model",
        default=DEFAULT_SORA_MODEL,
        help="Override Sora model identifier.",
    )
    parser.add_argument(
        "--refinement-model",
        default=DEFAULT_REASONING_MODEL,
        help="Override reasoning model identifier.",
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

    print("Enter your video prompt. End with Ctrl-D (Unix) or Ctrl-Z (Windows).")
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


def improve_prompt(
    client: OpenAI, prompt: str, model: str = DEFAULT_REASONING_MODEL
) -> PromptReview:
    system_instructions = (
        "You are an expert Sora prompt engineer. "
        "Rewrite prompts to maximize video fidelity and adhere to best practices. "
        "Respond ONLY with compact JSON containing exactly the keys "
        '`improved_prompt`, `analysis`, and `tips`.'
    )

    request_input = [
        {
            "role": "system",
            "content": [{"type": "input_text", "text": system_instructions}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": (
                        "Original prompt:\n"
                        f"{prompt}\n\n"
                        "Prompting guidance:\n"
                        f"{PROMPT_GUIDANCE}\n\n"
                        "Rewrite the prompt following this guidance."
                    ),
                }
            ],
        },
    ]

    try:
        response = client.responses.create(
            model=model,
            input=request_input,
        )
    except OpenAIError as exc:
        print(f"[warn] Prompt refinement failed: {exc}. Using original prompt.")
        return PromptReview(original=prompt, improved=prompt)

    improved, analysis, tips = parse_prompt_review(response, prompt)
    return PromptReview(original=prompt, improved=improved, analysis=analysis, tips=tips)


def parse_prompt_review(response: Any, fallback_prompt: str) -> Tuple[str, str, str]:
    """
    Extract JSON fields from the refinement response. Falls back to the original prompt
    if parsing fails or the expected keys are missing.
    """
    try:
        texts = []

        output_blocks = getattr(response, "output", None) or getattr(
            response, "outputs", None
        )
        if output_blocks:
            for block in output_blocks:
                content_list = block.get("content") if isinstance(block, dict) else None
                if not content_list:
                    continue
                for content_item in content_list:
                    if content_item.get("type") == "output_text":
                        texts.append(content_item.get("text", ""))
                    elif content_item.get("type") == "text":
                        texts.append(content_item.get("text", ""))

        output_text = getattr(response, "output_text", None)
        if output_text:
            texts.append(str(output_text))

        if not texts:
            raise ValueError("Response did not contain parseable text content.")

        raw_text = "\n".join(filter(None, texts)).strip()
        data = json.loads(raw_text)
        improved = data.get("improved_prompt") or fallback_prompt
        analysis = _coerce_to_text(data.get("analysis"))
        tips = _coerce_to_text(data.get("tips"))
        return improved.strip(), analysis.strip(), tips.strip()

    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"[warn] Failed to parse refinement output: {exc}.")
        return fallback_prompt, "", ""


def _coerce_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        parts = []
        for item in value:
            text = _coerce_to_text(item).strip()
            if text:
                parts.append(text)
        return "\n".join(parts)
    if isinstance(value, dict):
        try:
            return json.dumps(value, indent=2)
        except TypeError:
            return str(value)
    return str(value)


def choose_prompt(review: PromptReview, auto_approve: bool) -> str:
    if auto_approve or review.improved == review.original:
        print("Using refined prompt.")
        return review.improved

    display_prompt_review(review)

    choices = {
        "1": review.original,
        "2": review.improved,
    }

    while True:  # pragma: no branch - simple loop
        print("\nSelect prompt: [1] original, [2] refined, [3] re-enter manually")
        selection = input("> ").strip()
        if selection in ("1", "2"):
            return choices[selection]
        if selection == "3":
            return get_manual_prompt()
        print("Invalid selection. Please enter 1, 2, or 3.")


def get_manual_prompt() -> str:
    print("Enter your prompt override. Finish with a blank line.")
    lines = []
    while True:
        line = input()
        if not line.strip():
            break
        lines.append(line)
    prompt = "\n".join(lines).strip()
    if not prompt:
        raise SystemExit("Manual prompt cannot be empty.")
    return prompt


def display_prompt_review(review: PromptReview) -> None:
    print("\n--- Prompt Comparison ---")
    print("Original:\n")
    print(textwrap.indent(review.original, prefix="  "))
    print("\nRefined:\n")
    print(textwrap.indent(review.improved, prefix="  "))
    if review.analysis:
        print("\nAnalysis:\n")
        print(textwrap.indent(review.analysis, prefix="  "))
    if review.tips:
        print("\nTips:\n")
        print(textwrap.indent(review.tips, prefix="  "))


def build_video_request_payload(
    prompt: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    request: Dict[str, Any] = {
        "prompt": prompt,
    }
    if args.sora_model:
        request["model"] = args.sora_model

    seconds = normalize_duration(args.duration)
    if seconds is not None:
        request["seconds"] = seconds

    size = normalize_size(args.size)
    if size is not None:
        request["size"] = size

    return request


def normalize_duration(seconds: Optional[int]) -> Optional[str]:
    if seconds is None:
        return None
    if seconds in ALLOWED_VIDEO_SECONDS:
        return str(seconds)
    print(
        f"[warn] Unsupported duration '{seconds}'. Falling back to {DEFAULT_DURATION} seconds."
    )
    return str(DEFAULT_DURATION)


def normalize_size(size: Optional[str]) -> Optional[str]:
    if not size:
        return None
    normalized = size.lower()
    if normalized in ALLOWED_VIDEO_SIZES:
        return normalized
    print(f"[warn] Unsupported size '{size}'. Falling back to {DEFAULT_SIZE}.")
    return DEFAULT_SIZE


def create_sora_job(
    client: OpenAI,
    request: Dict[str, Any],
) -> Any:
    try:
        return client.videos.create(**request)
    except OpenAIError as exc:
        raise SystemExit(f"Failed to create Sora video job: {exc}") from exc


def poll_sora_job(
    client: OpenAI,
    initial_job: Any,
    poll_interval: float,
    timeout: float,
) -> Any:
    start_time = time.monotonic()
    last_status: Optional[str] = None
    last_progress: Optional[int] = None
    job = initial_job

    while True:  # pragma: no branch - manual polling
        status = getattr(job, "status", None)
        progress = getattr(job, "progress", None)
        if status != last_status or progress != last_progress:
            progress_note = f" ({progress}%)" if progress is not None else ""
            print(f"[status] {status}{progress_note}")
            last_status = status
            last_progress = progress

        if status in {"completed", "failed"}:
            return job

        if time.monotonic() - start_time > timeout:
            raise SystemExit(
                f"Timed out waiting for job {getattr(job, 'id', 'unknown')} after {timeout} seconds."
            )

        time.sleep(poll_interval)
        try:
            job = client.videos.retrieve(getattr(job, "id"))
        except OpenAIError as exc:
            raise SystemExit(f"Polling failed: {exc}") from exc


def resolve_output_path(args: argparse.Namespace, suffix: str = ".mp4") -> Path:
    if args.output:
        return args.output
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return Path(f"sora_result_{timestamp}{suffix}")


def download_video_asset(client: OpenAI, video_id: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        stream = client.videos.download_content(video_id, variant="video")
    except OpenAIError as exc:
        raise SystemExit(f"Failed to download video {video_id}: {exc}") from exc

    with destination.open("wb") as handle:
        for chunk in stream.iter_bytes():
            if chunk:
                handle.write(chunk)


def summarize_cost(job: Any) -> Optional[str]:
    usage = getattr(job, "usage", None)
    if usage is None and isinstance(job, dict):
        usage = job.get("usage")
    if usage is None:
        return None

    if hasattr(usage, "to_dict_recursive"):
        usage_dict = usage.to_dict_recursive()
    elif isinstance(usage, dict):
        usage_dict = usage
    else:
        usage_dict = dict(usage)  # type: ignore[arg-type]

    credits = usage_dict.get("total_credits")
    tokens = usage_dict.get("total_tokens")
    usd = usage_dict.get("total_cost_usd")

    parts = []
    if credits is not None:
        parts.append(f"credits: {credits}")
    if tokens is not None:
        parts.append(f"tokens: {tokens}")
    if usd is not None:
        parts.append(f"≈ ${usd:.4f} USD")

    if not parts:
        return None
    return " | ".join(parts)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)

    if args.env_file:
        populate_env_from_file(args.env_file)

    prompt = load_prompt_from_args(args)

    client: Optional[OpenAI] = None
    review: PromptReview

    if args.skip_refinement:
        print("Skipping prompt refinement (flag provided).")
        review = PromptReview(original=prompt, improved=prompt)
    elif args.dry_run:
        review = PromptReview(original=prompt, improved=prompt)
    else:
        api_key = ensure_api_key()
        client = instantiate_client(api_key)

        print("\nRefining prompt with GPT-5...\n")
        review = improve_prompt(client, prompt, model=args.refinement_model)

    if args.refine_only:
        display_prompt_review(review)
        print("\nRefined prompt ready to copy:\n")
        print(textwrap.indent(review.improved, prefix="  "))
        return

    final_prompt = choose_prompt(review, auto_approve=args.auto_approve)
    print("\nFinal prompt selected.\n")

    request_payload = build_video_request_payload(final_prompt, args)

    if args.dry_run:
        print("[dry-run] Payload to send:")
        print(json.dumps(request_payload, indent=2))
        return

    print("Submitting Sora video generation job...")
    if client is None:  # pragma: no cover - defensive guard
        api_key = ensure_api_key()
        client = instantiate_client(api_key)

    job = create_sora_job(client, request_payload)
    job_id = getattr(job, "id", None)
    if not job_id:
        raise SystemExit("API response missing video ID field.")

    print(f"Job ID: {job_id}")
    job = poll_sora_job(
        client,
        initial_job=job,
        poll_interval=args.poll_interval,
        timeout=args.timeout,
    )

    job_status = getattr(job, "status", None)
    if job_status != "completed":
        error = getattr(job, "error", None)
        if error:
            raise SystemExit(f"Job ended with status {job_status}: {error}")
        raise SystemExit(f"Job ended with status: {job_status}")

    target_path = resolve_output_path(args)
    print(f"Downloading video {job_id} to {target_path}...")
    download_video_asset(client, job_id, target_path)

    print(f"Video saved to {target_path.resolve()}")
    cost_summary = summarize_cost(job)
    if cost_summary:
        print(f"Usage: {cost_summary}")


if __name__ == "__main__":
    main()
