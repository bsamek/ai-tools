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
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

try:
    from openai import OpenAI, OpenAIError  # type: ignore
except ImportError as exc:  # pragma: no cover - runtime failure path
    raise SystemExit(
        "The OpenAI Python SDK is required. Install with `uv pip install openai`."
    ) from exc


DEFAULT_DURATION = 10
DEFAULT_SIZE = "1920x1080"
DEFAULT_SORA_MODEL = "sora-2"
DEFAULT_REASONING_MODEL = "gpt-5"


PROMPT_GUIDANCE = textwrap.dedent(
    """
    Focus on concrete visual descriptions covering:
    - Subject(s) with appearance, clothing, and behavior details.
    - Environment or setting, including time of day and atmosphere.
    - Camera perspective, motion, framing, and transitions.
    - Lighting, color palette, and mood.
    - Style (photorealistic, cinematic, animation) and post-processing notes.
    - Desired duration, aspect ratio, and output format hints.
    """
).strip()


@dataclass
class PromptReview:
    original: str
    improved: str
    analysis: str = ""
    tips: str = ""


@dataclass
class SoraJobStatus:
    status: str
    progress: Optional[float] = None
    eta_seconds: Optional[float] = None


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
    parser.add_argument("--output", type=Path, help="Destination filename for the video.")
    parser.add_argument(
        "--duration",
        type=int,
        default=DEFAULT_DURATION,
        help="Target duration in seconds.",
    )
    parser.add_argument(
        "--size",
        default=DEFAULT_SIZE,
        help="Frame size WIDTHxHEIGHT (e.g. 1920x1080).",
    )
    parser.add_argument(
        "--aspect-ratio",
        help="Optional aspect ratio hint (e.g. 16:9, 9:16, 1:1).",
    )
    parser.add_argument("--fps", type=int, help="Frames per second hint.")
    parser.add_argument("--seed", type=int, help="Random seed for deterministic runs.")
    parser.add_argument("--format", default="mp4", help="Desired output container.")
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
        analysis = data.get("analysis", "")
        tips = data.get("tips", "")
        return improved.strip(), analysis.strip(), tips.strip()

    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"[warn] Failed to parse refinement output: {exc}.")
        return fallback_prompt, "", ""


def choose_prompt(review: PromptReview, auto_approve: bool) -> str:
    if auto_approve or review.improved == review.original:
        print("Using refined prompt.")
        return review.improved

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


def build_video_request_payload(
    prompt: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    width_height = parse_size(args.size)
    video_params: Dict[str, Any] = {
        "format": args.format,
        "duration": args.duration,
    }
    if width_height:
        width, height = width_height
        video_params["width"] = width
        video_params["height"] = height
    if args.aspect_ratio:
        video_params["aspect_ratio"] = args.aspect_ratio
    if args.fps:
        video_params["fps"] = args.fps
    if args.seed is not None:
        video_params["seed"] = args.seed

    payload: Dict[str, Any] = {
        "model": args.sora_model,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                ],
            }
        ],
        "modalities": ["video"],
        "video": video_params,
    }
    return payload


def parse_size(size_str: Optional[str]) -> Optional[Tuple[int, int]]:
    if not size_str:
        return None
    try:
        width_str, height_str = size_str.lower().split("x")
        width, height = int(width_str), int(height_str)
        if width <= 0 or height <= 0:
            raise ValueError
        return width, height
    except (ValueError, AttributeError):
        print(f"[warn] Unrecognized size '{size_str}'. Falling back to Sora defaults.")
        return None


def create_sora_job(
    client: OpenAI,
    payload: Dict[str, Any],
) -> Any:
    try:
        return client.responses.create(**payload)
    except OpenAIError as exc:
        raise SystemExit(f"Failed to create Sora video job: {exc}") from exc


def poll_sora_job(
    client: OpenAI,
    response_id: str,
    poll_interval: float,
    timeout: float,
) -> Any:
    start_time = time.monotonic()
    last_status = ""

    while True:
        try:
            job = client.responses.retrieve(response_id)
        except OpenAIError as exc:
            raise SystemExit(f"Polling failed: {exc}") from exc

        status = getattr(job, "status", None) or job.get("status")  # type: ignore
        progress = extract_progress(job)
        if status != last_status or progress is not None:
            progress_note = f" ({progress*100:.1f}% complete)" if progress is not None else ""
            print(f"[status] {status}{progress_note}")
            last_status = status

        if status in {"completed", "failed", "cancelled"}:
            return job

        if time.monotonic() - start_time > timeout:
            raise SystemExit(
                f"Timed out waiting for job {response_id} after {timeout} seconds."
            )

        time.sleep(poll_interval)


def extract_progress(job: Any) -> Optional[float]:
    progress = getattr(job, "progress", None)
    if progress is None and isinstance(job, dict):
        progress = job.get("progress")
    if progress is None:
        metadata = getattr(job, "metadata", None) or {}
        if isinstance(metadata, dict):
            progress = metadata.get("progress")

    if progress is None:
        return None
    try:
        return float(progress)
    except (TypeError, ValueError):
        return None


def resolve_output_path(args: argparse.Namespace, suffix: str = ".mp4") -> Path:
    if args.output:
        return args.output
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    return Path(f"sora_result_{timestamp}{suffix}")


def extract_asset_id(job: Any) -> Optional[str]:
    """
    Walk the outputs to find a video asset identifier.
    """
    output_blocks = getattr(job, "output", None) or getattr(job, "outputs", None)
    if not output_blocks and isinstance(job, dict):
        output_blocks = job.get("output") or job.get("outputs")

    if not output_blocks:
        return None

    def search_content(content_items: Iterable[Dict[str, Any]]) -> Optional[str]:
        for item in content_items:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type in {"output_video", "video"} and item.get("asset_id"):
                return item["asset_id"]
            if item_type == "content" and isinstance(item.get("content"), list):
                nested = search_content(item["content"])
                if nested:
                    return nested
        return None

    for block in output_blocks:
        content_list = None
        if isinstance(block, dict):
            content_list = block.get("content")
        if isinstance(content_list, list):
            asset_id = search_content(content_list)
            if asset_id:
                return asset_id
    return None


def download_asset(client: OpenAI, asset_id: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        stream = client.assets.content(asset_id)
    except AttributeError:
        raise SystemExit(
            "The OpenAI client does not expose assets.content; update the SDK."
        )
    except OpenAIError as exc:
        raise SystemExit(f"Failed to access asset {asset_id}: {exc}") from exc

    try:
        with destination.open("wb") as handle:
            for chunk in stream:
                if not chunk:
                    continue
                if isinstance(chunk, bytes):
                    handle.write(chunk)
                elif hasattr(chunk, "decode"):
                    handle.write(chunk.decode("utf-8").encode("utf-8"))
                else:
                    data = getattr(chunk, "data", None)
                    if isinstance(data, bytes):
                        handle.write(data)
                    else:
                        raise TypeError(f"Unexpected chunk type: {type(chunk)}")
    except TypeError as exc:
        raise SystemExit(
            f"Unable to stream asset {asset_id}; received unsupported chunk type: {exc}"
        ) from exc


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
    prompt = load_prompt_from_args(args)

    client: Optional[OpenAI] = None
    review: PromptReview

    if args.dry_run:
        review = PromptReview(original=prompt, improved=prompt)
    else:
        api_key = ensure_api_key()
        client = instantiate_client(api_key)

        print("\nRefining prompt with GPT-5...\n")
        review = improve_prompt(client, prompt, model=args.refinement_model)

    final_prompt = choose_prompt(review, auto_approve=args.auto_approve)
    print("\nFinal prompt selected.\n")

    payload = build_video_request_payload(final_prompt, args)

    if args.dry_run:
        print("[dry-run] Payload to send:")
        print(json.dumps(payload, indent=2))
        return

    print("Submitting Sora video generation job...")
    if client is None:  # pragma: no cover - defensive guard
        api_key = ensure_api_key()
        client = instantiate_client(api_key)

    response = create_sora_job(client, payload)
    response_id = getattr(response, "id", None) or response.get("id")  # type: ignore
    if not response_id:
        raise SystemExit("API response missing ID field.")

    print(f"Job ID: {response_id}")
    job = poll_sora_job(
        client,
        response_id=response_id,
        poll_interval=args.poll_interval,
        timeout=args.timeout,
    )

    job_status = getattr(job, "status", None) or job.get("status")  # type: ignore
    if job_status != "completed":
        raise SystemExit(f"Job ended with status: {job_status}")

    asset_id = extract_asset_id(job)
    if not asset_id:
        raise SystemExit("Completed job did not include a downloadable asset.")

    target_path = resolve_output_path(args, suffix=f".{args.format.lstrip('.')}")
    print(f"Downloading video asset {asset_id} to {target_path}...")
    download_asset(client, asset_id, target_path)

    print(f"Video saved to {target_path.resolve()}")
    cost_summary = summarize_cost(job)
    if cost_summary:
        print(f"Usage: {cost_summary}")


if __name__ == "__main__":
    main()
