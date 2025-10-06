# Sora CLI Implementation Plan

## Objectives
- Build a single-file Python CLI (e.g. `sora_cli.py`) that can be executed with `uv run sora_cli.py`.
- Accept a user-provided video prompt, improve it with GPT-5 using OpenAI's Responses API, and let the user choose between the original and refined prompt.
- Create a Sora video generation job using the approved prompt, poll the job until completion, then download the resulting video into the current directory.
- Surface helpful guidance based on <https://platform.openai.com/docs/guides/video-generation> so the user writes effective prompts.

## Key References
- OpenAI Video Generation Guide: <https://platform.openai.com/docs/guides/video-generation>
- OpenAI Python SDK (>=1.0) docs: focus on Responses API for GPT-5 reasoning, and the Sora video endpoints exposed under `client.responses` and `client.assets`.
- Internal Sora API notes: `video.generate` requests accept `model: "sora-2"` plus `prompt` metadata, with preliminary `aspect_ratio`, `duration`, and `output_format` hints.

## CLI Design
- **Entry point:** `python sora_cli.py [--prompt PROMPT] [--file PROMPT_FILE] [--auto-approve] [--output OUTPUT_PATH]`.
- **Execution via uv:** document usage `uv run sora_cli.py --prompt "..."`
- **Prompt acquisition:** read from CLI flag or fallback to an interactive `input()` flow. Offer optional reading from stdin/file.
- **Prompt refinement:** 
  - Call GPT-5 through `client.responses.create()` with instructions to act as a Sora prompt engineer.
  - Include the original prompt, guidance from the docs (e.g., specify subject, motion, camera movement, lighting, mood, duration, aspect ratio).
  - Return a JSON structure with `improved_prompt`, `analysis`, `tips`.
- **Approval flow:** Display original and improved prompts with short analysis. Ask user to choose `[1] original`, `[2] improved`, or `[3] edit manually`. Loop until explicit confirmation unless `--auto-approve` is set.
- **Job creation:** Submit `client.responses.create()` (or the dedicated Sora endpoint if exposed) with `model="sora-2"` and necessary parameters (`prompt`, `duration`, `aspect_ratio`, `fps`, etc.). Capture the job ID.
- **Polling:** Periodically hit `client.responses.retrieve(job_id)` (or `client.responses.get`) until status is `completed` or `failed`. Respect rate limits with exponential/backoff (start 2s, cap 30s).
- **Download:** On completion, inspect `output[0].content` to find the video asset URL or ID. Download via the asset endpoint, saving to `OUTPUT_PATH` (default `./sora_result_<timestamp>.mp4`). Stream downloads to avoid large-memory usage.
- **Progress reporting:** Print concise status updates, including queue state and percent progress if provided.
- **Cost reporting:** After job completion, surface API usage (tokens, credits, USD estimate) so the user knows what was spent.

## Implementation Steps
1. **Scaffold CLI file**
   - Create `sora_cli.py` with `if __name__ == "__main__":` entry point.
   - Use `argparse` (keeps single-file footprint light) for CLI argument parsing.
   - Load `OPENAI_API_KEY` from environment; fail fast with actionable error if missing.
2. **Add prompt intake + GPT-5 refinement**
   - Implement helper `improve_prompt(prompt: str) -> PromptReview`.
   - Compose system and user messages aligning with Sora prompt best practices (e.g., include subject, environment, camera, motion, lighting, realism level, style).
   - Parse JSON output safely, fall back to raw text if parsing fails.
3. **Prompt approval interface**
   - Print side-by-side summary (maybe numbered bullet list). Accept user choice via `input()`. Support `--auto-approve` to select improved prompt automatically.
4. **Build Sora request payload**
  - Capture optional args for `duration`, `aspect_ratio`, `size`, `fps`, `seed`, `format`.
  - Validate values align with docs (e.g., duration between 1–60s, aspect ratio `16:9`, `9:16`, `1:1`).
  - Construct `responses.create` call with `input=[{"role": "user", "content": [{"type": "text", "text": final_prompt}]}]` and video generation tool parameters.
  - Default to `size="1920x1080"` (16:9) and `duration=10` seconds; allow overrides via CLI flags.
5. **Polling loop**
   - Poll job status until `status in {"completed", "failed", "cancelled"}`.
   - Print status updates; handle `requires_action` by surfacing message and exiting gracefully.
   - Implement timeout (configurable via `--timeout`).
6. **Download artifact**
   - Once job completes, fetch the asset URL via `client.responses.retrieve` result.
   - Stream download with `requests` (or `httpx` bundled via `uv`). Support custom filename via `--output`.
   - Report usage cost back to the user by reading `response.usage` (tokens, credits) and multiplying by published pricing when available.
7. **Error handling & retries**
   - Wrap API calls in try/except, display actionable messages.
   - Handle rate limit (HTTP 429) with exponential backoff.
8. **Documentation**
   - Add usage block in README snippet describing `uv run`.
   - Include notes about needing `OPENAI_API_KEY` and Sora access.
9. **Testing**
   - Create dry-run mode (`--dry-run`) to skip actual Sora call and print constructed payload (useful without API access).
   - Optionally add unit tests for prompt refinement parsing (in future separate file).

## Data Structures
- `PromptReview` dataclass with fields: `original`, `improved`, `analysis`, `tips`.
- `SoraJobStatus` simple dict capturing `status`, `progress`, `eta_seconds`.

## Open Questions / Assumptions
- Assume GPT-5 model ID `gpt-5` (adjust per availability).
- Assume Sora Responses API returns `output` with `content[0].asset_id` and that a follow-up `client.assets.content(asset_id)` streams binary data.
- Need confirmation whether poll interval and max duration meet product expectations.

## Next Actions
- Implement `sora_cli.py` following plan.
- Smoke test with `uv run sora_cli.py --prompt "..." --dry-run` to verify flow sans API credentials.
- Document limitations and follow up when Sora API schema stabilizes.
