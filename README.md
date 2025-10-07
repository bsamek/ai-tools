# ai-tools

## Image CLI

Generate images using gpt-image-1:

```bash
uv run image_cli.py --prompt "a gray tabby cat hugging an otter with an orange scarf"
```

Requirements:
- `uv` installed (dependencies are declared inline; no manual `pip install` needed).
- `OPENAI_API_KEY` exported (or store it in a local `.env` file).

The tool will generate an image and save it with a timestamp. Use `--output` to specify a custom filename:

```bash
uv run image_cli.py --prompt "sunset over mountains, landscape" --output sunset.png
```

For landscape/portrait orientation, include it in your prompt (e.g., "landscape orientation" or "portrait style").

Use `--help` for more options:
- `--file` - Read prompt from a file
- `--model` - Override model (defaults to gpt-5-mini)
- `--dry-run` - Preview without generating

## Sora CLI

Generate Sora videos with prompt refinement support:

```bash
uv run sora_cli.py --prompt "Describe your scene here" --dry-run
```

Requirements:
- `uv` installed (dependencies are declared inline; no manual `pip install` needed).
- `OPENAI_API_KEY` exported and provisioned for Sora access (or store it in a local `.env` file alongside `sora_cli.py`).

Run without `--dry-run` to submit a job, then wait for the video to download into the current directory. Use `--help` for advanced options (duration, output size, auto-approval, etc.).
Add `--skip-refinement` to send your prompt directly to Sora without the GPT-5 refinement pass.
Add `--refine-only` to run the refinement step (with GPT-5) and print the improved prompt without generating video.
