# ai-tools

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
