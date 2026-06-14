# Med Taxonomizer

**Med Taxonomizer** is a research toolkit for inducing a reusable medical taxonomy from free-text records, reviewing the taxonomy locally, and applying it in a structured labeling workflow.

The public repository contains code, prompts, documentation, a synthetic example, and a local reviewer. It does **not** include project datasets, run logs, internal notes, or paper-run artifacts.

## Install

```bash
uv sync --extra dev
```

If `uv` is not available:

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

## Prepare input text

The taxonomy pipeline expects a parquet file with a single `text` column. Convert a CSV or parquet file with free-text records:

```bash
uv run python scripts/prepare_texts.py \
  --input examples/synthetic/free_text.csv \
  --output data/inputs/texts.parquet \
  --text-column text
```

Accepted default text column names are `text`, `free_text`, `description`, `complication`, and `note`. Use `--text-column` for other names.

## Run the taxonomy pipeline

Set an OpenRouter-compatible API key if you use hosted models:

```bash
export OPENROUTER_API_KEY=...
```

Run all phases:

```bash
./scripts/run_taxonomy_pipeline.sh \
  --input data/inputs/texts.parquet \
  --run-id example_tree \
  --model qwen/qwen3.5-27b \
  --target clinical_events \
  --domain-context "Clinical free-text fragments for taxonomy induction." \
  --resume
```

Artifacts are written under:

```text
scripts/data/runs/<run-id>/
```

`data/`, `outputs/`, `logs/`, and `notes/` are ignored by Git so local datasets and run artifacts are not committed accidentally.

## Optional local taxonomy review

Create a review packet from a taxonomy JSON file:

```bash
./scripts/review_taxonomy.sh \
  --taxonomy scripts/data/runs/<run-id>/intermediate/taxonomy_tree_final_qwen.json \
  --output-dir scripts/data/runs/<run-id>/review
```

This writes:

- `review_queue.csv`
- `taxonomy_tree_curated_template.json`
- `review_instructions.md`

Run the zero-build browser reviewer:

```bash
cd apps/reviewer
python3 -m http.server 5173
```

Open:

```text
http://127.0.0.1:5173/standalone.html
```

The browser reviewer loads taxonomy JSON locally and exports a review CSV. No hosted service or bundled dataset is required.

## Repository structure

```text
med_taxonomizer/
├── website/                 # GitHub Pages landing page
├── apps/reviewer/           # Optional local browser reviewer
├── lib/                     # Shared Python helpers
├── scripts/                 # Pipeline phases, runner, and review utilities
├── workflows/               # Workflow notes and default configs
├── examples/synthetic/      # Synthetic demo input
├── docs/                    # Method and usage docs
└── tests/                   # Unit tests
```

## Public commands

- `uv run python scripts/prepare_texts.py`: prepare a text-only parquet input.
- `./scripts/run_taxonomy_pipeline.sh`: run the multi-phase taxonomy workflow.
- `./scripts/review_taxonomy.sh`: create a manual review packet.
- `cd apps/reviewer && python3 -m http.server 5173`: start the zero-build local reviewer.

## Privacy and data handling

Do not commit source datasets, source identifiers, patient-like identifiers, local run logs, model caches, or reviewer decisions. Keep any linkage back to source data in your own controlled analysis environment, outside the public repository and outside model prompts unless explicitly approved for your project.

## Developer checks

```bash
uv run --extra dev pytest
uv run --extra dev ruff check .
```

If `uv` is unavailable but the local virtualenv exists:

```bash
.venv/bin/python -m pytest
.venv/bin/ruff check .
```

## Citation

If you use this software, cite the repository and the version or commit used for your analysis. Additional manuscript citation details can be added alongside the software citation when available.

## License

MIT License. See [LICENSE](LICENSE).
