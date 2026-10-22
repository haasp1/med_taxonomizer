# Med Taxonomizer

[![DOI](https://zenodo.org/badge/1227985820.svg)](https://doi.org/10.5281/zenodo.21275129)

**Med Taxonomizer** is a research toolkit for inducing a reusable medical taxonomy from free-text records, reviewing the taxonomy locally, and applying it in a structured labeling workflow.

Version **v1.1.0** adds the AACT demonstration, Qwen configuration examples and configurable input paths. See [CHANGELOG.md](CHANGELOG.md).

The public repository contains code, prompts, documentation, a synthetic example, and a local reviewer. It does **not** include project datasets, run logs, or internal notes. The only paper-run artifacts are those of the public AACT transfer demonstration in [`paper_runs/aact_transfer/`](paper_runs/aact_transfer/), which are derived from public ClinicalTrials.gov text.

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

For the exact Qwen 3.5 settings used in the study (per-phase sampling, concurrency, retries) and how to run against a local OpenAI-compatible server with `--base-url`, see [docs/qwen-configuration.md](docs/qwen-configuration.md).

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
├── paper_runs/              # AACT public transfer run wrappers and exports
├── configs/                 # Documented Qwen 3.5 settings (hosted and local)
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
- `paper_runs/aact_transfer/run_aact_*.sh`: re-run the AACT public transfer demonstration.
- `cd apps/reviewer && python3 -m http.server 5173`: start the zero-build local reviewer.

## Data handling

Only the text column is sent to a model; identifiers and linkage back to source records stay in your own environment. Hosted calls go through OpenRouter with the zero-data-retention routing flag (`extra_body={"provider": {"zdr": True}}`) on every request, and every phase accepts `--base-url` for a local OpenAI-compatible server instead. The reported taxonomy-discovery and AACT runs used OpenRouter with this flag. MedGemma and Qwen 3.5 27B for the 300-entry held-out comparison ran locally. Details: [docs/data-handling.md](docs/data-handling.md).

Do not commit source datasets, source identifiers, patient-like identifiers, local run logs, model caches, or reviewer decisions.

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
