# Public entrypoints

Supported public commands:

## Prepare input text

```bash
uv run python scripts/prepare_texts.py \
  --input path/to/free_text.csv \
  --output data/inputs/texts.parquet \
  --text-column text
```

Creates a parquet file with one `text` column for the pipeline.

## Run taxonomy induction and labeling phases

```bash
./scripts/run_taxonomy_pipeline.sh \
  --input data/inputs/texts.parquet \
  --run-id my_run \
  --model qwen/qwen3.5-27b \
  --target clinical_events \
  --domain-context "Clinical free-text fragments for taxonomy induction." \
  --resume
```

The runner executes phases 00 through 09 and writes artifacts under `scripts/data/runs/<run-id>/`.

Useful options:

- `--start-phase` and `--end-phase`: run a subset of phases.
- `--resume`: reuse completed artifacts where supported.
- `--concurrency`: override model-call concurrency.
- `--base-url`: use an OpenAI-compatible local model endpoint.
- `--max-phase-attempts`: retry failed phases before stopping.

## Create a review packet

```bash
./scripts/review_taxonomy.sh \
  --taxonomy scripts/data/runs/<run-id>/intermediate/taxonomy_tree_final_qwen.json \
  --output-dir scripts/data/runs/<run-id>/review
```

## Start the local browser reviewer

```bash
cd apps/reviewer
python3 -m http.server 5173
```

Open `http://127.0.0.1:5173/standalone.html`.

## Data hygiene

Local datasets, run outputs, caches, notes, and logs are intentionally ignored by Git. Do not commit source data or reviewer decisions to the public repository.
