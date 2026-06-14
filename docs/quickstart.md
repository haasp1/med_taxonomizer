# Quickstart

This quickstart uses the synthetic CSV included in the repository.

## 1. Install

```bash
uv sync --extra dev
```

## 2. Prepare a text-only parquet file

```bash
uv run python scripts/prepare_texts.py \
  --input examples/synthetic/free_text.csv \
  --output data/inputs/texts.parquet \
  --text-column text
```

## 3. Configure a model provider

For hosted OpenRouter-compatible runs:

```bash
export OPENROUTER_API_KEY=...
```

For a local OpenAI-compatible server, pass `--base-url` to the runner.

## 4. Run the pipeline

```bash
./scripts/run_taxonomy_pipeline.sh \
  --input data/inputs/texts.parquet \
  --run-id quickstart_tree \
  --model qwen/qwen3.5-27b \
  --target clinical_events \
  --domain-context "Clinical free-text fragments for taxonomy induction." \
  --resume
```

Outputs are written under:

```text
scripts/data/runs/quickstart_tree/
```

## 5. Optional review

```bash
./scripts/review_taxonomy.sh \
  --taxonomy scripts/data/runs/quickstart_tree/intermediate/taxonomy_tree_final_qwen.json \
  --output-dir scripts/data/runs/quickstart_tree/review
```

Or start the local browser reviewer:

```bash
cd apps/reviewer
python3 -m http.server 5173
```

Open `http://127.0.0.1:5173/standalone.html`.
