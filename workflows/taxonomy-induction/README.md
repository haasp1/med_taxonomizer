# Taxonomy induction workflow

This workflow constructs candidate medical taxonomy nodes from free-text descriptions and applies the resulting taxonomy through the phase pipeline.

## Public entrypoint

Prepare a text-only parquet input:

```bash
uv run python scripts/prepare_texts.py \
  --input path/to/free_text.csv \
  --output data/inputs/texts.parquet \
  --text-column text
```

Run the pipeline from the repository root:

```bash
./scripts/run_taxonomy_pipeline.sh \
  --input data/inputs/texts.parquet \
  --run-id my_run \
  --model qwen/qwen3.5-27b \
  --target clinical_events \
  --domain-context "Clinical free-text fragments for taxonomy induction." \
  --resume
```

## Purpose

The goal is to turn heterogeneous phrases into a structured taxonomy and then apply that taxonomy to records as a label table. The workflow emphasizes transparent labels, concise definitions, aggregate provenance, generated runtime IDs, and reproducible exports.

## Inputs

Minimum input table before preparation:

```text
text
"short free-text description"
```

Supported text column names are documented in `docs/quickstart.md`. Synthetic examples are in `examples/synthetic/free_text.csv`.

Source identifiers are ignored by design. Do not send source IDs, patient IDs, case IDs, registry IDs, or hashed source IDs into model-facing workflows.

## Outputs

Pipeline artifacts are written under:

```text
scripts/data/runs/<run-id>/
```

Intermediate and final filenames depend on the phase and model provider. Use `docs/entrypoints.md` and the runner output for the active paths.

## Configuration

Use `--start-phase` and `--end-phase` to run a subset of phases. Use `--resume` to reuse completed artifacts where supported. Use `--base-url` for a local OpenAI-compatible model endpoint.
