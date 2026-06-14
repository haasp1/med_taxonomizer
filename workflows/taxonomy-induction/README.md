# Taxonomy induction workflow

This workflow constructs candidate medical complication taxonomy nodes from free-text descriptions.

## Public entrypoint

Use this command from the repository root:

```bash
./scripts/create_taxonomy.sh --input path/to/data.csv --output-dir outputs
```

Then label records with:

```bash
./scripts/label_records.sh \
  --input path/to/data.csv \
  --taxonomy outputs/<taxonomy_run_id>/taxonomy_tree_draft.json \
  --output-dir outputs
```

For a working synthetic run:

```bash
./scripts/run_example.sh
```

## Purpose

The goal is to turn heterogeneous phrases into a structured taxonomy and then apply that taxonomy to records as a label table. The workflow emphasizes transparent labels, concise definitions, aggregate provenance, generated internal IDs, and reproducible exports.

## Inputs

Minimum input table:

```text
text
"short free-text description"
```

Supported text column names are documented in `docs/quickstart.md`. Synthetic examples are in `examples/synthetic/free_text.csv`.

Source identifiers are ignored by design. Do not send source IDs, patient IDs, case IDs, registry IDs, or hashed source IDs into model-facing workflows.

## Outputs

Taxonomy runs write:

- `taxonomy_tree_draft.json`
- `candidate_nodes.csv`
- `run_metadata.json`

Labeling runs write:

- `labels.csv`
- `label_run_metadata.json`

## Configuration

See `configs/default.yaml` for the minimal synthetic configuration. The public command accepts `--text-column`, `--taxonomy-version`, and `--run-id` options. `--id-column` is deprecated and ignored.
