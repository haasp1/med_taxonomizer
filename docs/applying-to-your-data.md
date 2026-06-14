# Applying Med Taxonomizer to your own free-text data

Med Taxonomizer is designed for research teams that have free-text fields and want to derive an analyzable label space. Typical inputs are complication notes, adverse-event comments, discharge remarks, registry free-text fields, or other short medical descriptions.

The standard workflow is:

1. prepare a text-only parquet input,
2. run the multi-phase taxonomy pipeline,
3. optionally review the taxonomy locally,
4. export taxonomy and label artifacts for downstream analysis.

Any linkage back to the original analysis table should stay in the research team's controlled local environment, not in model-facing prompts, traces, or repository outputs.

## 1. Prepare an input table

Start with a CSV or parquet table containing at least one free-text column.

Recommended minimal CSV:

```text
text
"short free-text description"
"another description"
```

Optional grouping columns such as timepoint, site, cohort, or reviewer split can remain in the source table for local analysis. The public preparation script reads the selected text column and writes a new parquet file with a single `text` column.

Do not include patient IDs, case IDs, registry IDs, source record IDs, or hashed source IDs in model-facing inputs. Fresh runtime IDs used by the pipeline are not hashes and are not derived from source IDs, because hashed or relinkable IDs can still be pseudonymised personal data under GDPR.

Keep project-specific sensitive material outside the repository. Use synthetic examples for tests, demos, and screenshots.

## 2. Create the text parquet input

```bash
uv run python scripts/prepare_texts.py \
  --input path/to/your_data.csv \
  --output data/inputs/texts.parquet \
  --text-column text
```

If the input text column is not named `text`, pass the correct column name with `--text-column`.

## 3. Run the taxonomy pipeline

```bash
./scripts/run_taxonomy_pipeline.sh \
  --input data/inputs/texts.parquet \
  --run-id my_run \
  --model qwen/qwen3.5-27b \
  --target clinical_events \
  --domain-context "Clinical free-text fragments for taxonomy induction." \
  --resume
```

Useful options:

- `--start-phase` and `--end-phase`: run a subset of phases.
- `--resume`: reuse completed artifacts where supported.
- `--concurrency`: override model-call concurrency.
- `--base-url`: use an OpenAI-compatible local model endpoint.
- `--max-phase-attempts`: retry failed phases before stopping.

Artifacts are written under:

```text
scripts/data/runs/<run-id>/
```

## 4. Optional local review

For spreadsheet-style review:

```bash
./scripts/review_taxonomy.sh \
  --taxonomy scripts/data/runs/<run-id>/intermediate/taxonomy_tree_final_qwen.json \
  --output-dir scripts/data/runs/<run-id>/review
```

For browser review:

```bash
cd apps/reviewer
python3 -m http.server 5173
```

Open `http://127.0.0.1:5173/standalone.html`.

## Data handling checklist

- Use synthetic data for public examples and screenshots.
- Keep source identifiers outside model-facing input files.
- Keep source linkage tables outside this repository.
- Inspect generated artifacts before sharing them.
- Do not commit `data/`, `scripts/data/`, `outputs/`, `logs/`, model caches, or reviewer decisions.
