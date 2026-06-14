# Expert review workflow

This workflow is optional paper-support tooling. It is not required for the default taxonomy creation and full-corpus labeling workflow.

## Optional entrypoint

Use this only if a project explicitly wants a manual taxonomy curation packet:

```bash
./scripts/review_taxonomy.sh \
  --taxonomy outputs/<run_id>/taxonomy_tree_final.json \
  --output-dir outputs/<run_id>/review
```

For a local browser interface:

```bash
cd apps/reviewer
python3 -m http.server 5173
```

Open `http://127.0.0.1:5173/standalone.html`.

## Purpose

The optional review packet records decisions such as approval, renaming, merging, splitting, and hierarchy revision. For standard users, taxonomy induction and full-corpus labeling can be run without this review step.

## Outputs

The review packet contains:

- `review_queue.csv`: one row per draft node with decision columns.
- `taxonomy_tree_curated_template.json`: JSON template to update after decisions are reconciled.
- `review_instructions.md`: short reviewer instructions.

## Decision names

Use plain decision names in review tables:

- `approve`
- `rename`
- `merge`
- `split`
- `reject`

Avoid internal phase names in reviewer-facing files.
