# Applying Med Taxonomizer to your own free-text data

Med Taxonomizer is designed for research teams that already have free-text fields but do not yet have an analyzable label space. Typical inputs are complication notes, adverse-event comments, discharge remarks, registry free-text fields, or other short medical descriptions.

The standard workflow is:

1. create a taxonomy from free text,
2. label the records with that taxonomy,
3. export a tidy label table for downstream analysis.

Any linkage back to the original analysis table should stay in the research team's controlled local environment, not in model-facing prompts, traces, or repository outputs.

## 1. Prepare an input table

Start with a table containing at least one free-text column.

Recommended format for a first run:

```text
text,group
"short free-text description",development
"another description",development
```

Optional grouping columns such as timepoint, site, cohort, or reviewer split can remain in the source table for your own local analysis. The public taxonomy and labeling scripts currently read only the selected text column.

Do not include patient IDs, case IDs, registry IDs, source record IDs, or hashed source IDs in model-facing inputs. The CLI ignores source identifier columns and creates fresh internal IDs such as `medtax-000001`. These are not hashes and are not derived from source IDs, because hashed/relinkable IDs can still be pseudonymised personal data under GDPR.

Keep project-specific sensitive material outside the repository. Use synthetic examples for tests, demos, and screenshots.

## 2. Create the taxonomy

```bash
./scripts/create_taxonomy.sh \
  --input path/to/your_data.csv \
  --output-dir outputs \
  --taxonomy-version draft-YYYYMMDD
```

If the input text column is not named `text`, pass it explicitly:

```bash
./scripts/create_taxonomy.sh \
  --input path/to/your_data.csv \
  --text-column complication_note \
  --output-dir outputs
```

Current public baseline:

1. load and validate the input table,
2. ignore source identifier columns,
3. create fresh internal runtime IDs,
4. normalize and filter the text,
5. flag high-risk privacy markers,
6. build a deterministic draft taxonomy,
7. export a draft taxonomy tree,
8. export a candidate-node table,
9. export run metadata and warnings.

Expected outputs:

- `taxonomy_tree_draft.json`,
- `candidate_nodes.csv`,
- `run_metadata.json`.

## 3. Label the records

Use the taxonomy path printed by the previous command:

```bash
./scripts/label_records.sh \
  --input path/to/your_data.csv \
  --taxonomy outputs/<taxonomy_run_id>/taxonomy_tree_draft.json \
  --output-dir outputs
```

Expected outputs:

- `labels.csv`,
- `label_run_metadata.json`.

A typical model-facing output table contains:

```text
internal_record_id,node_id,label_name,taxonomy_version,labeler,run_id
medtax-000001,node-0001,Leaf label,taxonomy_YYYYMMDD,model-or-baseline,run-001
```

If the research team needs to join labels back to the original analysis table, do that only in the controlled local analysis environment using a separately governed linkage process. Do not commit that linkage table or send it to LLM providers.

## 4. Optional annotation/benchmarking work

The annotation UI and manual review materials are paper-support tooling. They are useful if you explicitly want to create a reviewed sample or benchmark labeling quality, but they are not required for the default taxonomy + full-corpus labeling workflow.

If used, freeze:

- taxonomy version,
- prompt file,
- model identifier,
- output schema,
- sampling and run settings.

## 5. Analyze downstream

Common downstream analyses include:

- concept prevalence,
- subgroup comparisons,
- timepoint profiles,
- inter-site variation,
- regression or outcome models,
- manual audit lists for uncertain categories.

## Minimal checklist

- [ ] Model-facing inputs contain free text only, not source IDs.
- [ ] Source IDs are not hashed and sent as “anonymous” IDs.
- [ ] Generated internal IDs are used in taxonomy/label outputs.
- [ ] Text normalization and filtering are documented.
- [ ] Taxonomy tree is versioned.
- [ ] Full-corpus labels include taxonomy version and model/run metadata.
- [ ] Optional reviewed/annotated sample is saved only when benchmarking is needed.
- [ ] No project-specific sensitive records or linkage tables are committed to the repository.
