# Method overview

Med Taxonomizer supports a practical workflow for turning medical free-text descriptions into a taxonomy and a label table.

## 1. Taxonomy creation

The taxonomy creation workflow turns a set of free-text descriptions into candidate taxonomy nodes. The workflow is designed to preserve interpretability: each proposed node has a label, definition, parent relationship, and aggregate provenance count.

Typical steps:

1. Load a CSV free-text column.
2. Ignore source identifier columns.
3. Create generated internal runtime IDs.
4. Normalize and filter text.
5. Generate candidate labels.
6. Group and consolidate overlapping candidates.
7. Export a versioned taxonomy.

## 2. Record labeling

The labeling workflow applies a taxonomy to the full free-text corpus and exports a tidy label table.

Typical steps:

1. Read the selected taxonomy version.
2. Process records in batches using generated internal IDs.
3. Validate labels against the taxonomy/schema.
4. Export labels with taxonomy version, model/labeler, and run metadata.

## Optional annotation and benchmarking

Annotation/review tooling is optional paper-support infrastructure. It can be used to create a reviewed benchmark sample or compare model configurations, but it is not required for the standard taxonomy creation and full-corpus labeling workflow.

## Outputs

The intended standard outputs are:

- versioned taxonomy files,
- candidate-node summaries,
- full-corpus label tables,
- run metadata with the ID policy and warnings.

Source IDs, patient IDs, case IDs, registry IDs, and hashed source IDs should not appear in model-facing outputs.
