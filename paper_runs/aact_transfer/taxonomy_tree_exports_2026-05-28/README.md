# AACT taxonomy-tree exports (exported 2026-05-28)

Hierarchy artifacts from the full Qwen 3.5 27B taxonomy run on the 1,979 AACT
coronary-intervention text fragments (run id
`aact_qwen35_27b_20260510_taxonomizer_full1979`, produced by
`../run_aact_taxonomy_full1979_20260510.sh`). They are copies of the files that
the pipeline writes to `scripts/data/runs/<run-id>/intermediate/`, which is not
tracked by Git.

The pruned tree is the complete hierarchical AACT taxonomy (six meta domains,
36 subcategories, 74 sub-subcategories).

## Important distinction

The Phase 04 per-sample labels are flat `meta` + `category` assignments, e.g.
`adverse_events:cardiovascular`. The Phase 07/08/09 tree artifacts add deeper
paths such as `cardiovascular/ischemia_infarction`, but these subpaths were
**not assigned back to every sample**. Use the per-sample exports in
`../phase04_label_exports_2026-05-28/` for source-type alignment and
prompt-sensitivity analyses, and the tree files here to describe the induced
hierarchy. Do not imply that every AACT fragment has an adjudicated three-level
path.

## Files

- `taxonomy_tree_final_qwen.json`: final hierarchy (Phase 08 output).
- `taxonomy_tree_final_pruned_qwen.json`: final hierarchy after cross-domain
  ambiguity pruning (Phase 09 output). This is the reported AACT taxonomy.
- `taxonomy_tree_final_pruning_report_qwen.json`: Phase 09 pruning decisions.
- `taxonomy_tree_initial_qwen.json`: initial hierarchy from subcategory
  discovery (Phase 07 output), before tree consolidation and pruning.
- `taxonomy_tree_final_qwen_flattened.csv`: final hierarchy as a table, 129
  rows (44 category rows, 85 deeper subpath rows).
- `taxonomy_tree_final_pruned_qwen_flattened.csv`: pruned hierarchy as a table,
  110 rows (36 category rows, 74 deeper subpath rows).
- `taxonomy_tree_initial_qwen_flattened.csv`: header only; the initial artifact
  uses a different internal structure and is not flattened.

## Flattened CSV columns

- `meta`: top-level meta domain.
- `phase04_category`: corresponding flat Phase 04 category, e.g.
  `adverse_events:cardiovascular`.
- `tree_subpath`: hierarchy path below the meta, e.g.
  `cardiovascular/ischemia_infarction`.
- `tree_depth_after_meta`: 1 for the Phase 04 category level, 2 for one deeper
  level.
- `tree_leaf`: final path component.
- `description`: model-generated category or path description.
