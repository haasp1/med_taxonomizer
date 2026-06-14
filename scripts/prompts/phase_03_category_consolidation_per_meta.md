# Phase 03: Category Consolidation (Per-Meta)

## Context

You are consolidating multiple category discovery runs for a single meta-category.

Meta-category: **{meta_category}**
Description: {meta_description}

## Constraints

- Max categories: **{max_categories}** (including `other`)
- Category names must be snake_case and single-concept
- Include **other** as a catch-all
- Do **not** invent new concepts; only merge/rename equivalent concepts from the runs
- Be conservative: if unsure, keep categories separate

## Input Runs

Use these run IDs in `members`.

{runs}

## Output

Return:
- `meta_category`
- `grouping_criterion` (pick the most consistent one across runs)
- `reasoning`
- `clusters`: each with
  - `code`
  - `description`
  - `members` (list of entries like `seed_42:infection`)
