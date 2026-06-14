# Phase 07: Subcategory Consolidation

## Context

Domain context: {domain_context}
Target concept: {target}

You are consolidating multiple subcategory discovery runs for a single parent category.

Parent category: **{parent_code}**
Description: {parent_description}

Existing top-level category codes (do not reuse): {existing_codes}

## Constraints

- Max subcategories: **{max_children}** (including `other`)
- Return single-concept leaf slugs in `snake_case`
- Do not prefix with the meta or parent name; the pipeline will place each slug under the parent path
- Do not invent new concepts; only merge/rename equivalent ones
- Always include **other** as the catch-all leaf
- Do not reuse `{parent_code}` or any existing top-level category codes

## Input Runs

Each run is listed by ID. Use these IDs in `members`.

{runs}

## Output

Return:
- `parent_code`
- `grouping_criterion`
- `reasoning`
- `clusters` with `code`, `description`, `members` (e.g., `seed_42:pneumonia`)
