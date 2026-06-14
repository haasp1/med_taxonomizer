# Phase 01: Meta-Category Consolidation

## Context

Domain context: {domain_context}
You are consolidating multiple meta-category discovery runs into a single, robust set.
Target: **{target}**

## Constraints

- Max meta-categories: **{max_metas}**
- Meta-categories must be simple, single-concept, snake_case
- Include **other** as a catch-all
- Exactly one meta-category should be marked **is_target = true**
- Do **not** invent new concepts; only merge/rename/split equivalent concepts from the runs 
- Meta categories must be mutually exclusive, single‑purpose, and defined by semantic role (not by overlapping context or source).
- For each non-target meta-category, explicitly assess the risk that it could absorb target items; if risk exists, tighten the definition or split/rename to prevent target leakage.
- Each meta description must make the boundary clear: state what belongs in the meta and what should stay out
- Prefer meta sets with clean semantic-role separation over broader but fuzzier coverage

## Input Runs

Each run is listed by ID. Use these IDs in `members`.

{runs}

## Output

Return:
- `reasoning`
- `clusters`: each with
  - `code`
  - `description`
  - `is_target`
  - `members` (list of entries like `seed_42:{target}`)
