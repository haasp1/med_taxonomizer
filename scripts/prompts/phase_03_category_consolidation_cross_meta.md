# Phase 03: Category Consolidation (Cross-Meta)
# Phase 03: Category Consolidation (Cross-Meta)

## Context

Domain context: {domain_context}
Target concept: {target}

Meta-categories are fixed to this list:
{meta_categories}

Input categories are already consolidated within each meta-category. Your task is cross-meta consolidation into a final flat taxonomy.

## Constraints

- You may rename, merge, split, or redefine categories to improve separation and clarity
- You may introduce new category codes when splitting/renaming, but each new code must map to existing input concepts (no new concepts)
- Each final category must belong to exactly one meta-category
- Each output category must list input members it derives from; for merges, union members; for splits, partition members; do not drop members
- Category names must be snake_case and single-concept
- Always include **other** in each meta-category
- Max categories per meta: **{max_categories}**
- For each non-target meta, explicitly check every category for potential target overlap; if overlap exists, either move it to the target meta or tighten its definition to exclude target, but only move when the evidence is strong
- After forming the metas, perform a cross-meta audit and reassign any category that fits another meta better
- When uncertain, prefer preserving clean boundaries and tightening definitions over collapsing ambiguous content into the target meta
- The target meta must not become a fallback home for ambiguous, mixed, or weakly matched categories
- Prefer mutually exclusive semantic roles over broader but blurrier consolidation

## Input Categories

Each category is listed by meta and code. Use these IDs in `members`.

{categories}

## Output

Return:
- `reasoning`
- `metas`: list of
  - `meta_code`
  - `categories`: list of clusters with
    - `code`
    - `description`
    - `members` (entries like `meta_code:category_code`)

In `reasoning`, include a brief \"Changes\" note covering renames/merges/splits, cross-meta moves, and any leakage-prevention decisions.
