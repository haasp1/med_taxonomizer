# Phase 05: Category Healing Audit

## Context

Domain context: {domain_context}
Target concept: {target}

You are reviewing rows currently labeled **{meta}:other**.
This is a **constrained flat taxonomy patch** step, not meta discovery, not subcategory discovery, and not a full taxonomy rewrite.

Meta under review: **{meta}**
Meta description: {meta_description}
Current size of this meta:other bucket: {other_count}

Existing categories for this meta:
{existing_categories}

Other metas and their current categories:
{other_meta_context}

## Task

Review the sampled rows and decide whether this meta is missing any **flat sibling categories** at the same abstraction level as the existing categories.

Return:
- `summary`: short summary of the major patterns in the sampled other bucket
- `proposed_categories`: up to **{max_new_categories}** new flat categories for this same meta; return as few as possible and return an empty list if no strong patch is warranted
- `remaining_other_estimate`: estimated number of rows that should still remain `other`

## Rules

- Stay within the same meta; do not invent a new meta
- Do not propose subcategories of an existing category
- Prefer an existing category when it already fits
- Ignore true meta-level outliers (`other:other`) completely; they are not part of this task
- Be conservative: only propose categories that recur clearly across the sampled rows
- Category codes must be snake_case and single-concept
- Do not return `other` as a proposed category
- Treat the existing categories as the reference sibling set; new categories must match their abstraction level
- Do not propose umbrella categories that restate the meta itself or the target concept
- Do not propose categories that are too broad, too vague, or simple restatements of charting noise
- Do not propose categories that are too narrow, one-off, or effectively case-specific subtypes
- Add as few new categories as possible; default to zero unless the omission is clear and recurring
- Most runs should propose fewer than the maximum; do not fill the limit just because it is available
- Use the other-meta context to detect rows that belong elsewhere
- If a sampled pattern is primarily covered by another meta, do not create a new category here; count it toward `remaining_other_estimate`
- For non-target metas, do not carve out {target} categories; if a row primarily describes a {target}, it should remain `other` in this step
- If a sampled pattern would be better handled by moving or redefining an existing category, do not create a new category here; keep it in `other`
- Only propose a category when it would clearly absorb a recurring cluster of rows that are currently stranded in `other`

## Rows

{samples}
