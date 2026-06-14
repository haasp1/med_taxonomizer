# Phase 05: Targeted Other Relabeling

## Context

Domain context: {domain_context}
Target concept: {target}

You are relabeling rows that are currently assigned to **{meta}:other**.
This is a targeted validation step for a proposed flat taxonomy patch inside a single meta.

Meta under review: **{meta}**
Meta description: {meta_description}

Allowed categories for this meta:
{allowed_categories}

Other metas and their current categories:
{other_meta_context}

## Task

For each row, choose exactly one category from the allowed list for the **target row only**.
If none of the allowed categories clearly fits, keep the row as `other`.

Return:
- `sample_uuid`: the raw row identifier from the input, without brackets or added text
- `category`: one allowed category or `other`
- `reasoning`: short justification
- `suggested_category`: optional suggestion only if the row stays `other`

## Rules

- Stay strictly within this meta
- Do not invent a new category
- Prefer an existing non-`other` category when it clearly fits
- Use a tentative new category only if the row clearly matches it
- If the row is still too ambiguous or out-of-scope for this meta, keep `other`
- Ignore true meta-level outliers (`other:other`) completely; they are not part of this task
- The sample may already have other labels; those are fixed context only and must not be revised
- Reclassify only the current **{meta}:other** row; do not reinterpret the whole sample or add extra labels
- Use the other-meta context to detect rows that primarily belong somewhere else
- Do not force a row into this meta if the text primarily reflects another meta or the target concept
- For non-target metas, if the row mainly describes a {target}, keep `other` here rather than creating ambiguity
- Prefer keeping `other` over making a weak or overly broad assignment
- Use the candidate categories as sibling flat categories, not as subtypes or umbrella buckets

## Rows

{samples}
