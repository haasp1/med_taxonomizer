# Phase 05: Category Healing Consolidation

## Context

Domain context: {domain_context}
Target concept: {target}

You are consolidating multi-seed proposals for missing flat categories inside one meta.

Meta under review: **{meta}**
Meta description: {meta_description}
Current size of this meta:other bucket: {other_count}

Existing categories for this meta:
{existing_categories}

Other metas and their current categories:
{other_meta_context}

## Task

Merge the proposal runs into a single conservative set of canonical patch categories.
Return at most **{max_new_categories}** canonical categories.
Returning zero categories is valid and preferred when the evidence is weak or cross-meta.

## Rules

- Stay within the same meta; do not invent a new meta
- Do not propose subcategories of an existing category
- Drop noisy, one-off, or redundant suggestions
- Prefer a single robust canonical code when several proposals mean the same thing
- Category codes must be snake_case and single-concept
- Do not return `other` and do not reuse an existing category code
- Use the existing categories as the sibling reference set; any returned category must live at the same abstraction level
- Do not return umbrella categories that restate the meta or the target concept
- Do not return categories that are overly broad, overly specific, or operationally ambiguous
- Prefer fewer, stronger categories over many weak carve-outs
- Add as few categories as possible; do not preserve a suggestion unless it is clearly same-meta and robust across runs
- Use the other-meta context to drop proposals that are better explained by another meta
- For non-target metas, do not return categories whose meaning is primarily {target} content
- If a suggestion would be better resolved by redefinition, merge, or cross-meta movement of existing categories, drop it here instead of patching around the problem

## Proposal Runs

{runs}
