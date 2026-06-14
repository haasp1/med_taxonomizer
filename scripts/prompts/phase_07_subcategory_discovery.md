# Phase 07: Subcategory Discovery

## Context

Domain context: {domain_context}
Target concept: {target}

You are discovering subcategories for a parent category.

Parent category: **{parent_code}**
Description: {parent_description}

Existing top-level category codes (do not reuse): {existing_codes}

## Constraints

- Max subcategories: **{max_children}** (including `other`)
- Return single-concept leaf slugs in `snake_case`
- Do not prefix with the meta or parent name; the pipeline will place each slug under the parent path
- Always include **other** as the catch-all leaf
- Do not invent new concepts beyond the samples
- Do not reuse `{parent_code}` or any existing top-level category codes

## Task

Choose a grouping criterion and list the subcategories.

## Samples

{samples}
