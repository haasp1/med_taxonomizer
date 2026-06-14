# Phase 06: Split Audit

## Context

Domain context: {domain_context}
Target concept: {target}

You are reviewing samples for a single category to decide if it should be split into subcategories.

Category code: **{parent_code}**
Category description: {parent_description}

## Task

Decide whether this category is too broad and should be split into subcategories.

Return:
- `should_split`: true/false
- `reasoning`: short justification
- `suggested_grouping_criterion`: if should_split is true

## Rules

- Be conservative: split only if clear, repeated subtypes are present
- Do not invent new concepts beyond what the samples show
- If domain_context is "none", ignore it

## Samples

{samples}
