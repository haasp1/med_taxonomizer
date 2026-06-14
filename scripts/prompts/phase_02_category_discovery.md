# Phase 02: Category Discovery

## Context

Domain context: {domain_context}
Meta-category: **{meta_category}**
Description: {meta_description}

**Important:**
- This is a sample from the full dataset.
- Your job is to identify **broad** categories that will generalize
- We will refine after labeling

## Task

Choose a **grouping criterion** that fits this meta-category, then propose categories:

- Examples of criteria: organ system, event type, intervention type, endpoint type

First reason about what criterion makes sense for THIS meta-category, then create categories based on that criterion.

## Rules

1. **Stay HIGH LEVEL**: Broad categories > many specific ones.
2. **Always include "other"** as a catch-all.
3. **Simple names**: snake_case (e.g., `hemorrhage`, `infection`, `cardiac`)
4. Choose one primary grouping criterion and keep sibling categories parallel to it.
5. Respect the semantic role implied by the meta-category description; do not mix fundamentally different roles within the same sibling set.
6. Do not create categories that merely restate the meta-category itself or act as vague umbrella buckets beyond `other`.
7. Avoid mixed or catch-all categories such as broad "general", "mixed", or parent-restating labels when a cleaner parallel category set is possible.
8. If domain_context is "none", ignore it.

## Samples

{samples}
