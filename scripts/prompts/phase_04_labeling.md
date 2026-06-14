# Phase 04: Sample Labeling

## Context

Domain context: {domain_context}
Language note (optional): {language_note}

## Taxonomy

### Meta-Categories
{meta_categories_list}

### Categories per Meta-Category
{categories_list}

## Task

For each sample, assign one or more labels. Return results using the exact UUID provided for each sample.

Each result consists of:
- **sample_uuid**: The UUID from the input (e.g., "a1b2c3d4-...")
- **labels**: List of labels, each with:
  - **meta**: The meta-category from the list above
  - **category**: A category within that meta-category
  - **reasoning**: Brief explanation (1 sentence, English)

## Guidelines

1. **CRITICAL: Label ALL samples** - Every UUID in the input MUST appear in your output
2. A sample may have MULTIPLE labels if it describes multiple distinct events
3. Order labels by clinical significance (most important first)
4. Use "other" category only when no existing category fits
5. If using "other", suggest a new category name in `suggested_category`
6. Match the meta-category to the category
7. If domain_context or language_note is "none", ignore it

## Samples

{samples}
