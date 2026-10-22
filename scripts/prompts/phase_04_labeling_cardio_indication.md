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

## Cardiology-indication labeling rule

This corpus comes from coronary artery disease / acute coronary syndrome / myocardial infarction / PCI / stenting / angioplasty / coronary revascularization trials.

For this labeling run, **conditions** must be interpreted narrowly:

- Use `conditions` only for baseline diseases, study indications, eligibility-defining diagnoses, or clearly pre-existing comorbidities that could characterize the enrolled cardiology/coronary-intervention study population.
- Do **not** label a term as `conditions` merely because it is a diagnosis or disease in general clinical ontology.
- Prefer `adverse_events` when a diagnosis-like term is plausibly incident, treatment-emergent, post-procedural, safety-relevant, or a complication in this coronary-intervention context.
- Acute non-indication diagnoses such as pneumonia, sepsis, infection, bleeding, stroke, renal failure, anemia, respiratory failure, or similar complications should generally be labeled as adverse events unless the text explicitly frames them as baseline eligibility/indication or an outcome measurement.
- Cardiologic terms that are plausible study indications or baseline cardiovascular disease states (e.g., coronary artery disease, acute coronary syndrome, myocardial infarction as enrollment indication, heart failure as comorbidity) may be `conditions` when the wording supports a baseline/study-population role. If the same term is presented as a safety endpoint, post-procedural complication, or adverse occurrence, label it as `adverse_events` or `outcomes` as appropriate.

## Samples

{samples}
