# Architecture

Med Taxonomizer is organized as a research toolkit with three layers: a method layer, a software layer, and a presentation layer.

## Method layer

The method layer describes the reproducible process used to move from free-text descriptions to a taxonomy and label table:

1. **Taxonomy creation** — normalize input text, generate candidate terms, consolidate near-duplicates, and produce taxonomy nodes.
2. **Record labeling** — apply the taxonomy to records and export a tidy label table for downstream analysis.
3. **Optional taxonomy review** — create reviewed taxonomy decisions or benchmark samples when needed for paper-specific validation.

## Software layer

- `lib/` provides shared Python helpers for taxonomy paths, model naming, data loading, and LLM requests.
- `scripts/` provides the standard public entrypoints: create taxonomy, label records, and run the synthetic example.
- `apps/reviewer/` contains an optional local browser reviewer for taxonomy trees; it is not needed for the standard command-line workflow.

## Presentation layer

- `website/` is the GitHub Pages landing page for paper readers; users do not need to run it locally for the standard workflow.
- `docs/` contains method notes and implementation details.
- `examples/synthetic/` provides minimal synthetic examples for demos and tests.

## Design principles

- README first: users should be able to run the standard workflow without jumping through several docs.
- Public-facing names should be self-contained; avoid internal manuscript numbering.
- Source identifiers are ignored; generated internal IDs are not hashes or pseudonyms of source IDs.
- Examples should be synthetic and understandable without access to the study context.
- Outputs should be versioned, inspectable, and suitable for citation after release.
