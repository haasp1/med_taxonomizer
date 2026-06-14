# Phase 08: Final Tree Consolidation (Per Meta)

## Context

Domain context: {domain_context}
Target concept: {target}

You are consolidating an existing taxonomy tree for a single meta-category.

Meta-category: **{meta}**

## Constraints

- Return a flat list of canonical meta-relative paths
- Use slash-separated paths like `hemorrhage`, `hemorrhage/intracranial_hemorrhage`, `cardiovascular/other`
- You may rename nodes to cleaner canonical identifiers
- Preserve node count: every input row must still be represented exactly once in the final output
- No cross-meta moves (only within this meta)
- Max children per parent path: **{max_children}**
- Keep segments in `snake_case`

## Input Paths (current)

{tree}

## Output

Return:
- `reasoning`
- `paths`: list of `{path, description}`
