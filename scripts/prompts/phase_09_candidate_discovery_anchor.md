# Phase 09a2: Candidate Discovery By Shared Anchor

## Context

Domain context: {domain_context}
Target concept: {target}
Scope type: {scope_type}
Scope metas: {scope_metas}

You are identifying possible taxonomy ambiguities with very high recall.

## Goal

Return candidate groups of nodes that may compete during labeling because they
share the same anchor concern.

This is still a discovery step:
- over-call candidates rather than miss them
- do not make pruning decisions yet

## Rules

- Use only the provided tree, codes, paths, and descriptions
- Do not use outside knowledge
- Look for repeated anchor concerns across branches, such as the same site, event, finding, status, or named subtype appearing in more than one place
- A strong discovery signal is a shared anchor in the code, path, or description
- Within one meta, if two nodes share the same site or location anchor and one is a broader bucket while the other is a narrower event at that same anchor, include that pair
- If the same concern appears as both a branch owner and a leaf elsewhere, include that owner-level conflict as a candidate
- In `cross_meta`, if the same concern is owned by family-level nodes in both metas, include the competing family owners as a candidate before considering leaf-level variants
- If `scope_type` is `cross_meta`, every candidate group must include at least one target-meta node and at least one non-target-meta node
- If `scope_type` is `cross_meta`, do not depend on choosing among multiple same-meta target alternatives in the same candidate
- Candidate groups may include leaf nodes or internal nodes
- A candidate group should contain two or more node ids
- Return only existing node ids from the input tree
- Do not propose keep/remove actions

## Input Tree

{tree}

## Output

Return:
- `candidates`: list of `{concern, node_ids}`
