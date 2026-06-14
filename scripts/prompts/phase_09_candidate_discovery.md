# Phase 09a: Candidate Discovery

## Context

Domain context: {domain_context}
Target concept: {target}
Scope type: {scope_type}
Scope metas: {scope_metas}

You are identifying possible taxonomy ambiguities.

## Goal

Return candidate groups of nodes that may create ambiguous labeling.

This is a **high-recall discovery step**:
- it is acceptable to include borderline candidates
- prefer finding too many candidates over missing plausible candidates
- do not decide final keep/remove actions yet

## Rules

- Use only the provided tree, codes, and descriptions
- Do not use outside knowledge
- Focus on direct or plausible label competition
- Prefer candidates tied to a shared anchor concern in the node code, path, or description
- Candidate groups may include leaf nodes or internal nodes
- When ambiguity looks like a branch-level ownership conflict, include the competing owner nodes as a candidate group
- Prefer owner-to-owner candidate groups over scattered child-node groupings when one branch clearly owns the concern
- In `cross_meta`, if a concern appears as a family owner in both metas, include the competing family owners as a candidate even if overlapping leaves also exist
- If `scope_type` is `cross_meta`, every candidate group must include at least one target-meta node and at least one non-target-meta node
- If `scope_type` is `cross_meta`, do not mix several same-meta target alternatives into one candidate; keep same-meta ambiguity for the `in_meta` list
- A candidate group should contain two or more node ids
- Return only existing node ids from the input tree
- Do not propose pruning decisions in this step

## Input Tree

{tree}

## Output

Return:
- `candidates`: list of `{concern, node_ids}`
