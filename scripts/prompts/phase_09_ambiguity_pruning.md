# Phase 09: Ambiguity Pruning

## Context

Domain context: {domain_context}
Target concept: {target}

You are auditing a taxonomy tree for labeling ambiguity.

## Goal

Find nodes anywhere in the tree that would create ambiguous labeling because
they represent the same or strongly overlapping concern.

For each ambiguous concern:
- keep one canonical node
- remove the competing node or nodes

If you remove an internal node, its full subtree will also be removed.
Prefer removing the highest-level competing node when the ambiguity is caused
by an entire branch rather than a single child.

## Decision Rules

- Use only the provided codes, descriptions, and tree context
- Do not use outside knowledge
- Optimize for a taxonomy that supports unambiguous labeling
- You may prune leaf nodes or internal nodes
- Be decisive: if two nodes would compete during labeling, resolve the conflict
- When the same concern appears under the target meta and also under another meta,
  prefer the target-meta node as canonical owner if both would compete during labeling
- Within the same meta, if one node is a broader bucket and another is a more specific overlapping node,
  prefer the more specific node as canonical owner
- Only prune when the overlap is direct and obvious from the names and descriptions themselves
- Do not prune nodes merely because they are causally related, commonly co-occur, or sit in neighboring parts of the taxonomy
- Do not prune nodes that differ mainly as mechanism vs outcome, subtype vs broader result, or residual bucket vs specific label
- Prefer the smallest valid pruning decision that removes the ambiguity
- Remove a whole branch only when the branch label itself is the competing concern, not just one of its descendants
- Preserve intentional residual or catch-all nodes such as `other` unless they directly duplicate another residual node at the same decision level
- Do not invent new nodes, rename nodes, or rewrite the tree
- Return only node ids that already exist in the input
- List only the highest competing nodes to remove; do not also list descendants of a removed node

## Input Tree

{tree}

## Output

Return:
- `decisions`: list of `{concern, keep_node_id, remove_node_ids}`
