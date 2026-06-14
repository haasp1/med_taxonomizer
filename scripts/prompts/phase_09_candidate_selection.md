# Phase 09b: Candidate Selection

## Context

Domain context: {domain_context}
Target concept: {target}
Scope type: {scope_type}
Scope metas: {scope_metas}

You are reviewing one candidate ambiguity group and deciding whether it should
actually be used for pruning.

## Goal

Select only this candidate if it represents real, practical labeling ambiguity.

If selected:
- keep one canonical node
- remove the competing node or nodes

If not selected:
- return `should_prune=false`
- do not return a pruning decision

## Rules

- Use only the provided tree, candidate list, codes, and descriptions
- Do not use outside knowledge
- Prefer precision over recall in this step
- Only accept candidates where the ambiguity is direct and important for labeling
- Only accept a candidate when the same source mention could plausibly be labeled with either node as an alternative choice
- Strong evidence is a shared anchor concern in the node code, path, or description
- Reject candidates that are merely causally related, adjacent, residual, or only loosely associated
- Weak evidence is when one description only mentions the other as an example, cause, consequence, or related pathway
- Reject candidates where one node is mainly a cause, mechanism, site, precursor, subtype, or manifestation of the other rather than an interchangeable label
- Preserve intentional residual buckets unless they directly duplicate another residual bucket at the same decision level
- Reject any proposal that removes an explicit residual or catch-all child node only because it belongs under the kept family owner node
- Reject candidates that would replace multiple distinct subtype labels with one broader or mixed bucket
- Reject candidates made only of generic catch-all labels such as `other`
- If `scope_type` is `cross_meta`, accept only target-vs-other-meta competition; reject candidates that also depend on choosing among multiple same-meta target alternatives
- When the same concern appears under the target meta and another meta, prefer the target-meta node
- Within one meta, if a broad bucket competes with a more specific overlapping node, prefer the more specific node
- Within one meta, if two nodes share the same site or location anchor and one is a broad bucket while the other is a specific event at that anchor, prefer the specific node
- Reject any same-meta proposal that keeps the broader node and removes the more specific node for the same exact anchor concern
- For cross-meta family conflicts, prefer pruning the competing family owner node instead of deleting a scattered set of narrower nodes
- In `cross_meta`, reject a leaf-only cleanup when a competing family-owner candidate would better remove the ambiguity at the labeling surface
- If several sibling leaves only compete collectively with one node, select that candidate only when the family owner nodes themselves are the competing labels; otherwise reject it
- Remove a whole branch only when the branch label itself is the competing concern
- `keep_node_id` and every `remove_node_id` must come from the candidate group

## Input Tree

{tree}

## Candidate Group

{candidate}

## Output

Return:
- `should_prune`
- `keep_node_id`
- `remove_node_ids`
