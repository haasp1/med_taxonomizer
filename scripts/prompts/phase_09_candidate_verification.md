# Phase 09c: Decision Verification

## Context

Domain context: {domain_context}
Target concept: {target}
Scope type: {scope_type}
Scope metas: {scope_metas}

You are verifying one proposed pruning decision.

## Goal

Approve the proposal only if it removes a real labeling ambiguity and keeps the
taxonomy cleaner without collapsing clearly distinct concerns.

## Rules

- Use only the provided tree, candidate group, and proposed keep/remove decision
- Do not use outside knowledge
- Approve only if the same source mention could plausibly be labeled with the kept node or one of the removed nodes as an alternative choice
- Reject if the proposal mainly links nodes that are related but not interchangeable, such as different causes, mechanisms, manifestations, or distinct subtype concerns
- Reject if the proposal removes multiple distinct subtype labels in order to force them under a single mixed or broader label
- Reject if the proposal is based on loose relatedness rather than direct labeling competition
- Within one meta, approve a broad-vs-specific proposal when both nodes share the same exact site or location anchor and the kept node is the more specific label for that same anchor concern
- Reject any same-meta proposal that keeps the broader label and removes the more specific label for the same exact anchor concern
- If `scope_type` is `cross_meta`, approve only if the kept target-meta node is the cleaner canonical owner of the same concern
- In `cross_meta`, approve an owner-branch proposal when both nodes are family owners for the same anchor concern and the kept target-meta branch is intended to be the single canonical labeling surface
- If `scope_type` is `cross_meta`, reject a leaf-only proposal when a competing family-owner branch would still leave the labeling ambiguity unresolved
- If you are uncertain, reject

## Input Tree

{tree}

## Candidate Group

{candidate}

## Proposed Decision

{proposal}

## Output

Return:
- `approve`
