"""
Shared schema and normalization helpers for taxonomy labeling.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, create_model


def create_response_model(metas: list[str], codes: list[str]) -> type[BaseModel]:
    meta_literal = Literal[tuple(metas)]
    code_literal = Literal[tuple(codes)]

    label = create_model(
        "Label",
        meta=(meta_literal, Field(description="Meta-category")),
        code=(code_literal, Field(description="Category code")),
        reasoning=(str, Field(description="Brief explanation (1 sentence)")),
    )

    sample_result = create_model(
        "SampleResult",
        sample_id=(str, Field(description="Sample ID from input")),
        labels=(list[label], Field(description="Labels ordered by importance")),
    )

    return create_model(
        "BatchResult",
        samples=(list[sample_result], Field(description="Results for all samples in batch")),
    )


def normalize_sample_result(
    sample_result: Any,
    *,
    expected_id: str,
    taxonomy_index: Any,
) -> tuple[list[dict], list[str]]:
    errors: list[str] = []

    if sample_result is None:
        return [], ["Missing result"]

    if getattr(sample_result, "sample_id", None) != expected_id:
        errors.append("sample_id mismatch")

    labels = getattr(sample_result, "labels", []) or []

    normalized: list[dict] = []
    seen: set[tuple[str, str]] = set()

    for lbl in labels:
        meta = getattr(lbl, "meta", None)
        code = getattr(lbl, "code", None)
        if not isinstance(meta, str) or not isinstance(code, str):
            errors.append("Invalid label entry")
            continue
        if meta not in taxonomy_index.nodes_by_meta:
            errors.append(f"Unknown meta: {meta}")
            continue
        if code not in taxonomy_index.nodes_by_meta[meta]:
            errors.append(f"Invalid code for meta {meta}: {code}")
            continue
        if code not in taxonomy_index.leaf_codes_by_meta.get(meta, set()):
            errors.append(f"Non-leaf code not allowed for meta {meta}: {code}")
            continue
        key = (meta, code)
        if key in seen:
            continue
        seen.add(key)
        normalized.append(
            {
                "meta": meta,
                "code": code,
                "reasoning": getattr(lbl, "reasoning", ""),
            }
        )

    return normalized, errors
