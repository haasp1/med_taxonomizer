"""
Helpers for canonical taxonomy path identifiers used in Phase 07/08.
"""

from __future__ import annotations

import re
from typing import Any


def normalize_slug(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", (value or "").strip().lower())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


def split_code_parts(code: str) -> list[str]:
    parts = re.split(r"[:/]", (code or "").strip())
    return [slug for part in parts if (slug := normalize_slug(part))]


def canonical_root_code(raw_code: str) -> str:
    parts = split_code_parts(raw_code)
    if not parts:
        raise ValueError(f"Invalid root code: {raw_code!r}")
    return parts[-1]


def canonical_child_code(parent_code: str, raw_code: str) -> str:
    raw = (raw_code or "").strip()
    if not raw:
        raise ValueError("Child code cannot be empty")

    normalized_raw = normalize_slug(raw.replace(":", "_").replace("/", "_"))
    normalized_parent = normalize_slug(parent_code.replace("/", "_"))
    parent_leaf = split_code_parts(parent_code)[-1]
    parts = split_code_parts(raw)
    leaf = parts[-1] if parts else ""

    if (
        raw.lower() == "other"
        or leaf == "other"
        or normalized_raw == f"other_{normalized_parent}"
        or (normalized_raw.startswith("other_") and leaf == parent_leaf)
    ):
        leaf = "other"

    if not leaf:
        raise ValueError(f"Invalid child code: {raw_code!r}")
    return f"{parent_code}/{leaf}"


def canonicalize_meta_tree(meta_entry: dict[str, Any]) -> dict[str, Any]:
    meta = str(meta_entry.get("meta") or "")
    raw_roots = [r for r in (meta_entry.get("roots") or []) if isinstance(r, str)]
    raw_nodes = [n for n in (meta_entry.get("nodes") or []) if isinstance(n, dict)]

    node_map: dict[str, dict[str, Any]] = {}
    for node in raw_nodes:
        code = node.get("code")
        if not isinstance(code, str) or not code.strip():
            continue
        node_map[code] = {
            "description": str(node.get("description") or ""),
            "children": [c for c in (node.get("children") or []) if isinstance(c, str)],
        }

    canonical_nodes: dict[str, dict[str, Any]] = {}
    raw_to_canonical: dict[str, str] = {}

    def walk(raw_code: str, canonical_code: str, visiting: set[str]) -> None:
        if raw_code in visiting:
            raise ValueError(f"Cycle detected while canonicalizing meta {meta}: {raw_code}")
        if raw_code in raw_to_canonical:
            previous = raw_to_canonical[raw_code]
            if previous != canonical_code:
                raise ValueError(
                    f"Raw node {raw_code} maps to multiple canonical codes in meta {meta}: "
                    f"{previous} vs {canonical_code}"
                )
            return

        node = node_map.get(raw_code)
        if node is None:
            raise ValueError(f"Missing node referenced in meta {meta}: {raw_code}")

        visiting.add(raw_code)
        child_codes: list[str] = []
        for child_raw_code in node["children"]:
            child_canonical_code = canonical_child_code(canonical_code, child_raw_code)
            child_codes.append(child_canonical_code)
            walk(child_raw_code, child_canonical_code, visiting)
        visiting.remove(raw_code)

        raw_to_canonical[raw_code] = canonical_code
        canonical_nodes[canonical_code] = {
            "code": canonical_code,
            "description": node["description"],
            "children": child_codes,
        }

    canonical_roots: list[str] = []
    for raw_root in raw_roots:
        canonical_root = canonical_root_code(raw_root)
        canonical_roots.append(canonical_root)
        walk(raw_root, canonical_root, set())

    payload: dict[str, Any] = {
        "meta": meta,
        "roots": canonical_roots,
        "nodes": list(canonical_nodes.values()),
    }
    if "reasoning" in meta_entry:
        payload["reasoning"] = meta_entry["reasoning"]
    return payload


def canonicalize_taxonomy_tree(tree: dict[str, Any]) -> dict[str, Any]:
    metas = [
        canonicalize_meta_tree(meta_entry)
        for meta_entry in (tree.get("metas") or [])
        if isinstance(meta_entry, dict)
    ]
    return {
        **tree,
        "metas": metas,
    }
