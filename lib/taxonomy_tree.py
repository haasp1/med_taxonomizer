"""
Utilities for loading and rendering taxonomy trees (final taxonomy tree).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TaxonomyNode:
    code: str
    description: str
    children: list[str]


@dataclass(frozen=True)
class TaxonomyTreeIndex:
    domain_context: str
    target: str
    metas: list[str]
    roots_by_meta: dict[str, list[str]]
    nodes_by_meta: dict[str, dict[str, TaxonomyNode]]
    code_to_metas: dict[str, set[str]]


def load_taxonomy_tree(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Taxonomy tree not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if "metas" not in data:
        raise ValueError("Invalid taxonomy tree: missing 'metas'")
    return data


def build_taxonomy_index(tree: dict[str, Any]) -> TaxonomyTreeIndex:
    metas: list[str] = []
    roots_by_meta: dict[str, list[str]] = {}
    nodes_by_meta: dict[str, dict[str, TaxonomyNode]] = {}
    code_to_metas: dict[str, set[str]] = {}

    for meta_entry in tree.get("metas", []) or []:
        meta = meta_entry.get("meta")
        if not isinstance(meta, str):
            continue
        metas.append(meta)
        roots = meta_entry.get("roots", []) or []
        roots = [r for r in roots if isinstance(r, str)]
        roots_by_meta[meta] = roots

        node_map: dict[str, TaxonomyNode] = {}
        for node in meta_entry.get("nodes", []) or []:
            code = node.get("code")
            description = node.get("description") or ""
            children = node.get("children", []) or []
            if not isinstance(code, str):
                continue
            node_obj = TaxonomyNode(
                code=code,
                description=str(description),
                children=[c for c in children if isinstance(c, str)],
            )
            node_map[code] = node_obj
            code_to_metas.setdefault(code, set()).add(meta)

        nodes_by_meta[meta] = node_map

    domain_context = str(tree.get("domain_context", "") or "")
    target = str(tree.get("target", "") or "")

    return TaxonomyTreeIndex(
        domain_context=domain_context,
        target=target,
        metas=metas,
        roots_by_meta=roots_by_meta,
        nodes_by_meta=nodes_by_meta,
        code_to_metas=code_to_metas,
    )


def render_taxonomy_markdown(
    index: TaxonomyTreeIndex,
    *,
    detail: str = "full",
    max_depth: int | None = None,
) -> str:
    if detail not in {"full", "codes"}:
        raise ValueError("detail must be 'full' or 'codes'")

    lines: list[str] = []

    def render_node(meta: str, code: str, depth: int, visited: set[str]) -> None:
        if code in visited:
            return
        visited.add(code)
        node = index.nodes_by_meta.get(meta, {}).get(code)
        if node is None:
            return
        indent = "  " * depth
        if detail == "codes":
            lines.append(f"{indent}- {node.code}")
        else:
            lines.append(f"{indent}- {node.code}: {node.description}")
        if max_depth is not None and depth >= max_depth:
            return
        for child in node.children:
            render_node(meta, child, depth + 1, visited)

    for meta in index.metas:
        lines.append(f"### {meta}")
        roots = index.roots_by_meta.get(meta, [])
        if not roots:
            lines.append("- (no roots)")
            lines.append("")
            continue
        visited: set[str] = set()
        for root in roots:
            render_node(meta, root, 0, visited)
        lines.append("")

    return "\n".join(lines).strip()


def taxonomy_rows(index: TaxonomyTreeIndex) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for meta in index.metas:
        for code, node in index.nodes_by_meta.get(meta, {}).items():
            rows.append(
                {
                    "meta": meta,
                    "code": node.code,
                    "description": node.description,
                    "children": ", ".join(node.children),
                }
            )
    return rows
