#!/usr/bin/env python3
"""Create a human-review packet from a taxonomy JSON file.

This helper is intentionally generic: it reads a taxonomy tree JSON, extracts
node-like objects, and writes a CSV queue that can be edited in a spreadsheet or
loaded into the local reviewer app. No project data is bundled with the script.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

NODE_KEYS = {"node_id", "id", "stable_id", "code", "path", "label", "name", "title", "category", "definition", "description", "children"}
LABEL_KEYS = ("label", "name", "title", "category", "code")
ID_KEYS = ("node_id", "id", "stable_id", "code")
PATH_KEYS = ("path", "taxonomy_path", "full_path", "code")
DEFINITION_KEYS = ("definition", "description", "rationale")
PARENT_KEYS = ("parent", "parent_id", "parent_path", "meta")
SUPPORT_KEYS = ("support", "provenance_count", "count", "n")
CHILD_KEYS = ("nodes", "children", "categories", "subcategories", "leaves", "taxonomy")


def stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=False)


def pick(record: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            return stringify(value)
    return ""


def collect_nodes(value: Any, nodes: list[dict[str, Any]]) -> None:
    if isinstance(value, list):
        for item in value:
            collect_nodes(item, nodes)
        return
    if not isinstance(value, dict):
        return

    has_label = any(key in value for key in LABEL_KEYS)
    has_taxonomy_shape = any(key in value for key in NODE_KEYS)
    if has_label and has_taxonomy_shape:
        nodes.append(value)

    for key in CHILD_KEYS:
        if key in value:
            collect_nodes(value[key], nodes)


def create_review_packet(taxonomy_path: Path, output_dir: Path) -> None:
    data = json.loads(taxonomy_path.read_text(encoding="utf-8"))
    raw_nodes: list[dict[str, Any]] = []
    collect_nodes(data, raw_nodes)
    if not raw_nodes:
        raise ValueError("No taxonomy-like nodes found in JSON")

    output_dir.mkdir(parents=True, exist_ok=True)
    queue_path = output_dir / "review_queue.csv"
    template_path = output_dir / "taxonomy_tree_curated_template.json"
    instructions_path = output_dir / "review_instructions.md"

    fieldnames = [
        "node_id",
        "current_label",
        "current_definition",
        "path",
        "parent",
        "support",
        "decision",
        "new_label",
        "new_definition",
        "merge_into_node_id",
        "split_notes",
        "reviewer_notes",
    ]
    with queue_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, node in enumerate(raw_nodes, start=1):
            label = pick(node, LABEL_KEYS)
            path = pick(node, PATH_KEYS)
            writer.writerow(
                {
                    "node_id": pick(node, ID_KEYS) or f"node-{index:03d}",
                    "current_label": label or path or f"Node {index}",
                    "current_definition": pick(node, DEFINITION_KEYS),
                    "path": path,
                    "parent": pick(node, PARENT_KEYS),
                    "support": pick(node, SUPPORT_KEYS),
                    "decision": "",
                    "new_label": "",
                    "new_definition": "",
                    "merge_into_node_id": "",
                    "split_notes": "",
                    "reviewer_notes": "",
                }
            )

    template_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    instructions_path.write_text(
        "# Taxonomy review instructions\n\n"
        "1. Open `review_queue.csv` in a spreadsheet or the local reviewer app.\n"
        "2. For each row, choose one decision: `approve`, `rename`, `merge`, `split`, or `reject`.\n"
        "3. Fill `new_label` or `new_definition` when a node needs clearer wording.\n"
        "4. Fill `merge_into_node_id` when two nodes should be merged.\n"
        "5. Use `split_notes` when a node is too broad and needs children.\n"
        "6. Keep reviewer notes concise and de-identified.\n\n"
        "The JSON template is a copy of the input taxonomy. Update it only after review decisions are reconciled.\n",
        encoding="utf-8",
    )

    print(f"Created review queue: {queue_path}")
    print(f"Created curated taxonomy template: {template_path}")
    print(f"Created review instructions: {instructions_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a local human-review packet from a taxonomy JSON file.")
    parser.add_argument("--taxonomy", required=True, help="Path to taxonomy JSON")
    parser.add_argument("--output-dir", default="outputs/review", help="Directory for review outputs")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        create_review_packet(Path(args.taxonomy).expanduser().resolve(), Path(args.output_dir).expanduser().resolve())
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
