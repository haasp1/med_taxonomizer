"""
Prompt utilities for taxonomy-based labeling.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_prompt_template(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def build_output_schema() -> str:
    schema = {
        "samples": [
            {
                "sample_id": "string",
                "labels": [
                    {
                        "meta": "string",
                        "code": "string",
                        "reasoning": "string",
                    }
                ],
            }
        ]
    }
    return json.dumps(schema, indent=2, ensure_ascii=False)


def _get(item: Any, key: str) -> Any:
    if isinstance(item, dict):
        return item.get(key)
    return getattr(item, key)


def format_samples(samples: list[Any]) -> str:
    lines = []
    for s in samples:
        sample_id = _get(s, "sample_id")
        text = _get(s, "text")
        text = " ".join(str(text).split())
        lines.append(f"[{sample_id}] {text}")
    return "\n".join(lines)


def build_prompt(
    template: str,
    *,
    domain_context: str,
    target: str,
    language_note: str,
    taxonomy_md: str,
    output_schema: str,
    samples_text: str,
) -> str:
    return (
        template
        .replace("{domain_context}", domain_context)
        .replace("{target}", target)
        .replace("{language_note}", language_note)
        .replace("{taxonomy}", taxonomy_md)
        .replace("{output_schema}", output_schema)
        .replace("{samples}", samples_text)
    )
