"""
Phase 08: Final Tree Consolidation (Per Meta)

Generate the final taxonomy as simple meta-relative slash paths.

Usage:
    uv run python scripts/phase_08_tree_consolidation.py --sample
    uv run python scripts/phase_08_tree_consolidation.py
"""

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from lib.experiment_paths import resolve_experiment_artifact_paths
from lib.llm_client import (
    ValidationRetryError,
    append_validation_feedback_messages,
    append_validation_feedback_prompt,
    local_structured_call,
    make_openai_client,
    resolve_concurrency,
    run_with_validation_repair,
    structured_parse_call,
)
from lib.model_naming import provider_scoped_path, resolve_provider_path, resolve_temperature
from lib.taxonomy_paths import normalize_slug

load_dotenv()

EXPERIMENT_DIR = Path(__file__).parent
PROMPTS_DIR = EXPERIMENT_DIR / "prompts"
ARTIFACT_PATHS = resolve_experiment_artifact_paths(EXPERIMENT_DIR)
DATA_DIR = ARTIFACT_PATHS.data_dir
INTERMEDIATE_DIR = ARTIFACT_PATHS.intermediate_dir

DEFAULT_MODEL = "openai/gpt-5.2"
DEFAULT_INPUT_MODEL = "openai/gpt-4o"
DEFAULT_DOMAIN_CONTEXT = "Free-text entries from stroke intervention documentation."
DEFAULT_TARGET = "adverse_events"
DEFAULT_MAX_CHILDREN = 5
DEFAULT_CONCURRENCY = 10
DEFAULT_REQUEST_TIMEOUT_SECONDS = 180.0
DEFAULT_BATCH_MAX_ATTEMPTS = 4
DEFAULT_SYSTEM_PROMPT = (
    "You are a concise, efficient, decisive assistant. "
    "Think in 2-3 short blocks per sample without repetition or second-guessing, "
    "and then output your answer."
)
QWEN_TEMPERATURE = 0.8
QWEN_TOP_P = 0.95
QWEN_TOP_K = 20
QWEN_PRESENCE_PENALTY = 1.5


def configure_paths(run_id: str | None) -> None:
    global ARTIFACT_PATHS, DATA_DIR, INTERMEDIATE_DIR
    ARTIFACT_PATHS = resolve_experiment_artifact_paths(EXPERIMENT_DIR, run_id=run_id)
    DATA_DIR = ARTIFACT_PATHS.data_dir
    INTERMEDIATE_DIR = ARTIFACT_PATHS.intermediate_dir


class PathEntry(BaseModel):
    path: str = Field(description="Canonical meta-relative path such as 'hemorrhage/intracranial_hemorrhage'")
    description: str = Field(description="Brief description")


class PathConsolidationResult(BaseModel):
    reasoning: str
    paths: list[PathEntry]


def get_client(base_url: str | None = None):
    return make_openai_client(base_url=base_url, async_client=True)


def load_prompt() -> str:
    return (PROMPTS_DIR / "phase_08_tree_consolidation.md").read_text()


def load_initial_tree(model: str) -> dict[str, Any]:
    tree_file = resolve_provider_path(INTERMEDIATE_DIR / "taxonomy_tree_initial.json", model)
    if not tree_file.exists():
        raise FileNotFoundError("taxonomy_tree_initial.json not found")
    with open(tree_file) as f:
        return json.load(f)


def format_paths_for_prompt(meta: str, path_rows: list[dict[str, str]]) -> str:
    return json.dumps({"meta": meta, "paths": path_rows}, indent=2, ensure_ascii=False)


def get_generation_kwargs(model: str, *, local: bool, default_temperature: float) -> dict[str, Any]:
    if "qwen" in (model or "").lower():
        extra_body: dict[str, Any] = {"top_k": QWEN_TOP_K}
        if not local:
            extra_body["provider"] = {"zdr": True}
        return {
            "temperature": QWEN_TEMPERATURE,
            "top_p": QWEN_TOP_P,
            "presence_penalty": QWEN_PRESENCE_PENALTY,
            "extra_body": extra_body,
        }

    kwargs: dict[str, Any] = {"temperature": resolve_temperature(model, default_temperature)}
    if not local:
        kwargs["extra_body"] = {"provider": {"zdr": True}}
    return kwargs


def meta_tree_to_path_rows(meta_entry: dict[str, Any]) -> list[dict[str, str]]:
    node_map = {
        node["code"]: {
            "description": str(node.get("description") or ""),
            "children": [child for child in (node.get("children") or []) if isinstance(child, str)],
        }
        for node in meta_entry.get("nodes", [])
        if isinstance(node, dict) and isinstance(node.get("code"), str)
    }
    rows: list[dict[str, str]] = []
    visited: set[str] = set()

    def walk(code: str) -> None:
        if code in visited or code not in node_map:
            return
        visited.add(code)
        rows.append({"path": code, "description": node_map[code]["description"]})
        for child in node_map[code]["children"]:
            walk(child)

    for root in meta_entry.get("roots", []):
        if isinstance(root, str):
            walk(root)

    for code in sorted(node_map):
        walk(code)

    return rows


def normalize_path(path: str) -> str:
    parts = [normalize_slug(part) for part in (path or "").split("/")]
    parts = [part for part in parts if part]
    return "/".join(parts)


def normalize_meta_paths(path_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    normalized_rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in path_rows:
        path = normalize_path(str(row.get("path") or ""))
        if not path or path in seen:
            continue
        seen.add(path)
        normalized_rows.append(
            {
                "path": path,
                "description": str(row.get("description") or ""),
            }
        )
    return normalized_rows


def validate_paths(paths: list[PathEntry], expected_count: int, max_children: int) -> tuple[bool, str]:
    normalized_paths = [normalize_path(entry.path) for entry in paths]

    if len(normalized_paths) != expected_count:
        return False, f"Output path count changed ({len(normalized_paths)} != {expected_count})"
    if any(not path for path in normalized_paths):
        return False, "Empty path detected"
    if len(set(normalized_paths)) != len(normalized_paths):
        return False, "Duplicate paths detected"

    path_set = set(normalized_paths)
    child_counts: dict[str, int] = {}
    root_count = 0
    for path in normalized_paths:
        parts = path.split("/")
        if any(part != normalize_slug(part) for part in parts):
            return False, f"Invalid slug in path: {path}"
        if len(parts) == 1:
            root_count += 1
            continue
        parent = "/".join(parts[:-1])
        if parent not in path_set:
            return False, f"Missing parent path for {path}: {parent}"
        child_counts[parent] = child_counts.get(parent, 0) + 1

    if root_count == 0:
        return False, "No root paths returned"
    overflow = [parent for parent, count in child_counts.items() if count > max_children]
    if overflow:
        return False, f"Parents exceed max_children: {', '.join(sorted(overflow)[:5])}"
    return True, ""


async def consolidate_meta(
    client: AsyncOpenAI,
    meta: str,
    prompt: str,
    model: str,
    semaphore: asyncio.Semaphore,
    expected_count: int,
    max_children: int,
    local: bool = False,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    max_attempts: int = DEFAULT_BATCH_MAX_ATTEMPTS,
) -> dict[str, Any]:
    generation_kwargs = get_generation_kwargs(model, local=local, default_temperature=0.2)
    extra_body = dict(generation_kwargs.get("extra_body") or {})
    if not local:
        extra_body["reasoning"] = {"effort": "xhigh"}

    async def run_attempt(retry_context):
        async with semaphore:
            if local:
                result = await local_structured_call(
                    client,
                    model,
                    f"{system_prompt}\n\n{append_validation_feedback_prompt(prompt, retry_context)}",
                    PathConsolidationResult,
                    temperature=generation_kwargs.get("temperature"),
                    top_p=generation_kwargs.get("top_p"),
                    presence_penalty=generation_kwargs.get("presence_penalty"),
                    extra_body=extra_body or None,
                    max_attempts=1,
                )
            else:
                result = await structured_parse_call(
                    client,
                    model=model,
                    messages=append_validation_feedback_messages(
                        [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt},
                        ],
                        retry_context,
                    ),
                    response_format=PathConsolidationResult,
                    extra_body=extra_body,
                    temperature=generation_kwargs.get("temperature"),
                    top_p=generation_kwargs.get("top_p"),
                    presence_penalty=generation_kwargs.get("presence_penalty"),
                    max_attempts=1,
                    request_timeout_seconds=request_timeout_seconds,
                )
        ok, err = validate_paths(result.paths, expected_count, max_children)
        if not ok:
            raise ValidationRetryError(err, failed_result=result)
        return {
            "meta": meta,
            "paths": normalize_meta_paths(
                [{"path": entry.path, "description": entry.description} for entry in result.paths]
            ),
        }

    return await run_with_validation_repair(
        model=model,
        operation_label=f"tree consolidation {meta}",
        run_attempt=run_attempt,
        max_attempts=max_attempts,
    )


def save_tree(
    output_file: Path,
    domain_context: str,
    target: str,
    model: str,
    metas: list[dict[str, Any]],
) -> None:
    with open(output_file, "w") as f:
        json.dump(
            {
                "domain_context": domain_context,
                "target": target,
                "model": model,
                "metas": [
                    {
                        "meta": meta["meta"],
                        "paths": meta["paths"],
                    }
                    for meta in metas
                ],
            },
            f,
            indent=2,
            ensure_ascii=False,
        )


async def main(
    sample_mode: bool = False,
    domain_context: str = DEFAULT_DOMAIN_CONTEXT,
    target: str = DEFAULT_TARGET,
    max_children: int = DEFAULT_MAX_CHILDREN,
    model: str = DEFAULT_MODEL,
    input_model: str = DEFAULT_INPUT_MODEL,
    concurrency: int | None = None,
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    base_url: str | None = None,
    run_id: str | None = None,
):
    configure_paths(run_id)
    initial_tree = load_initial_tree(input_model)
    metas = initial_tree.get("metas", [])
    if sample_mode:
        metas = metas[:1]
        print(f"=== SAMPLE MODE: {len(metas)} meta trees ===")
    print(f"Artifact scope: {DATA_DIR}")

    client, local = get_client(base_url=base_url)
    concurrency = resolve_concurrency(concurrency, local, model=model)
    prompt_template = load_prompt()
    semaphore = asyncio.Semaphore(concurrency)
    io_lock = asyncio.Lock()

    output_tree = provider_scoped_path(INTERMEDIATE_DIR / "taxonomy_tree_final.json", model)
    final_metas: list[dict[str, Any] | None] = [None] * len(metas)
    completed = [0]
    errors: list[str] = []
    total = len(metas)

    print(f"Consolidating {total} metas with concurrency {concurrency}")

    async def process_meta(idx: int, meta_entry: dict[str, Any]) -> None:
        meta = meta_entry["meta"]
        path_rows = meta_tree_to_path_rows(meta_entry)
        prompt = (
            prompt_template
            .replace("{domain_context}", domain_context)
            .replace("{target}", target)
            .replace("{meta}", meta)
            .replace("{max_children}", str(max_children))
            .replace("{tree}", format_paths_for_prompt(meta, path_rows))
        )

        try:
            payload = await consolidate_meta(
                client,
                meta,
                prompt,
                model,
                semaphore,
                len(path_rows),
                max_children,
                local=local,
                system_prompt=system_prompt,
                request_timeout_seconds=request_timeout_seconds,
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{meta}: {exc}")
            payload = {
                "meta": meta,
                "paths": normalize_meta_paths(path_rows),
            }

        async with io_lock:
            final_metas[idx] = payload
            completed[0] += 1
            print(f"  [{completed[0]}/{total}] Meta {meta} consolidated")

            if not sample_mode:
                metas_out = [entry for entry in final_metas if entry is not None]
                save_tree(output_tree, domain_context, target, model, metas_out)

    await asyncio.gather(*(process_meta(idx, meta_entry) for idx, meta_entry in enumerate(metas)), return_exceptions=True)

    if errors:
        print(f"\nCompleted with {len(errors)} errors:")
        for err in errors[:5]:
            print(f"  - {err}")

    metas_out = [entry for entry in final_metas if entry is not None]
    if sample_mode:
        print("Sample mode - not saving results")
        return metas_out

    save_tree(output_tree, domain_context, target, model, metas_out)
    print(f"Saved final tree: {output_tree}")
    if errors:
        raise RuntimeError(f"Phase 08 incomplete: {len(errors)} metas failed consolidation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 08: Final Tree Consolidation")
    parser.add_argument("--sample", action="store_true", help="Sample mode (no save)")
    parser.add_argument("--domain-context", type=str, default=DEFAULT_DOMAIN_CONTEXT, help="Domain context")
    parser.add_argument("--target", type=str, default=DEFAULT_TARGET, help="Target concept")
    parser.add_argument("--max-children", type=int, default=DEFAULT_MAX_CHILDREN, help="Max children per node")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model to use")
    parser.add_argument("--input-model", type=str, default=DEFAULT_INPUT_MODEL, help="Model used to generate the Phase 07 initial tree")
    parser.add_argument("--concurrency", type=int, default=None, help="Parallel meta consolidations (defaults: Qwen 27B=8, Qwen 9B=20, otherwise 10; local=1)")
    parser.add_argument("--request-timeout-seconds", type=float, default=DEFAULT_REQUEST_TIMEOUT_SECONDS, help="Per-attempt timeout for a single request")
    parser.add_argument("--system-prompt", type=str, default=DEFAULT_SYSTEM_PROMPT, help="System prompt for Phase 08 calls")
    parser.add_argument("--base-url", type=str, default=None, help="Custom API base URL (e.g. LM Studio)")
    parser.add_argument("--run-id", type=str, default=None, help="Optional run id for isolated artifacts under data/runs/<run_id>/")

    args = parser.parse_args()
    asyncio.run(main(
        sample_mode=args.sample,
        domain_context=args.domain_context,
        target=args.target,
        max_children=args.max_children,
        model=args.model,
        input_model=args.input_model,
        concurrency=args.concurrency,
        request_timeout_seconds=args.request_timeout_seconds,
        system_prompt=args.system_prompt,
        base_url=args.base_url,
        run_id=args.run_id,
    ))
