"""
Phase 03: Category Consolidation (Two-Step)

Step 1: Consolidate runs within each meta-category (parallel).
Step 2: Cross-meta consolidation into a final flat taxonomy.

Usage:
    uv run python scripts/phase_03_category_consolidation.py --sample
    uv run python scripts/phase_03_category_consolidation.py
    uv run python scripts/phase_03_category_consolidation.py --max-categories 15
"""

import argparse
import asyncio
import json
import os
import re
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from lib.experiment_paths import resolve_experiment_artifact_paths
from lib.llm_client import make_openai_client, local_structured_call, local_structured_call_sync, extract_pydantic_json, resolve_concurrency, structured_parse_call
from lib.model_naming import get_provider_name, provider_scoped_path, resolve_provider_path, resolve_temperature

load_dotenv()

EXPERIMENT_DIR = Path(__file__).parent
PROMPTS_DIR = EXPERIMENT_DIR / "prompts"
ARTIFACT_PATHS = resolve_experiment_artifact_paths(EXPERIMENT_DIR)
DATA_DIR = ARTIFACT_PATHS.data_dir
INTERMEDIATE_DIR = ARTIFACT_PATHS.intermediate_dir

DEFAULT_MODEL = "openai/gpt-5.2"
DEFAULT_TARGET = "adverse_events"
DEFAULT_DOMAIN_CONTEXT = "Free-text entries from stroke intervention documentation."
DEFAULT_MAX_CATEGORIES = 18
DEFAULT_CONCURRENCY = 10
DEFAULT_REQUEST_TIMEOUT_SECONDS = 180.0


def configure_paths(run_id: str | None) -> None:
    global ARTIFACT_PATHS, DATA_DIR, INTERMEDIATE_DIR
    ARTIFACT_PATHS = resolve_experiment_artifact_paths(EXPERIMENT_DIR, run_id=run_id)
    DATA_DIR = ARTIFACT_PATHS.data_dir
    INTERMEDIATE_DIR = ARTIFACT_PATHS.intermediate_dir


class CategoryCluster(BaseModel):
    code: str = Field(description="Canonical category code in snake_case")
    description: str = Field(description="Brief description")
    members: list[str] = Field(description="Members like 'seed_42:infection' or 'meta:category'")


class PerMetaConsolidationResult(BaseModel):
    meta_category: str = Field(description="Meta-category code")
    grouping_criterion: str = Field(description="Grouping criterion used")
    reasoning: str = Field(description="Summary of consolidation decisions")
    clusters: list[CategoryCluster]


class MetaConsolidation(BaseModel):
    meta_code: str = Field(description="Meta-category code")
    categories: list[CategoryCluster]


class CategoryConsolidationResult(BaseModel):
    reasoning: str = Field(description="Summary of consolidation decisions")
    metas: list[MetaConsolidation]


def get_client(base_url: str | None = None):
    return make_openai_client(base_url=base_url, async_client=True)


def load_prompt_phase_03_cross_meta() -> str:
    return (PROMPTS_DIR / "phase_03_category_consolidation_cross_meta.md").read_text()


def load_prompt_phase_03_per_meta() -> str:
    return (PROMPTS_DIR / "phase_03_category_consolidation_per_meta.md").read_text()


def load_meta_categories(model: str) -> dict:
    meta_file = resolve_provider_path(
        INTERMEDIATE_DIR / "meta_categories_consolidated.json",
        model,
        fallback=DATA_DIR / "meta_categories.json",
    )
    if not meta_file.exists():
        raise FileNotFoundError("meta_categories*.json not found")

    with open(meta_file) as f:
        return json.load(f)


def load_runs(runs_dir: Path, model: str) -> list[dict]:
    if not runs_dir.exists():
        raise FileNotFoundError(f"Category runs directory not found: {runs_dir}")

    provider = get_provider_name(model)
    provider_pattern = re.compile(rf"cat_seed_\d+_{re.escape(provider)}\.json$")
    legacy_pattern = re.compile(r"cat_seed_\d+\.json$")

    provider_paths = sorted(
        path for path in runs_dir.glob("*/*.json")
        if provider_pattern.fullmatch(path.name)
    )
    legacy_paths = sorted(
        path for path in runs_dir.glob("*/*.json")
        if legacy_pattern.fullmatch(path.name)
    )
    paths = provider_paths or legacy_paths

    runs = []
    for path in paths:
        with open(path) as f:
            data = json.load(f)
        runs.append(data)

    if not runs:
        raise FileNotFoundError(f"No category runs found in {runs_dir}")
    return runs


def format_runs_for_meta(runs: list[dict]) -> str:
    runs_by_seed: dict[str, list[dict]] = {}
    for r in runs:
        seed = r.get("seed", "unknown")
        runs_by_seed.setdefault(f"seed_{seed}", []).append(r)

    lines = []
    for seed_id, seed_runs in runs_by_seed.items():
        lines.append(f"## {seed_id}")
        for r in seed_runs:
            for cat in r.get("categories", []):
                code = cat.get("code", cat.get("name", "unknown"))
                desc = cat.get("description", "")
                lines.append(f"- {seed_id}:{code}: {desc}")
        lines.append("")
    return "\n".join(lines).strip()


def format_categories_for_prompt(categories_by_meta: dict[str, dict]) -> str:
    lines = []
    for meta_code, meta_data in categories_by_meta.items():
        meta_desc = meta_data.get("meta_description", "")
        lines.append(f"## {meta_code} ({meta_desc})")
        for cat in meta_data.get("categories", []):
            code = cat.get("code", "unknown")
            desc = cat.get("description", "")
            lines.append(f"- {meta_code}:{code}: {desc}")
        lines.append("")
    return "\n".join(lines).strip()


async def consolidate_per_meta(
    client: AsyncOpenAI,
    runs: list[dict],
    meta_code: str,
    meta_description: str,
    model: str,
    max_categories: int,
    semaphore: asyncio.Semaphore,
    local: bool = False,
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
) -> PerMetaConsolidationResult:
    prompt_template = load_prompt_phase_03_per_meta()
    runs_formatted = format_runs_for_meta(runs)

    prompt = (
        prompt_template
        .replace("{meta_category}", meta_code)
        .replace("{meta_description}", meta_description)
        .replace("{max_categories}", str(max_categories))
        .replace("{runs}", runs_formatted)
    )

    async with semaphore:
        if local:
            return await local_structured_call(client, model, prompt, PerMetaConsolidationResult)

        return await structured_parse_call(
            client,
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format=PerMetaConsolidationResult,
            extra_body={
                "provider": {"zdr": True},
                "reasoning": {"effort": "high"},
            },
            temperature=resolve_temperature(model, 0.2),
            request_timeout_seconds=request_timeout_seconds,
        )


async def consolidate_cross_meta(
    client: AsyncOpenAI,
    categories_text: str,
    target: str,
    domain_context: str,
    meta_categories: list[dict],
    model: str,
    max_categories: int,
    local: bool = False,
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
) -> CategoryConsolidationResult:
    prompt_template = load_prompt_phase_03_cross_meta()

    meta_list = "\n".join(
        f"- {m['code']}: {m.get('description', '')}"
        for m in meta_categories
    )

    prompt = (
        prompt_template
        .replace("{target}", target)
        .replace("{domain_context}", domain_context)
        .replace("{meta_categories}", meta_list)
        .replace("{max_categories}", str(max_categories))
        .replace("{categories}", categories_text)
    )

    if local:
        return await local_structured_call(client, model, prompt, CategoryConsolidationResult)

    return await structured_parse_call(
        client,
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format=CategoryConsolidationResult,
        extra_body={
            "provider": {"zdr": True},
            "reasoning": {"effort": "high"},
        },
        temperature=resolve_temperature(model, 0.2),
    )


def compute_support_by_seed(members: list[str]) -> int:
    run_ids = set()
    for m in members:
        run_id = m.split(":", 1)[0].strip()
        if run_id:
            run_ids.add(run_id)
    return len(run_ids)


def compute_support_from_map(members: list[str], support_map: dict[str, int]) -> int:
    return sum(support_map.get(m, 0) for m in members)


def build_intermediate_payload(
    target: str,
    domain_context: str,
    model: str,
    max_categories: int,
    meta_categories: list[dict],
    categories_by_meta: dict[str, dict],
    grouping_by_meta: dict[str, str],
    reasoning_by_meta: dict[str, str],
    runs: list[dict],
) -> dict:
    return {
        "target": target,
        "domain_context": domain_context,
        "model": model,
        "max_categories": max_categories,
        "meta_categories": [m["code"] for m in meta_categories],
        "categories_by_meta": categories_by_meta,
        "grouping_criteria": grouping_by_meta,
        "reasoning_by_meta": reasoning_by_meta,
        "runs": sorted({r.get("seed") for r in runs if "seed" in r}),
    }


def save_intermediate(payload: dict, path: Path):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


async def main(
    sample_mode: bool = False,
    runs_dir: Path = None,
    output_file: Path = None,
    target: str = DEFAULT_TARGET,
    domain_context: str = DEFAULT_DOMAIN_CONTEXT,
    model: str = DEFAULT_MODEL,
    max_categories: int = DEFAULT_MAX_CATEGORIES,
    concurrency: int | None = None,
    base_url: str | None = None,
    run_id: str | None = None,
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
):
    configure_paths(run_id)
    if runs_dir is None:
        runs_dir = INTERMEDIATE_DIR / "category_runs"
    if output_file is None:
        output_file = provider_scoped_path(INTERMEDIATE_DIR / "categories_consolidated.json", model)

    intermediate_file = provider_scoped_path(INTERMEDIATE_DIR / "categories_by_meta_consolidated.json", model)
    print(f"Artifact scope: {DATA_DIR}")

    meta_data = load_meta_categories(model)
    meta_categories = meta_data.get("meta_categories", [])
    if meta_categories and isinstance(meta_categories[0], str):
        meta_categories = [{"code": m, "description": ""} for m in meta_categories]

    runs = load_runs(runs_dir, model)
    runs_by_meta: dict[str, list[dict]] = defaultdict(list)
    for r in runs:
        meta_code = r.get("meta_category")
        if meta_code:
            runs_by_meta[meta_code].append(r)

    if sample_mode:
        meta_subset = [meta_categories[0]["code"]] if meta_categories else list(runs_by_meta)[:1]
        runs_by_meta = {meta: runs_by_meta.get(meta, [])[:2] for meta in meta_subset}
        print(f"=== SAMPLE MODE: consolidating metas {list(runs_by_meta)} ===")

    client, local = get_client(base_url=base_url)
    concurrency = resolve_concurrency(concurrency, local, model=model)
    semaphore = asyncio.Semaphore(concurrency)
    io_lock = asyncio.Lock()

    # Step 1: per-meta consolidation
    categories_by_meta: dict[str, dict] = {}
    reasoning_by_meta: dict[str, str] = {}
    grouping_by_meta: dict[str, str] = {}

    meta_iter = meta_categories
    if sample_mode:
        meta_iter = [m for m in meta_categories if m["code"] in runs_by_meta]

    total = len(meta_iter)
    completed = [0]
    errors: list[tuple[str, str]] = []

    print(f"Step 1: per-meta consolidation ({total} metas), concurrency {concurrency}")

    async def process_meta(meta_entry: dict) -> None:
        meta_code = meta_entry["code"]
        meta_desc = meta_entry.get("description", "")
        is_target = meta_entry.get("is_target", False)
        meta_runs = runs_by_meta.get(meta_code, [])

        try:
            if meta_runs:
                result = await consolidate_per_meta(
                    client,
                    meta_runs,
                    meta_code,
                    meta_desc,
                    model,
                    max_categories,
                    semaphore,
                    local=local,
                )
                reasoning = result.reasoning
                grouping = result.grouping_criterion

                categories = []
                for cluster in result.clusters:
                    categories.append({
                        "code": cluster.code,
                        "description": cluster.description,
                        "support": compute_support_by_seed(cluster.members),
                        "members": cluster.members,
                    })
            else:
                reasoning = "No runs available"
                grouping = ""
                categories = []

            # Ensure "other" exists
            if "other" not in [c["code"] for c in categories]:
                categories.append({
                    "code": "other",
                    "description": "Catch-all for unclassifiable samples",
                    "support": 0,
                    "members": [],
                })

            meta_payload = {
                "meta_description": meta_desc,
                "is_target": is_target,
                "grouping_criterion": grouping,
                "reasoning": reasoning,
                "categories": categories,
            }

            async with io_lock:
                categories_by_meta[meta_code] = meta_payload
                reasoning_by_meta[meta_code] = reasoning
                grouping_by_meta[meta_code] = grouping
                completed[0] += 1
                print(f"  [{completed[0]}/{total}] Meta {meta_code} consolidated")

                if not sample_mode:
                    payload = build_intermediate_payload(
                        target,
                        domain_context,
                        model,
                        max_categories,
                        meta_categories,
                        categories_by_meta,
                        grouping_by_meta,
                        reasoning_by_meta,
                        runs,
                    )
                    save_intermediate(payload, intermediate_file)
        except Exception as exc:  # noqa: BLE001
            async with io_lock:
                errors.append((meta_code, str(exc)))
                categories_by_meta[meta_code] = {
                    "meta_description": meta_desc,
                    "is_target": is_target,
                    "grouping_criterion": "",
                    "reasoning": f"Error: {exc}",
                    "categories": [{
                        "code": "other",
                        "description": "Catch-all for unclassifiable samples",
                        "support": 0,
                        "members": [],
                    }],
                }
                reasoning_by_meta[meta_code] = f"Error: {exc}"
                grouping_by_meta[meta_code] = ""
                completed[0] += 1
                print(f"  [{completed[0]}/{total}] Meta {meta_code} failed: {exc}")

                if not sample_mode:
                    payload = build_intermediate_payload(
                        target,
                        domain_context,
                        model,
                        max_categories,
                        meta_categories,
                        categories_by_meta,
                        grouping_by_meta,
                        reasoning_by_meta,
                        runs,
                    )
                    save_intermediate(payload, intermediate_file)

    tasks = [process_meta(m) for m in meta_iter]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    failed = [r for r in results if isinstance(r, Exception)]
    if failed or errors:
        print(f"Step 1 completed with {len(errors) + len(failed)} errors")
        for meta_code, err in errors[:5]:
            print(f"  - {meta_code}: {err}")

    # Ensure all metas exist (even if omitted by model)
    for meta_entry in meta_categories:
        meta_code = meta_entry["code"]
        if meta_code in categories_by_meta:
            continue
        categories_by_meta[meta_code] = {
            "meta_description": meta_entry.get("description", ""),
            "is_target": meta_entry.get("is_target", False),
            "grouping_criterion": "",
            "reasoning": "No runs available",
            "categories": [{
                "code": "other",
                "description": "Catch-all for unclassifiable samples",
                "support": 0,
                "members": [],
            }],
        }
        reasoning_by_meta[meta_code] = "No runs available"
        grouping_by_meta[meta_code] = ""

    if not sample_mode:
        payload = build_intermediate_payload(
            target,
            domain_context,
            model,
            max_categories,
            meta_categories,
            categories_by_meta,
            grouping_by_meta,
            reasoning_by_meta,
            runs,
        )
        save_intermediate(payload, intermediate_file)
        print(f"Saved per-meta consolidation to: {intermediate_file}")

    # Step 2: cross-meta consolidation into final flat taxonomy
    print("Step 2: cross-meta consolidation")
    categories_text = format_categories_for_prompt(categories_by_meta)
    meta_for_step2 = [m for m in meta_categories if m["code"] in categories_by_meta]

    result = await consolidate_cross_meta(
        client,
        categories_text,
        target,
        domain_context,
        meta_for_step2,
        model,
        max_categories,
        local=local,
    )

    meta_codes = {m["code"] for m in meta_for_step2}
    categories_by_meta_final: dict[str, dict] = {}

    support_map = {}
    for meta_code, meta_data in categories_by_meta.items():
        for cat in meta_data.get("categories", []):
            support_map[f"{meta_code}:{cat['code']}"] = cat.get("support", 0)

    for meta in result.metas:
        if meta.meta_code not in meta_codes:
            raise ValueError(f"Unknown meta_code in consolidation output: {meta.meta_code}")

        categories = []
        for cluster in meta.categories:
            categories.append({
                "code": cluster.code,
                "description": cluster.description,
                "support": compute_support_from_map(cluster.members, support_map),
                "members": cluster.members,
            })

        # Ensure "other" exists
        if "other" not in [c["code"] for c in categories]:
            categories.append({
                "code": "other",
                "description": "Catch-all for unclassifiable samples",
                "support": 0,
                "members": [],
            })

        meta_desc = next((m.get("description", "") for m in meta_for_step2 if m["code"] == meta.meta_code), "")
        is_target = next((m.get("is_target", False) for m in meta_for_step2 if m["code"] == meta.meta_code), False)

        categories_by_meta_final[meta.meta_code] = {
            "meta_description": meta_desc,
            "is_target": is_target,
            "categories": categories,
        }

    # Ensure all metas exist (even if omitted by model)
    for meta_code in meta_codes:
        if meta_code in categories_by_meta_final:
            continue
        meta_desc = next((m.get("description", "") for m in meta_for_step2 if m["code"] == meta_code), "")
        is_target = next((m.get("is_target", False) for m in meta_for_step2 if m["code"] == meta_code), False)
        categories_by_meta_final[meta_code] = {
            "meta_description": meta_desc,
            "is_target": is_target,
            "categories": [{
                "code": "other",
                "description": "Catch-all for unclassifiable samples",
                "support": 0,
                "members": [],
            }],
        }

    if sample_mode:
        print("Sample mode - not saving results")
        return result

    output_data_final = {
        "target": target,
        "domain_context": domain_context,
        "model": model,
        "max_categories": max_categories,
        "meta_categories": [m["code"] for m in meta_for_step2],
        "categories_by_meta": categories_by_meta_final,
        "reasoning": result.reasoning,
        "runs": sorted({r.get("seed") for r in runs if "seed" in r}),
    }

    with open(output_file, "w") as f:
        json.dump(output_data_final, f, indent=2, ensure_ascii=False)
    print(f"Saved consolidated categories to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 03: Category Consolidation (Two-Step)")
    parser.add_argument("--sample", action="store_true", help="Sample mode (no save)")
    parser.add_argument("--runs-dir", type=Path, default=None, help="Directory with category runs")
    parser.add_argument("--output-file", type=Path, default=None, help="Output JSON file")
    parser.add_argument("--target", type=str, default=DEFAULT_TARGET, help="Target concept")
    parser.add_argument("--domain-context", type=str, default=DEFAULT_DOMAIN_CONTEXT, help="Domain context")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model to use")
    parser.add_argument("--max-categories", type=int, default=DEFAULT_MAX_CATEGORIES, help="Max categories per meta")
    parser.add_argument("--concurrency", type=int, default=None, help="Parallel meta consolidations (defaults: Qwen 27B=8, Qwen 9B=20, otherwise 10; local=1)")
    parser.add_argument("--base-url", type=str, default=None, help="Custom API base URL (e.g. LM Studio)")
    parser.add_argument("--run-id", type=str, default=None, help="Optional run id for isolated artifacts under data/runs/<run_id>/")
    parser.add_argument("--request-timeout-seconds", type=float, default=DEFAULT_REQUEST_TIMEOUT_SECONDS, help="Per-attempt timeout for a single request")

    args = parser.parse_args()
    asyncio.run(main(
        sample_mode=args.sample,
        runs_dir=args.runs_dir,
        output_file=args.output_file,
        target=args.target,
        domain_context=args.domain_context,
        model=args.model,
        max_categories=args.max_categories,
        concurrency=args.concurrency,
        base_url=args.base_url,
        run_id=args.run_id,
        request_timeout_seconds=args.request_timeout_seconds,
    ))
