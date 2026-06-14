"""
Phase 07: Subcategory Discovery + Consolidation

Discover subcategories for categories marked as splittable and consolidate
multi-seed results. Builds an initial taxonomy tree.

Usage:
    uv run python scripts/phase_07_subcategory_discovery.py --sample
    uv run python scripts/phase_07_subcategory_discovery.py
"""

import argparse
import asyncio
import json
import os
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from lib.experiment_paths import resolve_experiment_artifact_paths
from lib.llm_client import (
    ValidationRetryError,
    append_validation_feedback_messages,
    append_validation_feedback_prompt,
    extract_pydantic_json,
    local_structured_call,
    local_structured_call_sync,
    make_openai_client,
    resolve_concurrency,
    run_with_validation_repair,
    structured_parse_call,
)
from lib.model_naming import provider_scoped_path, resolve_provider_path, resolve_temperature
from lib.taxonomy_paths import canonical_child_code, canonical_root_code

load_dotenv()

EXPERIMENT_DIR = Path(__file__).parent
PROMPTS_DIR = EXPERIMENT_DIR / "prompts"
ARTIFACT_PATHS = resolve_experiment_artifact_paths(EXPERIMENT_DIR)
DATA_DIR = ARTIFACT_PATHS.data_dir
INTERMEDIATE_DIR = ARTIFACT_PATHS.intermediate_dir
CACHE_DIR = ARTIFACT_PATHS.cache_dir

DEFAULT_MODEL = "openai/gpt-4o"
DEFAULT_TAXONOMY_MODEL = "openai/gpt-5.2"
DEFAULT_LABELING_MODEL = "openai/gpt-4o"
DEFAULT_DOMAIN_CONTEXT = "Free-text entries from stroke intervention documentation."
DEFAULT_TARGET = "adverse_events"
DEFAULT_N_SAMPLES = 60
DEFAULT_SEEDS = [42, 123, 456]
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
    global ARTIFACT_PATHS, DATA_DIR, INTERMEDIATE_DIR, CACHE_DIR
    ARTIFACT_PATHS = resolve_experiment_artifact_paths(EXPERIMENT_DIR, run_id=run_id)
    DATA_DIR = ARTIFACT_PATHS.data_dir
    INTERMEDIATE_DIR = ARTIFACT_PATHS.intermediate_dir
    CACHE_DIR = ARTIFACT_PATHS.cache_dir


class Subcategory(BaseModel):
    code: str = Field(description="Short name in snake_case (English)")
    description: str = Field(description="Brief description of this subcategory")


class SubcategoryDiscoveryResult(BaseModel):
    parent_code: str = Field(description="Parent category")
    grouping_criterion: str = Field(description="Criterion used to group subcategories")
    categories: list[Subcategory]
    reasoning: str = Field(description="Why this grouping makes sense")


class SubcategoryCluster(BaseModel):
    code: str = Field(description="Canonical subcategory code in snake_case")
    description: str = Field(description="Brief description")
    members: list[str] = Field(description="Run-specific members like 'seed_42:pneumonia'")


class SubcategoryConsolidationResult(BaseModel):
    parent_code: str = Field(description="Parent category")
    grouping_criterion: str = Field(description="Criterion used to group subcategories")
    reasoning: str = Field(description="Summary of consolidation decisions")
    clusters: list[SubcategoryCluster]


def get_client(base_url: str | None = None):
    return make_openai_client(base_url=base_url, async_client=True)


def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text()


def load_taxonomy(taxonomy_model: str) -> dict:
    cat_file = resolve_provider_path(
        INTERMEDIATE_DIR / "categories_consolidated.json",
        taxonomy_model,
        fallback=DATA_DIR / "categories.json",
    )
    if not cat_file.exists():
        raise FileNotFoundError("No consolidated or legacy categories file found")
    with open(cat_file) as f:
        return json.load(f)


def load_split_summary(model: str) -> dict:
    summary_file = resolve_provider_path(INTERMEDIATE_DIR / "split_audit_summary.json", model)
    if not summary_file.exists():
        raise FileNotFoundError("split_audit_summary.json not found")
    with open(summary_file) as f:
        return json.load(f)


def load_texts_by_category(labeling_model: str) -> dict[tuple[str, str], list[str]]:
    db_path = resolve_provider_path(CACHE_DIR / "labels.db", labeling_model)
    texts_by_cat: dict[tuple[str, str], list[str]] = defaultdict(list)

    if db_path.exists():
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("""
            SELECT s.text, l.meta, l.category
            FROM labels l
            JOIN samples s ON s.id = l.sample_id
        """)
        for text, meta, category in cur.fetchall():
            texts_by_cat[(meta, category)].append(text)
        conn.close()
        return texts_by_cat

    jsonl_path = resolve_provider_path(CACHE_DIR / "labels.jsonl", labeling_model)
    if not jsonl_path.exists():
        raise FileNotFoundError("labels.db or labels.jsonl not found")

    with open(jsonl_path) as f:
        for line in f:
            record = json.loads(line)
            text = record.get("text", "")
            for lbl in record.get("labels", []):
                meta = lbl.get("meta")
                category = lbl.get("category")
                if meta and category:
                    texts_by_cat[(meta, category)].append(text)
    return texts_by_cat


def format_samples(texts: list[str]) -> str:
    return "\n".join(f"{i+1}. {text}" for i, text in enumerate(texts))


def format_retry_error(exc: Exception) -> str:
    message = " ".join(str(exc).split())
    if len(message) > 240:
        return f"{message[:237]}..."
    return message


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


async def run_discovery(
    client: AsyncOpenAI,
    texts: list[str],
    parent_code: str,
    parent_description: str,
    existing_codes: list[str],
    domain_context: str,
    target: str,
    max_children: int,
    model: str,
    semaphore: asyncio.Semaphore,
    local: bool = False,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    max_attempts: int = DEFAULT_BATCH_MAX_ATTEMPTS,
) -> SubcategoryDiscoveryResult:
    prompt = load_prompt("phase_07_subcategory_discovery.md")
    prompt = (
        prompt
        .replace("{domain_context}", domain_context)
        .replace("{target}", target)
        .replace("{parent_code}", parent_code)
        .replace("{parent_description}", parent_description)
        .replace("{existing_codes}", ", ".join(existing_codes) if existing_codes else "none")
        .replace("{max_children}", str(max_children))
        .replace("{samples}", format_samples(texts))
    )

    generation_kwargs = get_generation_kwargs(model, local=local, default_temperature=0.2)

    async def run_attempt(retry_context):
        async with semaphore:
            if local:
                result = await local_structured_call(
                    client,
                    model,
                    f"{system_prompt}\n\n{append_validation_feedback_prompt(prompt, retry_context)}",
                    SubcategoryDiscoveryResult,
                    temperature=generation_kwargs.get("temperature"),
                    top_p=generation_kwargs.get("top_p"),
                    presence_penalty=generation_kwargs.get("presence_penalty"),
                    extra_body=generation_kwargs.get("extra_body"),
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
                    response_format=SubcategoryDiscoveryResult,
                    extra_body=generation_kwargs.get("extra_body"),
                    temperature=generation_kwargs.get("temperature"),
                    top_p=generation_kwargs.get("top_p"),
                    presence_penalty=generation_kwargs.get("presence_penalty"),
                    max_attempts=1,
                    request_timeout_seconds=request_timeout_seconds,
                )
        return result

    return await run_with_validation_repair(
        model=model,
        operation_label=f"subcategory discovery {parent_code}",
        run_attempt=run_attempt,
        max_attempts=max_attempts,
    )


async def run_consolidation(
    client: AsyncOpenAI,
    runs: list[dict],
    parent_code: str,
    parent_description: str,
    existing_codes: list[str],
    domain_context: str,
    target: str,
    max_children: int,
    model: str,
    semaphore: asyncio.Semaphore,
    local: bool = False,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    max_attempts: int = DEFAULT_BATCH_MAX_ATTEMPTS,
) -> SubcategoryConsolidationResult:
    prompt = load_prompt("phase_07_subcategory_consolidation.md")

    # Format runs
    lines = []
    for r in runs:
        seed_id = f"seed_{r.get('seed')}"
        lines.append(f"## {seed_id}")
        for cat in r.get("categories", []):
            code = cat.get("code", cat.get("name", "unknown"))
            desc = cat.get("description", "")
            lines.append(f"- {seed_id}:{code}: {desc}")
        lines.append("")
    runs_formatted = "\n".join(lines).strip()

    prompt = (
        prompt
        .replace("{domain_context}", domain_context)
        .replace("{target}", target)
        .replace("{parent_code}", parent_code)
        .replace("{parent_description}", parent_description)
        .replace("{existing_codes}", ", ".join(existing_codes) if existing_codes else "none")
        .replace("{max_children}", str(max_children))
        .replace("{runs}", runs_formatted)
    )

    generation_kwargs = get_generation_kwargs(model, local=local, default_temperature=0.2)

    async def run_attempt(retry_context):
        async with semaphore:
            if local:
                result = await local_structured_call(
                    client,
                    model,
                    f"{system_prompt}\n\n{append_validation_feedback_prompt(prompt, retry_context)}",
                    SubcategoryConsolidationResult,
                    temperature=generation_kwargs.get("temperature"),
                    top_p=generation_kwargs.get("top_p"),
                    presence_penalty=generation_kwargs.get("presence_penalty"),
                    extra_body=generation_kwargs.get("extra_body"),
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
                    response_format=SubcategoryConsolidationResult,
                    extra_body=generation_kwargs.get("extra_body"),
                    temperature=generation_kwargs.get("temperature"),
                    top_p=generation_kwargs.get("top_p"),
                    presence_penalty=generation_kwargs.get("presence_penalty"),
                    max_attempts=1,
                    request_timeout_seconds=request_timeout_seconds,
                )
        return result

    return await run_with_validation_repair(
        model=model,
        operation_label=f"subcategory consolidation {parent_code}",
        run_attempt=run_attempt,
        max_attempts=max_attempts,
    )


def get_run_path(output_dir: Path, meta: str, parent_code: str, seed: int, model: str) -> Path:
    return provider_scoped_path(output_dir / meta / parent_code / f"subcat_seed_{seed}.json", model)


def save_run(output_dir: Path, meta: str, parent_code: str, seed: int, data: dict, model: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = get_run_path(output_dir, meta, parent_code, seed, model)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved: {output_file}")


def save_consolidated(path: Path, data: dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def compute_support(members: list[str]) -> int:
    run_ids = set()
    for m in members:
        run_id = m.split(":", 1)[0].strip()
        if run_id:
            run_ids.add(run_id)
    return len(run_ids)


async def main(
    sample_mode: bool = False,
    n_samples: int = DEFAULT_N_SAMPLES,
    seeds: list[int] = None,
    max_children: int = DEFAULT_MAX_CHILDREN,
    concurrency: int | None = None,
    domain_context: str = DEFAULT_DOMAIN_CONTEXT,
    target: str = DEFAULT_TARGET,
    model: str = DEFAULT_MODEL,
    taxonomy_model: str = DEFAULT_TAXONOMY_MODEL,
    labeling_model: str = DEFAULT_LABELING_MODEL,
    resume: bool = False,
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    base_url: str | None = None,
    run_id: str | None = None,
):
    configure_paths(run_id)
    print(f"Artifact scope: {DATA_DIR}")
    if seeds is None:
        seeds = DEFAULT_SEEDS

    taxonomy = load_taxonomy(taxonomy_model)
    split_summary = load_split_summary(model)
    texts_by_cat = load_texts_by_category(labeling_model)

    # Determine split targets
    split_targets = [
        v for v in split_summary.values() if v.get("should_split")
    ]
    if sample_mode:
        split_targets = split_targets[:1]
        seeds = seeds[:2]
        n_samples = min(n_samples, 20)
        print(f"=== SAMPLE MODE: {len(split_targets)} parents, seeds {seeds} ===")

    output_dir = INTERMEDIATE_DIR / "subcategory_runs"
    consolidated_by_parent = {}
    client, local = get_client(base_url=base_url)
    concurrency = resolve_concurrency(concurrency, local, model=model)
    semaphore = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()
    print(f"Concurrency: {concurrency} parallel requests")

    parent_contexts: dict[str, dict[str, Any]] = {}
    discovery_jobs = []
    resumed_runs = 0

    for target_entry in split_targets:
        meta = target_entry["meta"]
        parent_code = target_entry["parent_code"]
        parent_desc = target_entry.get("parent_description", "")
        existing_codes = [
            c.get("code", c.get("name", "unknown"))
            for c in taxonomy["categories_by_meta"][meta].get("categories", [])
        ]

        key = f"{meta}:{parent_code}"
        parent_contexts[key] = {
            "meta": meta,
            "parent_code": parent_code,
            "parent_desc": parent_desc,
            "existing_codes": existing_codes,
        }

        texts = texts_by_cat.get((meta, parent_code), [])
        if not texts:
            continue
        df = pd.DataFrame({"text": texts})
        for seed in seeds:
            output_path = get_run_path(output_dir, meta, parent_code, seed, model)
            if resume and output_path.exists() and not sample_mode:
                with open(output_path) as f:
                    run_data = json.load(f)
                discovery_jobs.append({
                    "key": key,
                    "meta": meta,
                    "parent_code": parent_code,
                    "seed": seed,
                    "resume_data": run_data,
                })
                resumed_runs += 1
                continue
            sample_df = df.sample(n=min(n_samples, len(df)), random_state=seed)
            sample_texts = sample_df["text"].tolist()
            discovery_jobs.append({
                "key": key,
                "meta": meta,
                "parent_code": parent_code,
                "parent_desc": parent_desc,
                "existing_codes": existing_codes,
                "seed": seed,
                "sample_texts": sample_texts,
            })

    if not discovery_jobs:
        print("No discovery jobs to run (no split targets with available texts).")
    elif resume and resumed_runs:
        print(f"Resume enabled: reusing {resumed_runs} saved subcategory discovery runs")

    discovery_total = len(discovery_jobs)
    discovery_done = [0]

    async def process_discovery(job: dict[str, Any]):
        if "resume_data" in job:
            run_data = job["resume_data"]
            async with write_lock:
                discovery_done[0] += 1
                print(
                    f"[{discovery_done[0]}/{discovery_total}] Resume {job['meta']}:{job['parent_code']} seed {job['seed']}"
                )
            return job["key"], run_data
        try:
            result = await run_discovery(
                client,
                job["sample_texts"],
                parent_code=job["parent_code"],
                parent_description=job["parent_desc"],
                existing_codes=job["existing_codes"],
                domain_context=domain_context,
                target=target,
                max_children=max_children,
                model=model,
                semaphore=semaphore,
                local=local,
                system_prompt=system_prompt,
                request_timeout_seconds=request_timeout_seconds,
            )
        except Exception as exc:
            return RuntimeError(
                f"Discovery failed for {job['meta']}:{job['parent_code']} seed {job['seed']}: {exc}"
            )

        run_data = {
            "seed": job["seed"],
            "meta": job["meta"],
            "parent_code": job["parent_code"],
            "parent_description": job["parent_desc"],
            "n_samples": len(job["sample_texts"]),
            "grouping_criterion": result.grouping_criterion,
            "reasoning": result.reasoning,
            "categories": [c.model_dump() for c in result.categories],
        }

        if not sample_mode:
            async with write_lock:
                save_run(output_dir, job["meta"], job["parent_code"], job["seed"], run_data, model)

        async with write_lock:
            discovery_done[0] += 1
            print(
                f"[{discovery_done[0]}/{discovery_total}] Discovery {job['meta']}:{job['parent_code']} seed {job['seed']}"
            )

        return job["key"], run_data

    discovery_tasks = [process_discovery(job) for job in discovery_jobs]
    discovery_results = await asyncio.gather(*discovery_tasks, return_exceptions=True)

    runs_by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    discovery_errors = [r for r in discovery_results if isinstance(r, Exception)]
    for result in discovery_results:
        if isinstance(result, Exception):
            continue
        key, run_data = result
        runs_by_parent[key].append(run_data)

    if discovery_errors:
        print(f"\n{len(discovery_errors)} discovery runs failed:")
        for err in discovery_errors[:5]:
            print(f"  - {err}")

    consolidation_total = len(runs_by_parent)
    consolidation_done = [0]
    consolidated_path = provider_scoped_path(INTERMEDIATE_DIR / "subcategories_consolidated.json", model)

    async def process_consolidation(parent_key: str, runs: list[dict[str, Any]]):
        context = parent_contexts[parent_key]
        meta = context["meta"]
        parent_code = context["parent_code"]
        parent_desc = context["parent_desc"]
        existing_codes = context["existing_codes"]
        parent_tree_code = canonical_root_code(parent_code)

        try:
            consolidation = await run_consolidation(
                client,
                runs,
                parent_code=parent_code,
                parent_description=parent_desc,
                existing_codes=existing_codes,
                domain_context=domain_context,
                target=target,
                max_children=max_children,
                model=model,
                semaphore=semaphore,
                local=local,
                system_prompt=system_prompt,
                request_timeout_seconds=request_timeout_seconds,
            )
        except Exception as exc:
            return RuntimeError(f"Consolidation failed for {parent_key}: {exc}")

        clusters = []
        seen_codes: set[str] = set()
        for cluster in consolidation.clusters:
            code = canonical_child_code(parent_tree_code, cluster.code)
            if code in existing_codes:
                print(f"Skipping subcategory that duplicates top-level code: {code}")
                continue
            if code in seen_codes:
                continue
            seen_codes.add(code)
            clusters.append({
                "code": code,
                "description": cluster.description,
                "support": compute_support(cluster.members),
                "members": cluster.members,
            })

        expected_other = canonical_child_code(parent_tree_code, "other")
        if expected_other not in [c["code"] for c in clusters]:
            clusters.append({
                "code": expected_other,
                "description": "Catch-all for unclassifiable samples within this parent category",
                "support": 0,
                "members": [],
            })

        entry = {
            "meta": meta,
            "parent_code": parent_code,
            "parent_description": parent_desc,
            "grouping_criterion": consolidation.grouping_criterion,
            "reasoning": consolidation.reasoning,
            "subcategories": clusters,
        }

        async with write_lock:
            consolidated_by_parent[parent_key] = entry
            if not sample_mode:
                save_consolidated(consolidated_path, consolidated_by_parent)
            consolidation_done[0] += 1
            print(f"[{consolidation_done[0]}/{consolidation_total}] Consolidated {parent_key}")

        return entry

    consolidation_tasks = [
        process_consolidation(parent_key, runs)
        for parent_key, runs in runs_by_parent.items()
    ]
    consolidation_results = await asyncio.gather(*consolidation_tasks, return_exceptions=True)

    consolidation_errors = [r for r in consolidation_results if isinstance(r, Exception)]
    if consolidation_errors:
        print(f"\n{len(consolidation_errors)} consolidations failed:")
        for err in consolidation_errors[:5]:
            print(f"  - {err}")

    # Build initial tree
    metas_out = []
    for meta, info in taxonomy["categories_by_meta"].items():
        roots = []
        nodes = []
        used_codes = set()

        # Top-level categories (roots)
        for cat in info.get("categories", []):
            code = cat.get("code", cat.get("name", "unknown"))
            tree_code = canonical_root_code(code)
            desc = cat.get("description", "")
            if tree_code in used_codes:
                print(f"Duplicate top-level code in meta {meta}: {tree_code}. Skipping.")
                continue
            used_codes.add(tree_code)
            roots.append(tree_code)

            key = f"{meta}:{code}"
            subcats = consolidated_by_parent.get(key, {}).get("subcategories", [])
            child_codes = []

            # Add subcategory nodes (and collect accepted children)
            for sc in subcats:
                if sc["code"] in used_codes:
                    print(f"Duplicate subcategory code in meta {meta}: {sc['code']}. Skipping.")
                    continue
                used_codes.add(sc["code"])
                child_codes.append(sc["code"])
                nodes.append({
                    "code": sc["code"],
                    "description": sc.get("description", ""),
                    "children": [],
                })

            nodes.append({
                "code": tree_code,
                "description": desc,
                "children": child_codes,
            })

        metas_out.append({
            "meta": meta,
            "roots": roots,
            "nodes": nodes,
        })

    if sample_mode:
        print("Sample mode - not saving results")
        return consolidated_by_parent

    # Save consolidated subcategories
    save_consolidated(consolidated_path, consolidated_by_parent)
    print(f"Saved: {consolidated_path}")

    # Save initial tree
    output_tree = provider_scoped_path(INTERMEDIATE_DIR / "taxonomy_tree_initial.json", model)
    with open(output_tree, "w") as f:
        json.dump(
            {
                "domain_context": domain_context,
                "target": target,
                "model": model,
                "max_children": max_children,
                "metas": metas_out,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Saved: {output_tree}")
    if discovery_errors or consolidation_errors:
        raise RuntimeError(
            "Phase 07 incomplete: "
            f"{len(discovery_errors)} discovery runs failed, {len(consolidation_errors)} consolidations failed."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 07: Subcategory Discovery + Consolidation")
    parser.add_argument("--sample", action="store_true", help="Sample mode (no save)")
    parser.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES, help="Samples per discovery run")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS, help="Random seeds")
    parser.add_argument("--max-children", type=int, default=DEFAULT_MAX_CHILDREN, help="Max subcategories")
    parser.add_argument("--concurrency", type=int, default=None, help="Parallel API calls (defaults: Qwen 27B=8, Qwen 9B=20, otherwise 10; local=1)")
    parser.add_argument("--domain-context", type=str, default=DEFAULT_DOMAIN_CONTEXT, help="Domain context")
    parser.add_argument("--target", type=str, default=DEFAULT_TARGET, help="Target concept")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model to use")
    parser.add_argument("--taxonomy-model", type=str, default=DEFAULT_TAXONOMY_MODEL, help="Model used to generate the Phase 00-03 taxonomy")
    parser.add_argument("--labeling-model", type=str, default=DEFAULT_LABELING_MODEL, help="Model used for Phase 04 labeling")
    parser.add_argument("--resume", action="store_true", help="Reuse existing per-seed subcategory discovery runs")
    parser.add_argument("--request-timeout-seconds", type=float, default=DEFAULT_REQUEST_TIMEOUT_SECONDS, help="Per-attempt timeout for a single request")
    parser.add_argument("--system-prompt", type=str, default=DEFAULT_SYSTEM_PROMPT, help="System prompt for Phase 07 calls")
    parser.add_argument("--base-url", type=str, default=None, help="Custom API base URL (e.g. LM Studio)")
    parser.add_argument("--run-id", type=str, default=None, help="Optional run id for isolated artifacts under data/runs/<run_id>/")

    args = parser.parse_args()
    asyncio.run(main(
        sample_mode=args.sample,
        n_samples=args.n_samples,
        seeds=args.seeds,
        max_children=args.max_children,
        concurrency=args.concurrency,
        domain_context=args.domain_context,
        target=args.target,
        model=args.model,
        taxonomy_model=args.taxonomy_model,
        labeling_model=args.labeling_model,
        resume=args.resume,
        request_timeout_seconds=args.request_timeout_seconds,
        system_prompt=args.system_prompt,
        base_url=args.base_url,
        run_id=args.run_id,
    ))
