"""
Phase 02: Category Discovery (Ensemble)

Run category discovery multiple times with different seeds for each meta-category.
Each run is saved for later consolidation.

Usage:
    uv run python scripts/phase_02_category_discovery.py --sample
    uv run python scripts/phase_02_category_discovery.py
    uv run python scripts/phase_02_category_discovery.py --seeds 21 42 84
"""

import argparse
import asyncio
import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from lib.experiment_paths import resolve_experiment_artifact_paths
from lib.llm_client import make_openai_client, local_structured_call, local_structured_call_sync, extract_pydantic_json, resolve_concurrency, structured_parse_call
from lib.model_naming import provider_scoped_path, resolve_provider_path, resolve_temperature

load_dotenv()

# --- Configuration ---

EXPERIMENT_DIR = Path(__file__).parent
PROMPTS_DIR = EXPERIMENT_DIR / "prompts"
ARTIFACT_PATHS = resolve_experiment_artifact_paths(EXPERIMENT_DIR)
DATA_DIR = ARTIFACT_PATHS.data_dir
INTERMEDIATE_DIR = ARTIFACT_PATHS.intermediate_dir
INPUT_DATA = Path("experiments/01_data_preparation/data/texts_sufficient_only.parquet")

DEFAULT_MODEL = "openai/gpt-5.2"
DEFAULT_N_SAMPLES = 200
DEFAULT_SEEDS = [42, 123, 456]
DEFAULT_DOMAIN_CONTEXT = "Free-text entries from stroke intervention documentation."
DEFAULT_CONCURRENCY = 10
DEFAULT_REQUEST_TIMEOUT_SECONDS = 180.0


def configure_paths(run_id: str | None) -> None:
    global ARTIFACT_PATHS, DATA_DIR, INTERMEDIATE_DIR
    ARTIFACT_PATHS = resolve_experiment_artifact_paths(EXPERIMENT_DIR, run_id=run_id)
    DATA_DIR = ARTIFACT_PATHS.data_dir
    INTERMEDIATE_DIR = ARTIFACT_PATHS.intermediate_dir


# --- Pydantic Models ---

class Category(BaseModel):
    code: str = Field(description="Short name in snake_case (English)")
    description: str = Field(description="Brief description of this category")


class CategoryDiscoveryResult(BaseModel):
    meta_category: str = Field(description="The meta-category being analyzed")
    grouping_criterion: str = Field(description="The criterion used to group categories")
    categories: list[Category]
    reasoning: str = Field(description="Why this grouping criterion makes sense")


def get_client(base_url: str | None = None):
    return make_openai_client(base_url=base_url, async_client=True)


def load_prompt() -> str:
    prompt_file = PROMPTS_DIR / "phase_02_category_discovery.md"
    return prompt_file.read_text()


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


def format_samples(texts: list[str]) -> str:
    return "\n".join(f"{i+1}. {text}" for i, text in enumerate(texts))


async def run_discovery_for_meta(
    client: AsyncOpenAI,
    texts: list[str],
    meta_code: str,
    meta_description: str,
    domain_context: str,
    model: str,
    semaphore: asyncio.Semaphore,
    local: bool = False,
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
) -> CategoryDiscoveryResult:
    async with semaphore:
        prompt_template = load_prompt()
        prompt = (
            prompt_template
            .replace("{domain_context}", domain_context)
            .replace("{meta_category}", meta_code)
            .replace("{meta_description}", meta_description)
            .replace("{samples}", format_samples(texts))
        )

        if local:
            return await local_structured_call(client, model, prompt, CategoryDiscoveryResult)

        return await structured_parse_call(
            client,
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format=CategoryDiscoveryResult,
            extra_body={
                "provider": {"zdr": True},
                "reasoning": {"effort": "high"},
            },
            temperature=resolve_temperature(model, 0.2),
            request_timeout_seconds=request_timeout_seconds,
        )


def save_run(run_data: dict, output_dir: Path, meta_code: str, seed: int, model: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = get_run_path(output_dir, meta_code, seed, model)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(run_data, f, indent=2, ensure_ascii=False)
    print(f"Saved: {output_file}")


def get_run_path(output_dir: Path, meta_code: str, seed: int, model: str) -> Path:
    return provider_scoped_path(output_dir / meta_code / f"cat_seed_{seed}.json", model)


async def main(
    sample_mode: bool = False,
    n_samples: int = DEFAULT_N_SAMPLES,
    seeds: list[int] = None,
    domain_context: str = DEFAULT_DOMAIN_CONTEXT,
    model: str = DEFAULT_MODEL,
    concurrency: int | None = None,
    resume: bool = False,
    base_url: str | None = None,
    run_id: str | None = None,
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
):
    configure_paths(run_id)
    if seeds is None:
        seeds = DEFAULT_SEEDS

    if sample_mode:
        n_samples = 40
        seeds = seeds[:2]
        print(f"=== SAMPLE MODE: {n_samples} samples, seeds {seeds} ===")

    print(f"Artifact scope: {DATA_DIR}")
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    output_dir = INTERMEDIATE_DIR / "category_runs"

    meta_data = load_meta_categories(model)
    meta_categories = meta_data.get("meta_categories", [])
    if isinstance(meta_categories, list) and meta_categories and isinstance(meta_categories[0], str):
        meta_categories = [{"code": m, "description": ""} for m in meta_categories]

    print(f"Loading data from {INPUT_DATA}...")
    df = pd.read_parquet(INPUT_DATA)
    print(f"Loaded {len(df)} total samples")

    client, local = get_client(base_url=base_url)
    concurrency = resolve_concurrency(concurrency, local, model=model)
    semaphore = asyncio.Semaphore(concurrency)
    io_lock = asyncio.Lock()
    print(f"Concurrency: {concurrency} parallel requests")

    seed_samples: list[tuple[int, list[str]]] = []
    for seed in seeds:
        df_sample = df.sample(n=min(n_samples, len(df)), random_state=seed)
        texts = df_sample["text"].tolist()
        seed_samples.append((seed, texts))

    total_tasks = len(seed_samples) * len(meta_categories)
    completed = [0]
    resumed_runs = 0

    async def process_meta(seed: int, texts: list[str], meta_code: str, meta_desc: str) -> None:
        try:
            output_path = get_run_path(output_dir, meta_code, seed, model)
            if resume and output_path.exists() and not sample_mode:
                async with io_lock:
                    completed[0] += 1
                    print(f"  [{completed[0]}/{total_tasks}] {meta_code} already complete for seed {seed}")
                return
            if meta_code == "other":
                run_data = {
                    "seed": seed,
                    "meta_category": meta_code,
                    "meta_description": meta_desc,
                    "n_samples": len(texts),
                    "model": model,
                    "domain_context": domain_context,
                    "grouping_criterion": "none",
                    "reasoning": "Catch-all meta-category; no subcategories required.",
                    "categories": [{"code": "other", "description": "Catch-all for unclassifiable samples"}],
                }
            else:
                result = await run_discovery_for_meta(
                    client, texts, meta_code, meta_desc, domain_context, model, semaphore, local=local,
                    request_timeout_seconds=request_timeout_seconds,
                )
                run_data = {
                    "seed": seed,
                    "meta_category": meta_code,
                    "meta_description": meta_desc,
                    "n_samples": len(texts),
                    "model": model,
                    "domain_context": domain_context,
                    "grouping_criterion": result.grouping_criterion,
                    "reasoning": result.reasoning,
                    "categories": [c.model_dump() for c in result.categories],
                }

            async with io_lock:
                if meta_code == "other":
                    print(f"  Seed {seed} {meta_code}: 1 category (default)")
                else:
                    print(f"  Seed {seed} {meta_code}: {len(run_data['categories'])} categories")

                if not sample_mode:
                    save_run(run_data, output_dir, meta_code, seed, model)

                completed[0] += 1
                print(f"  [{completed[0]}/{total_tasks}] {meta_code} complete for seed {seed}")
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Seed {seed} meta {meta_code} failed: {exc}") from exc

    tasks = []
    if resume and not sample_mode:
        for seed, _texts in seed_samples:
            for mc in meta_categories:
                output_path = get_run_path(output_dir, mc["code"], seed, model)
                if output_path.exists():
                    resumed_runs += 1
        if resumed_runs:
            print(f"Resume enabled: reusing {resumed_runs} saved category discovery runs")
    for seed, texts in seed_samples:
        for mc in meta_categories:
            meta_code = mc["code"]
            meta_desc = mc.get("description", "")
            tasks.append(process_meta(seed, texts, meta_code, meta_desc))

    results = await asyncio.gather(*tasks, return_exceptions=True)
    errors = [r for r in results if isinstance(r, Exception)]
    if errors:
        print(f"\n{len(errors)} meta-category runs failed:")
        for err in errors[:5]:
            print(f"  - {err}")
        raise RuntimeError(f"Phase 02 incomplete: {len(errors)} meta-category runs failed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 02: Category Discovery (Ensemble)")
    parser.add_argument("--sample", action="store_true", help="Run with fewer samples and seeds")
    parser.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES, help="Number of samples")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS, help="Random seeds")
    parser.add_argument("--domain-context", type=str, default=DEFAULT_DOMAIN_CONTEXT, help="Domain context")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model to use")
    parser.add_argument("--concurrency", type=int, default=None, help="Parallel API calls (defaults: Qwen 27B=8, Qwen 9B=20, otherwise 10; local=1)")
    parser.add_argument("--resume", action="store_true", help="Reuse existing per-meta category discovery runs")
    parser.add_argument("--base-url", type=str, default=None, help="Custom API base URL (e.g. LM Studio)")
    parser.add_argument("--run-id", type=str, default=None, help="Optional run id for isolated artifacts under data/runs/<run_id>/")
    parser.add_argument("--request-timeout-seconds", type=float, default=DEFAULT_REQUEST_TIMEOUT_SECONDS, help="Per-attempt timeout for a single request")

    args = parser.parse_args()
    asyncio.run(main(
        sample_mode=args.sample,
        n_samples=args.n_samples,
        seeds=args.seeds,
        domain_context=args.domain_context,
        model=args.model,
        concurrency=args.concurrency,
        resume=args.resume,
        base_url=args.base_url,
        run_id=args.run_id,
        request_timeout_seconds=args.request_timeout_seconds,
    ))
