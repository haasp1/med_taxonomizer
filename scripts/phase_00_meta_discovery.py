"""
Phase 00: Meta-Category Discovery (Ensemble)

Run meta-category discovery multiple times with different random seeds.
Each run is saved immediately for robustness and later consolidation.

Usage:
    uv run python scripts/phase_00_meta_discovery.py --sample
    uv run python scripts/phase_00_meta_discovery.py
    uv run python scripts/phase_00_meta_discovery.py --seeds 21 42 84
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
from lib.llm_client import make_openai_client, local_structured_call, extract_pydantic_json, resolve_concurrency, structured_parse_call
from lib.model_naming import provider_scoped_path, resolve_temperature

load_dotenv()


# --- Configuration ---

EXPERIMENT_DIR = Path(__file__).parent
PROMPTS_DIR = EXPERIMENT_DIR / "prompts"
ARTIFACT_PATHS = resolve_experiment_artifact_paths(EXPERIMENT_DIR)
DATA_DIR = ARTIFACT_PATHS.data_dir
INTERMEDIATE_DIR = ARTIFACT_PATHS.intermediate_dir
INPUT_DATA = Path("data/inputs/texts.parquet")

DEFAULT_MODEL = "openai/gpt-5.2"
DEFAULT_TARGET = "adverse_events"
DEFAULT_DOMAIN_CONTEXT = "Free-text entries from stroke intervention documentation."
DEFAULT_META_GUIDANCE = "none"
DEFAULT_N_SAMPLES = 200
DEFAULT_SEEDS = [42, 123, 456, 789, 2024, 4096]
DEFAULT_CONCURRENCY = 10
DEFAULT_REQUEST_TIMEOUT_SECONDS = 180.0


def configure_paths(run_id: str | None) -> None:
    global ARTIFACT_PATHS, DATA_DIR, INTERMEDIATE_DIR
    ARTIFACT_PATHS = resolve_experiment_artifact_paths(EXPERIMENT_DIR, run_id=run_id)
    DATA_DIR = ARTIFACT_PATHS.data_dir
    INTERMEDIATE_DIR = ARTIFACT_PATHS.intermediate_dir


# --- Pydantic Models ---

class MetaCategory(BaseModel):
    code: str = Field(description="Short code in snake_case (English)")
    description: str = Field(description="Brief description of this meta-category")
    is_target: bool = Field(description="True if this meta-category represents the target")
    example_indices: list[int] = Field(
        default_factory=list,
        description="Indices of example samples (from the numbered list)"
    )


class MetaDiscoveryResult(BaseModel):
    meta_categories: list[MetaCategory]
    reasoning: str = Field(description="Brief explanation of how these categories were identified")


# --- Client Setup ---

def get_client(base_url: str | None = None) -> tuple[AsyncOpenAI, bool]:
    client, is_local = make_openai_client(base_url=base_url, async_client=True)
    return client, is_local


def load_prompt() -> str:
    prompt_file = PROMPTS_DIR / "phase_00_meta_discovery.md"
    return prompt_file.read_text()


def format_samples(texts: list[str]) -> str:
    return "\n".join(f"{i+1}. {text}" for i, text in enumerate(texts))


async def run_discovery(
    client: AsyncOpenAI,
    texts: list[str],
    target: str,
    domain_context: str,
    meta_guidance: str,
    model: str,
    semaphore: asyncio.Semaphore,
    local: bool = False,
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
) -> MetaDiscoveryResult:
    async with semaphore:
        prompt_template = load_prompt()
        prompt = (
            prompt_template
            .replace("{target}", target)
            .replace("{domain_context}", domain_context)
            .replace("{meta_guidance}", meta_guidance)
            .replace("{samples}", format_samples(texts))
        )

        if local:
            return await local_structured_call(client, model, prompt, MetaDiscoveryResult)
        else:
            return await structured_parse_call(
                client,
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format=MetaDiscoveryResult,
                extra_body={
                    "provider": {"zdr": True},
                    "reasoning": {"effort": "high"},
                },
                temperature=resolve_temperature(model, 0.6),
                request_timeout_seconds=request_timeout_seconds,
            )


def save_run(run_data: dict, output_dir: Path, seed: int, model: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = get_run_path(output_dir, seed, model)
    with open(output_file, "w") as f:
        json.dump(run_data, f, indent=2, ensure_ascii=False)
    print(f"Saved: {output_file}")


def get_run_path(output_dir: Path, seed: int, model: str) -> Path:
    return provider_scoped_path(output_dir / f"meta_seed_{seed}.json", model)


async def main(
    sample_mode: bool = False,
    input_data: Path = INPUT_DATA,
    n_samples: int = DEFAULT_N_SAMPLES,
    seeds: list[int] = None,
    target: str = DEFAULT_TARGET,
    domain_context: str = DEFAULT_DOMAIN_CONTEXT,
    meta_guidance: str = DEFAULT_META_GUIDANCE,
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
    output_dir = INTERMEDIATE_DIR / "meta_runs"

    print(f"Loading data from {input_data}...")
    df = pd.read_parquet(input_data)
    if "text" not in df.columns:
        raise ValueError("Input parquet must contain a 'text' column")
    print(f"Loaded {len(df)} total samples")

    client, local = get_client(base_url=base_url)
    concurrency = resolve_concurrency(concurrency, local, model=model)
    semaphore = asyncio.Semaphore(concurrency)
    io_lock = asyncio.Lock()
    print(f"Concurrency: {concurrency} parallel requests")

    seed_samples: list[tuple[int, list[str]]] = []
    resumed_runs = 0
    for seed in seeds:
        output_path = get_run_path(output_dir, seed, model)
        if resume and output_path.exists() and not sample_mode:
            resumed_runs += 1
            seed_samples.append((seed, []))
            continue
        df_sample = df.sample(n=min(n_samples, len(df)), random_state=seed)
        texts = df_sample["text"].tolist()
        seed_samples.append((seed, texts))

    completed = [0]
    total = len(seed_samples)
    if resume and resumed_runs:
        print(f"Resume enabled: reusing {resumed_runs} saved meta discovery runs")

    async def process_seed(seed: int, texts: list[str]) -> None:
        try:
            if resume and not texts and not sample_mode:
                async with io_lock:
                    completed[0] += 1
                    print(f"  [{completed[0]}/{total}] Seed {seed} already complete")
                return
            async with io_lock:
                print(f"\nSeed {seed}: running meta discovery on {len(texts)} samples...")

            result = await run_discovery(
                client, texts, target, domain_context, meta_guidance, model, semaphore,
                local=local,
                request_timeout_seconds=request_timeout_seconds,
            )

            run_data = {
                "seed": seed,
                "target": target,
                "domain_context": domain_context,
                "meta_guidance": meta_guidance,
                "n_samples": len(texts),
                "model": model,
                "meta_categories": [mc.model_dump() for mc in result.meta_categories],
                "reasoning": result.reasoning,
            }

            async with io_lock:
                print(f"  Found {len(result.meta_categories)} meta-categories")
                for mc in result.meta_categories:
                    marker = " [TARGET]" if mc.is_target else ""
                    print(f"    - {mc.code}{marker}")

                if sample_mode:
                    print("  Sample mode - not saving results")
                else:
                    save_run(run_data, output_dir, seed, model)

                completed[0] += 1
                print(f"  [{completed[0]}/{total}] Seed {seed} complete")
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Seed {seed} failed: {exc}") from exc

    tasks = [process_seed(seed, texts) for seed, texts in seed_samples]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    errors = [r for r in results if isinstance(r, Exception)]
    if errors:
        print(f"\n{len(errors)} seeds failed:")
        for err in errors[:5]:
            print(f"  - {err}")
        raise RuntimeError(f"Phase 00 incomplete: {len(errors)} seeds failed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 00: Meta-Category Discovery (Ensemble)")
    parser.add_argument("--sample", action="store_true", help="Run with fewer samples and seeds")
    parser.add_argument("--input", type=Path, default=INPUT_DATA, help="Parquet file with a text column")
    parser.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES, help="Number of samples")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS, help="Random seeds")
    parser.add_argument("--target", type=str, default=DEFAULT_TARGET, help="Target description")
    parser.add_argument("--domain-context", type=str, default=DEFAULT_DOMAIN_CONTEXT, help="Domain context")
    parser.add_argument("--meta-guidance", type=str, default=DEFAULT_META_GUIDANCE, help="Optional meta guidance")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model to use")
    parser.add_argument("--concurrency", type=int, default=None, help="Parallel API calls (defaults: Qwen 27B=8, Qwen 9B=20, otherwise 10; local=1)")
    parser.add_argument("--resume", action="store_true", help="Reuse existing per-seed meta discovery runs")
    parser.add_argument("--base-url", type=str, default=None, help="Custom API base URL (e.g. http://192.168.0.70:1234/v1 for LM Studio)")
    parser.add_argument("--run-id", type=str, default=None, help="Optional run id for isolated artifacts under data/runs/<run_id>/")
    parser.add_argument("--request-timeout-seconds", type=float, default=DEFAULT_REQUEST_TIMEOUT_SECONDS, help="Per-attempt timeout for a single request")

    args = parser.parse_args()
    asyncio.run(main(
        sample_mode=args.sample,
        input_data=args.input,
        n_samples=args.n_samples,
        seeds=args.seeds,
        target=args.target,
        domain_context=args.domain_context,
        meta_guidance=args.meta_guidance,
        model=args.model,
        concurrency=args.concurrency,
        resume=args.resume,
        base_url=args.base_url,
        run_id=args.run_id,
        request_timeout_seconds=args.request_timeout_seconds,
    ))
