"""
Phase 06: Split Audit

Decide which categories are broad enough to split into subcategories.
Runs multiple audits per category (different seeds) and summarizes votes.

Usage:
    uv run python scripts/phase_06_split_audit.py --sample
    uv run python scripts/phase_06_split_audit.py
    uv run python scripts/phase_06_split_audit.py --min-labels 200 --seeds 21 42 84
"""

import argparse
import asyncio
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from lib.experiment_paths import resolve_experiment_artifact_paths
from lib.llm_client import (
    append_validation_feedback_messages,
    append_validation_feedback_prompt,
    local_structured_call,
    make_openai_client,
    resolve_concurrency,
    run_with_validation_repair,
    structured_parse_call,
)
from lib.model_naming import provider_scoped_path, resolve_provider_path, resolve_temperature

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
DEFAULT_MIN_LABELS = None
DEFAULT_MIN_LABEL_FRACTION = 0.02
DEFAULT_N_SAMPLES = 50
DEFAULT_SEEDS = [42, 123, 456]
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


class SplitAuditResult(BaseModel):
    parent_code: str
    should_split: bool
    reasoning: str
    suggested_grouping_criterion: str | None = None


def get_client(base_url: str | None = None):
    return make_openai_client(base_url=base_url, async_client=True)


def load_prompt() -> str:
    prompt_file = PROMPTS_DIR / "phase_06_split_audit.md"
    return prompt_file.read_text()


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

    # Fallback to JSONL
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


def get_generation_kwargs(model: str, *, local: bool, default_temperature: float) -> dict:
    if "qwen" in (model or "").lower():
        extra_body: dict[str, object] = {"top_k": QWEN_TOP_K}
        if not local:
            extra_body["provider"] = {"zdr": True}
        return {
            "temperature": QWEN_TEMPERATURE,
            "top_p": QWEN_TOP_P,
            "presence_penalty": QWEN_PRESENCE_PENALTY,
            "extra_body": extra_body,
        }

    kwargs: dict[str, object] = {"temperature": resolve_temperature(model, default_temperature)}
    if not local:
        kwargs["extra_body"] = {"provider": {"zdr": True}}
    return kwargs


async def run_audit(
    client: AsyncOpenAI,
    texts: list[str],
    parent_code: str,
    parent_description: str,
    domain_context: str,
    target: str,
    model: str,
    semaphore: asyncio.Semaphore,
    local: bool = False,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    max_attempts: int = DEFAULT_BATCH_MAX_ATTEMPTS,
) -> SplitAuditResult:
    prompt_template = load_prompt()
    prompt = (
        prompt_template
        .replace("{domain_context}", domain_context)
        .replace("{target}", target)
        .replace("{parent_code}", parent_code)
        .replace("{parent_description}", parent_description)
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
                    SplitAuditResult,
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
                    response_format=SplitAuditResult,
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
        operation_label=f"split audit {parent_code}",
        run_attempt=run_attempt,
        max_attempts=max_attempts,
    )


def get_run_path(output_dir: Path, meta: str, parent_code: str, seed: int, model: str) -> Path:
    parent_dir = output_dir / meta
    return provider_scoped_path(parent_dir / f"{parent_code}_seed_{seed}.json", model)


def save_run(output_dir: Path, meta: str, parent_code: str, seed: int, data: dict, model: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = get_run_path(output_dir, meta, parent_code, seed, model)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved: {output_file}")


async def main(
    sample_mode: bool = False,
    min_labels: int | None = DEFAULT_MIN_LABELS,
    n_samples: int = DEFAULT_N_SAMPLES,
    seeds: list[int] = None,
    domain_context: str = DEFAULT_DOMAIN_CONTEXT,
    target: str = DEFAULT_TARGET,
    model: str = DEFAULT_MODEL,
    taxonomy_model: str = DEFAULT_TAXONOMY_MODEL,
    labeling_model: str = DEFAULT_LABELING_MODEL,
    concurrency: int | None = None,
    resume: bool = False,
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    base_url: str | None = None,
    run_id: str | None = None,
):
    if seeds is None:
        seeds = DEFAULT_SEEDS

    configure_paths(run_id)
    print(f"Artifact scope: {DATA_DIR}")
    taxonomy = load_taxonomy(taxonomy_model)
    texts_by_cat = load_texts_by_category(labeling_model)

    # Build category list with counts and descriptions
    candidates = []
    for meta, info in taxonomy["categories_by_meta"].items():
        for cat in info.get("categories", []):
            code = cat.get("code", cat.get("name", "unknown"))
            desc = cat.get("description", "")
            count = len(texts_by_cat.get((meta, code), []))
            candidates.append((meta, code, desc, count))

    total_label_count = sum(len(v) for v in texts_by_cat.values())
    if min_labels is None:
        # Public-tool adaptation requested by Hannah: replace the original absolute
        # 200-label split gate with a relative 2% gate. Everything else remains
        # copied taxonomy phase logic.
        min_labels = max(1, int(total_label_count * DEFAULT_MIN_LABEL_FRACTION + 0.999999))
    print(f"Split audit min_labels={min_labels} ({DEFAULT_MIN_LABEL_FRACTION:.0%} default if not explicitly set; total_label_count={total_label_count})")

    # Filter by min_labels
    candidates = [c for c in candidates if c[3] >= min_labels and c[1] != "other"]
    candidates.sort(key=lambda x: -x[3])

    if sample_mode:
        candidates = candidates[:2]
        seeds = seeds[:2]
        n_samples = min(n_samples, 20)
        print(f"=== SAMPLE MODE: {len(candidates)} categories, seeds {seeds} ===")

    output_dir = INTERMEDIATE_DIR / "split_audit_runs"
    summary = {}
    client, local = get_client(base_url=base_url)
    concurrency = resolve_concurrency(concurrency, local, model=model)
    semaphore = asyncio.Semaphore(concurrency)
    print(f"Concurrency: {concurrency} parallel requests")

    jobs = []
    contexts = {}
    resumed_runs = 0
    for meta, code, desc, count in candidates:
        texts = texts_by_cat.get((meta, code), [])
        if not texts:
            continue
        contexts[f"{meta}:{code}"] = {
            "meta": meta,
            "parent_code": code,
            "parent_description": desc,
            "label_count": count,
        }
        df = pd.DataFrame({"text": texts})
        for seed in seeds:
            output_path = get_run_path(output_dir, meta, code, seed, model)
            if resume and output_path.exists():
                with open(output_path) as f:
                    run_data = json.load(f)
                jobs.append(
                    {
                        "resume_data": run_data,
                        "meta": meta,
                        "code": code,
                        "seed": seed,
                    }
                )
                resumed_runs += 1
                continue
            sample_df = df.sample(n=min(n_samples, len(df)), random_state=seed)
            jobs.append(
                {
                    "meta": meta,
                    "code": code,
                    "desc": desc,
                    "count": count,
                    "seed": seed,
                    "sample_texts": sample_df["text"].tolist(),
                }
            )

    if resume and resumed_runs:
        print(f"Resume enabled: reusing {resumed_runs} saved split audit runs")

    async def process_job(job: dict[str, Any]):
        if "resume_data" in job:
            run_data = job["resume_data"]
            print(
                f"Resume {job['meta']}:{job['code']} seed {job['seed']} "
                f"should_split={run_data['should_split']}"
            )
            return run_data
        try:
            result = await run_audit(
                client,
                job["sample_texts"],
                parent_code=job["code"],
                parent_description=job["desc"],
                domain_context=domain_context,
                target=target,
                model=model,
                semaphore=semaphore,
                local=local,
                system_prompt=system_prompt,
                request_timeout_seconds=request_timeout_seconds,
            )
            run_data = {
                "seed": job["seed"],
                "meta": job["meta"],
                "parent_code": job["code"],
                "parent_description": job["desc"],
                "n_samples": len(job["sample_texts"]),
                "should_split": result.should_split,
                "reasoning": result.reasoning,
                "suggested_grouping_criterion": result.suggested_grouping_criterion,
            }
            if not sample_mode:
                save_run(output_dir, job["meta"], job["code"], job["seed"], run_data, model)
            print(
                f"Audit {job['meta']}:{job['code']} seed {job['seed']} "
                f"should_split={result.should_split}"
            )
            return run_data
        except Exception as exc:
            return RuntimeError(
                f"Split audit failed for {job['meta']}:{job['code']} seed {job['seed']}: {exc}"
            )

    results = await asyncio.gather(*[process_job(job) for job in jobs], return_exceptions=True)
    grouped_runs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    errors = [result for result in results if isinstance(result, Exception)]
    for result in results:
        if isinstance(result, Exception):
            continue
        grouped_runs[f"{result['meta']}:{result['parent_code']}"].append(result)

    for summary_key, runs in grouped_runs.items():
        context = contexts[summary_key]
        votes = [run["should_split"] for run in runs]
        criteria = [run["suggested_grouping_criterion"] for run in runs if run["suggested_grouping_criterion"]]
        vote_counts = Counter(votes)
        should_split = vote_counts[True] >= (len(votes) // 2 + 1)
        criterion = Counter(criteria).most_common(1)[0][0] if criteria else None
        summary[summary_key] = {
            "meta": context["meta"],
            "parent_code": context["parent_code"],
            "parent_description": context["parent_description"],
            "label_count": context["label_count"],
            "votes": dict(vote_counts),
            "should_split": should_split,
            "suggested_grouping_criterion": criterion,
        }

    if errors:
        print(f"\n{len(errors)} split audit runs failed:")
        for err in errors[:5]:
            print(f"  - {err}")

    output_summary = provider_scoped_path(INTERMEDIATE_DIR / "split_audit_summary.json", model)
    if sample_mode:
        print("Sample mode - not saving summary")
        return summary

    with open(output_summary, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved summary: {output_summary}")
    if errors:
        raise RuntimeError(f"Phase 06 incomplete: {len(errors)} split audit runs failed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 06: Split Audit")
    parser.add_argument("--sample", action="store_true", help="Sample mode (no save)")
    parser.add_argument("--min-labels", type=int, default=DEFAULT_MIN_LABELS, help="Min labels for audit; default is ceil(2%% of current labels)")
    parser.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES, help="Samples per audit")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS, help="Random seeds")
    parser.add_argument("--domain-context", type=str, default=DEFAULT_DOMAIN_CONTEXT, help="Domain context")
    parser.add_argument("--target", type=str, default=DEFAULT_TARGET, help="Target concept")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model to use")
    parser.add_argument("--taxonomy-model", type=str, default=DEFAULT_TAXONOMY_MODEL, help="Model used to generate the Phase 00-03 taxonomy")
    parser.add_argument("--labeling-model", type=str, default=DEFAULT_LABELING_MODEL, help="Model used for Phase 04 labeling")
    parser.add_argument("--concurrency", type=int, default=None, help="Parallel API calls (defaults: Qwen 27B=8, Qwen 9B=20, otherwise 10; local=1)")
    parser.add_argument("--resume", action="store_true", help="Reuse existing per-seed split audit runs")
    parser.add_argument("--request-timeout-seconds", type=float, default=DEFAULT_REQUEST_TIMEOUT_SECONDS, help="Per-attempt timeout for a single request")
    parser.add_argument("--system-prompt", type=str, default=DEFAULT_SYSTEM_PROMPT, help="System prompt for split audit calls")
    parser.add_argument("--base-url", type=str, default=None, help="Custom API base URL (e.g. LM Studio)")
    parser.add_argument("--run-id", type=str, default=None, help="Optional run id for isolated artifacts under data/runs/<run_id>/")

    args = parser.parse_args()
    asyncio.run(main(
        sample_mode=args.sample,
        min_labels=args.min_labels,
        n_samples=args.n_samples,
        seeds=args.seeds,
        domain_context=args.domain_context,
        target=args.target,
        model=args.model,
        taxonomy_model=args.taxonomy_model,
        labeling_model=args.labeling_model,
        concurrency=args.concurrency,
        resume=args.resume,
        request_timeout_seconds=args.request_timeout_seconds,
        system_prompt=args.system_prompt,
        base_url=args.base_url,
        run_id=args.run_id,
    ))
