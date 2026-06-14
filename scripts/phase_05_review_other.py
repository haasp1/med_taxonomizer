"""
Phase 05: Heal category-level 'other' labels in place.

Reviews labels currently assigned to meta:other (excluding other:other),
proposes missing flat categories within the same meta, relabels only those
existing other rows, and updates taxonomy + cached outputs without rerunning
Phase 04 end-to-end.

Usage:
    uv run python scripts/phase_05_review_other.py --sample --analyze-only
    uv run python scripts/phase_05_review_other.py --model qwen/qwen3.5-27b --taxonomy-model qwen/qwen3.5-27b --labeling-model qwen/qwen3.5-27b
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import re
import sqlite3
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, create_model

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
DEFAULT_MIN_OTHER_COUNT = 20
DEFAULT_MIN_OTHER_RATE = 0.10
DEFAULT_N_SAMPLES = 50
DEFAULT_SEEDS = [42, 123, 456]
DEFAULT_MAX_NEW_CATEGORIES = 4
DEFAULT_MIN_CATEGORY_SUPPORT = 2
DEFAULT_MIN_ESTIMATED_COUNT = 10
DEFAULT_MIN_APPLIED_COUNT = 3
DEFAULT_RELABEL_BATCH_SIZE = 25
DEFAULT_CONCURRENCY = 10
DEFAULT_DOMAIN_CONTEXT = "Free-text entries from stroke intervention documentation."
DEFAULT_TARGET = "adverse_events"
DEFAULT_REQUEST_TIMEOUT_SECONDS = 180.0
DEFAULT_BATCH_MAX_ATTEMPTS = 4
DEFAULT_SYSTEM_PROMPT = (
    "You are a concise, efficient, decisive assistant. "
    "Think in 2-3 short blocks per sample without repetition or second-guessing, "
    "and then output your answer."
)
QWEN_TEMPERATURE = 0.6
QWEN_TOP_P = 0.95
QWEN_TOP_K = 20
QWEN_PRESENCE_PENALTY = 1.5


def configure_paths(run_id: str | None) -> None:
    global ARTIFACT_PATHS, DATA_DIR, INTERMEDIATE_DIR, CACHE_DIR
    ARTIFACT_PATHS = resolve_experiment_artifact_paths(EXPERIMENT_DIR, run_id=run_id)
    DATA_DIR = ARTIFACT_PATHS.data_dir
    INTERMEDIATE_DIR = ARTIFACT_PATHS.intermediate_dir
    CACHE_DIR = ARTIFACT_PATHS.cache_dir


class ProposedCategory(BaseModel):
    code: str = Field(description="Short category code in snake_case")
    description: str = Field(description="Brief description")
    estimated_count: int = Field(description="Estimated number of current meta:other rows that fit")


class CategoryProposalRun(BaseModel):
    meta: str = Field(description="Meta-category under review")
    summary: str = Field(description="Summary of the major patterns in the other bucket")
    proposed_categories: list[ProposedCategory]
    remaining_other_estimate: int = Field(description="Estimated number of rows that should remain other")


class CategoryProposalCluster(BaseModel):
    code: str = Field(description="Canonical code in snake_case")
    description: str = Field(description="Brief description")
    members: list[str] = Field(description="Run-specific members like 'seed_42:cardiac_arrhythmia'")
    estimated_count: int = Field(description="Estimated number of current meta:other rows that fit")


class CategoryConsolidationResult(BaseModel):
    meta: str = Field(description="Meta-category under review")
    summary: str = Field(description="Summary of consolidation decisions")
    clusters: list[CategoryProposalCluster]


def create_relabel_models(allowed_categories: list[str]) -> tuple[type[BaseModel], type[BaseModel]]:
    """Create dynamic response models for targeted other-row relabeling."""
    CategoryLiteral = Literal[tuple(sorted(allowed_categories))]
    RelabelChoice = create_model(
        "RelabelChoice",
        sample_uuid=(str, Field(description="Stable row identifier from input")),
        category=(CategoryLiteral, Field(description="Best category within the same meta or other")),
        reasoning=(str, Field(description="Short justification")),
        suggested_category=(str | None, Field(default=None, description="Optional suggestion if category stays other")),
    )
    RelabelBatchResult = create_model(
        "RelabelBatchResult",
        assignments=(list[RelabelChoice], Field(description="Assignment for every input row")),
    )
    return RelabelChoice, RelabelBatchResult


def get_client(base_url: str | None = None):
    return make_openai_client(base_url=base_url, async_client=True)


def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text()


def normalize_code(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", (value or "").strip().lower())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


def normalize_row_identifier(value: str) -> str:
    """Normalize model-returned row identifiers to the raw numeric id."""
    normalized = (value or "").strip()
    if normalized.startswith("[") and normalized.endswith("]"):
        normalized = normalized[1:-1].strip()
    return normalized


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

    kwargs: dict[str, Any] = {
        "temperature": resolve_temperature(model, default_temperature),
    }
    if not local:
        kwargs["extra_body"] = {"provider": {"zdr": True}}
    return kwargs


def category_code(category: dict[str, Any]) -> str:
    return category.get("code", category.get("name", "unknown"))


def load_taxonomy(taxonomy_model: str) -> tuple[dict[str, Any], Path]:
    path = resolve_provider_path(
        INTERMEDIATE_DIR / "categories_consolidated.json",
        taxonomy_model,
        fallback=DATA_DIR / "categories.json",
    )
    if not path.exists():
        raise FileNotFoundError("No consolidated or legacy categories file found")
    with open(path) as f:
        return json.load(f), path


def save_taxonomy(data: dict[str, Any], taxonomy_model: str) -> Path:
    path = provider_scoped_path(INTERMEDIATE_DIR / "categories_consolidated.json", taxonomy_model)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path


def taxonomy_to_hash_payload(taxonomy: dict[str, Any]) -> tuple[list[str], dict[str, list[str]]]:
    meta_codes = taxonomy.get("meta_categories") or list(taxonomy.get("categories_by_meta", {}).keys())
    categories_by_meta = {
        meta: [category_code(category) for category in info.get("categories", [])]
        for meta, info in taxonomy.get("categories_by_meta", {}).items()
    }
    for meta in meta_codes:
        if "other" not in categories_by_meta.get(meta, []):
            categories_by_meta.setdefault(meta, []).append("other")
    return meta_codes, categories_by_meta


def compute_taxonomy_hash(taxonomy: dict[str, Any]) -> str:
    meta_codes, categories_by_meta = taxonomy_to_hash_payload(taxonomy)
    canonical = json.dumps(
        {"meta_codes": meta_codes, "categories_by_meta": categories_by_meta},
        sort_keys=True,
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_state(labeling_model: str) -> tuple[dict[str, Any], Path]:
    state_path = resolve_provider_path(CACHE_DIR / "labeling_state.json", labeling_model)
    if not state_path.exists():
        raise FileNotFoundError("No labeling state found. Run Phase 04 first.")
    with open(state_path) as f:
        return json.load(f), state_path


def get_cache_paths(labeling_model: str) -> tuple[Path, Path]:
    db_path = resolve_provider_path(CACHE_DIR / "labels.db", labeling_model)
    results_path = resolve_provider_path(CACHE_DIR / "labels.jsonl", labeling_model)
    if not db_path.exists():
        raise FileNotFoundError("No labels.db found. Run Phase 04 first.")
    return db_path, results_path


def ensure_labeling_complete(state: dict[str, Any]) -> None:
    total_samples = int(state.get("total_samples", 0))
    processed = int(state.get("last_processed_index", 0))
    if total_samples and processed < total_samples:
        raise RuntimeError(
            "Phase 05 requires a complete Phase 04 run. "
            f"Only {processed}/{total_samples} samples are marked processed."
        )


def load_category_counts(conn: sqlite3.Connection) -> dict[str, dict[str, int]]:
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT meta, category, COUNT(*)
        FROM labels
        GROUP BY meta, category
        """
    )
    counts: dict[str, dict[str, int]] = defaultdict(dict)
    for meta, category, count in cursor.fetchall():
        counts[meta][category] = count
    return counts


def load_other_rows(conn: sqlite3.Connection) -> tuple[list[dict[str, Any]], int]:
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT l.id, s.id, s.original_index, s.text, l.meta, l.reasoning, l.suggested_category
        FROM labels l
        JOIN samples s ON s.id = l.sample_id
        WHERE l.category = 'other' AND l.meta != 'other'
        ORDER BY l.meta ASC, s.original_index ASC, l.id ASC
        """
    )
    rows = [
        {
            "label_id": label_id,
            "sample_id": sample_id,
            "original_index": original_index,
            "text": text,
            "meta": meta,
            "reasoning": reasoning or "",
            "suggested_category": suggested_category,
        }
        for label_id, sample_id, original_index, text, meta, reasoning, suggested_category in cursor.fetchall()
    ]

    sample_ids = sorted({row["sample_id"] for row in rows})
    labels_by_sample: dict[int, list[dict[str, Any]]] = defaultdict(list)
    if sample_ids:
        placeholders = ",".join("?" for _ in sample_ids)
        cursor.execute(
            f"""
            SELECT l.id, l.sample_id, l.meta, l.category, l.reasoning, l.suggested_category
            FROM labels l
            WHERE l.sample_id IN ({placeholders})
            ORDER BY l.sample_id ASC, l.id ASC
            """,
            sample_ids,
        )
        for label_id, sample_id, meta, category, reasoning, suggested_category in cursor.fetchall():
            labels_by_sample[int(sample_id)].append(
                {
                    "label_id": int(label_id),
                    "meta": meta,
                    "category": category,
                    "reasoning": reasoning or "",
                    "suggested_category": suggested_category,
                }
            )

    for row in rows:
        row["sample_label_context"] = [
            label
            for label in labels_by_sample.get(int(row["sample_id"]), [])
            if label["label_id"] != int(row["label_id"])
        ]

    cursor.execute(
        """
        SELECT COUNT(*)
        FROM labels
        WHERE category = 'other' AND meta = 'other'
        """
    )
    ignored_other_other_count = int(cursor.fetchone()[0])
    return rows, ignored_other_other_count


def sample_rows(rows: list[dict[str, Any]], n_samples: int, seed: int) -> list[dict[str, Any]]:
    if len(rows) <= n_samples:
        return list(rows)
    sampler = random.Random(seed)
    indices = sampler.sample(range(len(rows)), n_samples)
    indices.sort()
    return [rows[idx] for idx in indices]


def format_category_list(categories: list[dict[str, Any]]) -> str:
    lines = []
    seen: set[str] = set()
    for category in categories:
        code = category_code(category)
        if code in seen:
            continue
        seen.add(code)
        lines.append(f"- {category_code(category)}: {category.get('description', '')}")
    if "other" not in seen:
        lines.append("- other: Catch-all for unmatched rows within this meta")
    return "\n".join(lines) if lines else "- other: Catch-all for unmatched rows"


def format_other_meta_context(taxonomy: dict[str, Any], focus_meta: str) -> str:
    lines: list[str] = []
    categories_by_meta = taxonomy.get("categories_by_meta", {})
    for meta, info in categories_by_meta.items():
        if meta == focus_meta:
            continue
        description = info.get("meta_description", "")
        lines.append(f"- {meta}: {description}")
        category_codes = [category_code(category) for category in info.get("categories", [])]
        if category_codes:
            lines.append(f"  categories: {', '.join(category_codes)}")
    return "\n".join(lines) if lines else "- none"


def format_existing_labels(labels: list[dict[str, Any]]) -> str:
    if not labels:
        return "none"

    parts: list[str] = []
    for label in labels:
        item = f"{label['meta']}:{label['category']}"
        suggested = normalize_code(label.get("suggested_category") or "")
        if label["category"] == "other" and suggested:
            item += f" (suggested={suggested})"
        parts.append(item)
    return "; ".join(parts)


def format_other_samples(rows: list[dict[str, Any]]) -> str:
    formatted = []
    for row in rows:
        suggested = row.get("suggested_category") or "none"
        reasoning = row.get("reasoning") or "none"
        existing_labels = format_existing_labels(row.get("sample_label_context", []))
        formatted.append(
            f"sample_uuid: {row['label_id']}\n"
            f"text: {row['text']}\n"
            f"existing_labels_on_sample: {existing_labels}\n"
            f"  previous_reasoning: {reasoning}\n"
            f"  previous_suggested_category: {suggested}"
        )
    return "\n".join(formatted)


def format_proposal_runs(runs: list[dict[str, Any]]) -> str:
    lines = []
    for run in runs:
        seed_id = f"seed_{run['seed']}"
        lines.append(f"## {seed_id}")
        lines.append(f"Summary: {run['summary']}")
        for category in run["proposed_categories"]:
            lines.append(
                f"- {seed_id}:{category['code']}: {category['description']} "
                f"(estimated_count={category['estimated_count']})"
            )
        lines.append(f"Remaining other estimate: {run['remaining_other_estimate']}")
        lines.append("")
    return "\n".join(lines).strip()


def build_candidate_meta_rows(
    taxonomy: dict[str, Any],
    other_rows: list[dict[str, Any]],
    category_counts: dict[str, dict[str, int]],
    min_other_count: int,
    min_other_rate: float,
) -> list[dict[str, Any]]:
    rows_by_meta: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in other_rows:
        rows_by_meta[row["meta"]].append(row)

    candidates = []
    for meta, rows in rows_by_meta.items():
        if meta not in taxonomy.get("categories_by_meta", {}):
            continue
        meta_counts = category_counts.get(meta, {})
        total_count = sum(meta_counts.values())
        other_count = len(rows)
        other_rate = (other_count / total_count) if total_count else 0.0
        if other_count < min_other_count and other_rate < min_other_rate:
            continue
        meta_info = taxonomy["categories_by_meta"][meta]
        candidates.append(
            {
                "meta": meta,
                "meta_description": meta_info.get("meta_description", ""),
                "rows": rows,
                "other_count": other_count,
                "total_count": total_count,
                "other_rate": other_rate,
                "existing_categories": list(meta_info.get("categories", [])),
            }
        )

    candidates.sort(key=lambda item: (-item["other_count"], item["meta"]))
    return candidates


async def call_structured_with_retry(
    client: AsyncOpenAI,
    *,
    model: str,
    prompt: str,
    response_model,
    local: bool,
    semaphore: asyncio.Semaphore,
    system_prompt: str,
    request_timeout_seconds: float,
    temperature: float,
    operation_label: str,
    validator=None,
    max_attempts: int = DEFAULT_BATCH_MAX_ATTEMPTS,
):
    generation_kwargs = get_generation_kwargs(model, local=local, default_temperature=temperature)
    async def run_attempt(retry_context):
        async with semaphore:
            if local:
                prompt_with_system = f"{system_prompt}\n\n{append_validation_feedback_prompt(prompt, retry_context)}"
                result = await local_structured_call(
                    client,
                    model,
                    prompt_with_system,
                    response_model,
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
                    response_format=response_model,
                    extra_body=generation_kwargs.get("extra_body"),
                    temperature=generation_kwargs.get("temperature"),
                    top_p=generation_kwargs.get("top_p"),
                    presence_penalty=generation_kwargs.get("presence_penalty"),
                    max_attempts=1,
                    request_timeout_seconds=request_timeout_seconds,
                )
        if validator is not None:
            try:
                validator(result)
            except Exception as exc:
                raise ValidationRetryError(str(exc), failed_result=result) from exc
        return result

    return await run_with_validation_repair(
        model=model,
        operation_label=operation_label,
        run_attempt=run_attempt,
        max_attempts=max_attempts,
    )


def normalize_proposed_categories(proposed_categories: list[ProposedCategory]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for category in proposed_categories:
        code = normalize_code(category.code)
        if not code or code == "other" or code in seen:
            continue
        seen.add(code)
        normalized.append(
            {
                "code": code,
                "description": category.description.strip(),
                "estimated_count": int(category.estimated_count),
            }
        )
    return normalized


async def run_category_proposal(
    client: AsyncOpenAI,
    *,
    model: str,
    local: bool,
    taxonomy: dict[str, Any],
    meta: str,
    meta_description: str,
    existing_categories: list[dict[str, Any]],
    sampled_rows: list[dict[str, Any]],
    other_count: int,
    max_new_categories: int,
    domain_context: str,
    target: str,
    semaphore: asyncio.Semaphore,
    system_prompt: str,
    request_timeout_seconds: float,
) -> CategoryProposalRun:
    prompt = load_prompt("phase_05_other_category_audit.md")
    prompt = (
        prompt
        .replace("{domain_context}", domain_context)
        .replace("{target}", target)
        .replace("{meta}", meta)
        .replace("{meta_description}", meta_description or "none")
        .replace("{other_count}", str(other_count))
        .replace("{max_new_categories}", str(max_new_categories))
        .replace("{existing_categories}", format_category_list(existing_categories))
        .replace("{other_meta_context}", format_other_meta_context(taxonomy, meta))
        .replace("{samples}", format_other_samples(sampled_rows))
    )
    return await call_structured_with_retry(
        client,
        model=model,
        prompt=prompt,
        response_model=CategoryProposalRun,
        local=local,
        semaphore=semaphore,
        system_prompt=system_prompt,
        request_timeout_seconds=request_timeout_seconds,
        temperature=0.2,
        operation_label=f"proposal meta {meta}",
    )


async def run_category_consolidation(
    client: AsyncOpenAI,
    *,
    model: str,
    local: bool,
    taxonomy: dict[str, Any],
    meta: str,
    meta_description: str,
    existing_categories: list[dict[str, Any]],
    runs: list[dict[str, Any]],
    other_count: int,
    max_new_categories: int,
    domain_context: str,
    target: str,
    semaphore: asyncio.Semaphore,
    system_prompt: str,
    request_timeout_seconds: float,
) -> CategoryConsolidationResult:
    prompt = load_prompt("phase_05_other_category_consolidation.md")
    prompt = (
        prompt
        .replace("{domain_context}", domain_context)
        .replace("{target}", target)
        .replace("{meta}", meta)
        .replace("{meta_description}", meta_description or "none")
        .replace("{other_count}", str(other_count))
        .replace("{max_new_categories}", str(max_new_categories))
        .replace("{existing_categories}", format_category_list(existing_categories))
        .replace("{other_meta_context}", format_other_meta_context(taxonomy, meta))
        .replace("{runs}", format_proposal_runs(runs))
    )
    return await call_structured_with_retry(
        client,
        model=model,
        prompt=prompt,
        response_model=CategoryConsolidationResult,
        local=local,
        semaphore=semaphore,
        system_prompt=system_prompt,
        request_timeout_seconds=request_timeout_seconds,
        temperature=0.2,
        operation_label=f"consolidation meta {meta}",
    )


def compute_support(members: list[str]) -> int:
    run_ids = {member.split(":", 1)[0].strip() for member in members if member}
    return len(run_ids)


def build_reserved_code_aliases(values: set[str]) -> set[str]:
    aliases: set[str] = set()
    for value in values:
        code = normalize_code(value)
        if not code:
            continue
        aliases.add(code)
        if code.endswith("s") and len(code) > 3:
            aliases.add(code[:-1])
        else:
            aliases.add(f"{code}s")
    return aliases


def is_reserved_candidate_code(code: str, reserved_aliases: set[str]) -> bool:
    if code in reserved_aliases:
        return True
    return any(code.startswith(f"{alias}_") for alias in reserved_aliases if alias)


def select_new_categories(
    consolidation: CategoryConsolidationResult,
    existing_codes: set[str],
    reserved_codes: set[str],
    min_category_support: int,
    min_estimated_count: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for cluster in consolidation.clusters:
        code = normalize_code(cluster.code)
        if (
            not code
            or code == "other"
            or code in existing_codes
            or is_reserved_candidate_code(code, reserved_codes)
            or code in seen
        ):
            continue
        support = compute_support(cluster.members)
        estimated_count = int(cluster.estimated_count)
        if support < min_category_support:
            continue
        if estimated_count < min_estimated_count:
            continue
        seen.add(code)
        selected.append(
            {
                "code": code,
                "description": cluster.description.strip(),
                "support": support,
                "estimated_count": estimated_count,
                "members": cluster.members,
                "added_by_phase_05": True,
            }
        )
    return selected


async def run_relabel_batch(
    client: AsyncOpenAI,
    *,
    model: str,
    local: bool,
    taxonomy: dict[str, Any],
    meta: str,
    meta_description: str,
    allowed_categories: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    domain_context: str,
    target: str,
    semaphore: asyncio.Semaphore,
    system_prompt: str,
    request_timeout_seconds: float,
    max_attempts: int = DEFAULT_BATCH_MAX_ATTEMPTS,
) -> list[dict[str, Any]]:
    allowed_codes = [category_code(category) for category in allowed_categories]
    if "other" not in allowed_codes:
        allowed_codes.append("other")
    _, BatchResultModel = create_relabel_models(allowed_codes)

    prompt = load_prompt("phase_05_other_relabel.md")
    prompt = (
        prompt
        .replace("{domain_context}", domain_context)
        .replace("{target}", target)
        .replace("{meta}", meta)
        .replace("{meta_description}", meta_description or "none")
        .replace("{allowed_categories}", format_category_list(allowed_categories))
        .replace("{other_meta_context}", format_other_meta_context(taxonomy, meta))
        .replace("{samples}", format_other_samples(rows))
    )

    generation_kwargs = get_generation_kwargs(model, local=local, default_temperature=0.2)
    sent_ids = {str(row["label_id"]) for row in rows}
    async def run_attempt(retry_context):
        async with semaphore:
            if local:
                prompt_with_system = f"{system_prompt}\n\n{append_validation_feedback_prompt(prompt, retry_context)}"
                result = await local_structured_call(
                    client,
                    model,
                    prompt_with_system,
                    BatchResultModel,
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
                    response_format=BatchResultModel,
                    extra_body=generation_kwargs.get("extra_body"),
                    temperature=generation_kwargs.get("temperature"),
                    top_p=generation_kwargs.get("top_p"),
                    presence_penalty=generation_kwargs.get("presence_penalty"),
                    max_attempts=1,
                    request_timeout_seconds=request_timeout_seconds,
                )

        normalized_ids = [normalize_row_identifier(assignment.sample_uuid) for assignment in result.assignments]
        received_ids = set(normalized_ids)
        if len(received_ids) != len(normalized_ids):
            raise ValidationRetryError(
                f"Duplicate row identifiers returned for meta {meta}",
                failed_result=result,
            )
        if sent_ids != received_ids:
            missing = sorted(sent_ids - received_ids)
            extra = sorted(received_ids - sent_ids)
            raise ValidationRetryError(
                f"Relabel batch mismatch for meta {meta}. Missing={missing[:5]} Extra={extra[:5]}",
                failed_result=result,
            )

        batch_assignments: list[dict[str, Any]] = []
        try:
            for assignment in result.assignments:
                normalized_id = normalize_row_identifier(assignment.sample_uuid)
                assigned_category = assignment.category
                suggested_category = normalize_code(assignment.suggested_category or "") or None
                if assigned_category != "other":
                    suggested_category = None
                batch_assignments.append(
                    {
                        "label_id": int(normalized_id),
                        "meta": meta,
                        "category": assigned_category,
                        "reasoning": assignment.reasoning,
                        "suggested_category": suggested_category,
                    }
                )
        except Exception as exc:
            raise ValidationRetryError(str(exc), failed_result=result) from exc
        return batch_assignments

    return await run_with_validation_repair(
        model=model,
        operation_label=f"relabel meta {meta}",
        run_attempt=run_attempt,
        max_attempts=max_attempts,
    )


async def relabel_other_rows(
    client: AsyncOpenAI,
    *,
    model: str,
    local: bool,
    taxonomy: dict[str, Any],
    meta: str,
    meta_description: str,
    allowed_categories: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    relabel_batch_size: int,
    domain_context: str,
    target: str,
    semaphore: asyncio.Semaphore,
    system_prompt: str,
    request_timeout_seconds: float,
) -> list[dict[str, Any]]:
    assignments: list[dict[str, Any]] = []
    batches = [
        (start, rows[start:start + relabel_batch_size])
        for start in range(0, len(rows), relabel_batch_size)
    ]

    async def process_batch(start: int, batch: list[dict[str, Any]]):
        batch_assignments = await run_relabel_batch(
            client,
            model=model,
            local=local,
            taxonomy=taxonomy,
            meta=meta,
            meta_description=meta_description,
            allowed_categories=allowed_categories,
            rows=batch,
            domain_context=domain_context,
            target=target,
            semaphore=semaphore,
            system_prompt=system_prompt,
            request_timeout_seconds=request_timeout_seconds,
        )
        print(f"  Relabeled meta {meta}: rows {start + 1}-{start + len(batch)} / {len(rows)}")
        return start, batch_assignments

    results = await asyncio.gather(
        *[process_batch(start, batch) for start, batch in batches],
        return_exceptions=True,
    )
    errors = [result for result in results if isinstance(result, Exception)]
    if errors:
        raise RuntimeError(
            f"{len(errors)} relabel batches failed for meta {meta}: {errors[0]}"
        )
    for _start, batch_assignments in sorted(results, key=lambda item: item[0]):
        assignments.extend(batch_assignments)
    return assignments


def finalize_assignments(
    assignments: list[dict[str, Any]],
    tentative_new_categories: list[dict[str, Any]],
    min_applied_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    tentative_codes = {category["code"] for category in tentative_new_categories}
    assignment_counts = Counter(
        assignment["category"] for assignment in assignments if assignment["category"] in tentative_codes
    )

    finalized_categories = [
        category for category in tentative_new_categories
        if assignment_counts.get(category["code"], 0) >= min_applied_count
    ]
    finalized_codes = {category["code"] for category in finalized_categories}

    adjusted_assignments: list[dict[str, Any]] = []
    for assignment in assignments:
        adjusted = dict(assignment)
        if adjusted["category"] in tentative_codes and adjusted["category"] not in finalized_codes:
            dropped_code = adjusted["category"]
            adjusted["category"] = "other"
            adjusted["suggested_category"] = dropped_code
            adjusted["reasoning"] = (
                f"Retained as other after Phase 05 review; tentative category '{dropped_code}' "
                "did not reach the applied-count threshold."
            )
        adjusted_assignments.append(adjusted)

    final_assignment_counts = Counter(assignment["category"] for assignment in adjusted_assignments)
    return finalized_categories, adjusted_assignments, dict(final_assignment_counts)


def add_new_categories_to_taxonomy(
    taxonomy: dict[str, Any],
    categories_by_meta: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    updated = deepcopy(taxonomy)
    for meta, categories in categories_by_meta.items():
        updated["categories_by_meta"][meta].setdefault("categories", [])
        updated["categories_by_meta"][meta]["categories"].extend(categories)
    return updated


def apply_assignments(conn: sqlite3.Connection, assignments: list[dict[str, Any]]) -> None:
    cursor = conn.cursor()
    for assignment in assignments:
        cursor.execute(
            """
            UPDATE labels
            SET category = ?, reasoning = ?, suggested_category = ?
            WHERE id = ?
            """,
            (
                assignment["category"],
                assignment["reasoning"],
                assignment["suggested_category"],
                assignment["label_id"],
            ),
        )
    conn.commit()


def rebuild_results_jsonl(conn: sqlite3.Connection, results_path: Path) -> None:
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT s.original_index, s.text, l.id, l.meta, l.category, l.reasoning, l.suggested_category
        FROM samples s
        LEFT JOIN labels l ON l.sample_id = s.id
        ORDER BY s.original_index ASC, l.id ASC
        """
    )
    records: list[dict[str, Any]] = []
    current_index: int | None = None
    current_record: dict[str, Any] | None = None
    for original_index, text, _label_id, meta, category, reasoning, suggested_category in cursor.fetchall():
        if current_index != original_index:
            if current_record is not None:
                records.append(current_record)
            current_index = original_index
            current_record = {
                "original_index": original_index,
                "text": text,
                "labels": [],
            }
        if meta is not None and category is not None:
            current_record["labels"].append(
                {
                    "meta": meta,
                    "category": category,
                    "reasoning": reasoning,
                    "suggested_category": suggested_category,
                }
            )
    if current_record is not None:
        records.append(current_record)

    with open(results_path, "w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def rebuild_state(
    conn: sqlite3.Connection,
    state: dict[str, Any],
    *,
    taxonomy_hash: str,
    model_used: str,
) -> dict[str, Any]:
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM samples")
    total_samples = int(cursor.fetchone()[0])

    cursor.execute(
        """
        SELECT meta, category, COUNT(*)
        FROM labels
        GROUP BY meta, category
        ORDER BY meta ASC, category ASC
        """
    )
    category_counts: dict[str, dict[str, int]] = defaultdict(dict)
    for meta, category, count in cursor.fetchall():
        category_counts[meta][category] = count

    cursor.execute(
        """
        SELECT s.original_index, s.text, l.meta, l.suggested_category, l.reasoning
        FROM labels l
        JOIN samples s ON s.id = l.sample_id
        WHERE l.category = 'other'
        ORDER BY s.original_index ASC, l.id ASC
        """
    )
    other_samples = [
        {
            "index": original_index,
            "text": text,
            "meta": meta,
            "suggested": suggested_category,
            "reasoning": reasoning,
        }
        for original_index, text, meta, suggested_category, reasoning in cursor.fetchall()
    ]

    updated = dict(state)
    updated["last_processed_index"] = total_samples
    updated["total_samples"] = total_samples
    updated["category_counts"] = category_counts
    updated["other_samples"] = other_samples
    updated["last_updated"] = datetime.now().isoformat()
    updated["model_used"] = model_used
    updated["taxonomy_hash"] = taxonomy_hash
    return updated


def save_state(state: dict[str, Any], state_path: Path) -> None:
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def print_meta_summary(candidate: dict[str, Any]) -> None:
    print(
        f"- {candidate['meta']}: other_count={candidate['other_count']} "
        f"total={candidate['total_count']} other_rate={candidate['other_rate']:.1%}"
    )


async def main(
    sample_mode: bool = False,
    analyze_only: bool = False,
    min_other_count: int = DEFAULT_MIN_OTHER_COUNT,
    min_other_rate: float = DEFAULT_MIN_OTHER_RATE,
    n_samples: int = DEFAULT_N_SAMPLES,
    seeds: list[int] | None = None,
    max_new_categories: int = DEFAULT_MAX_NEW_CATEGORIES,
    min_category_support: int = DEFAULT_MIN_CATEGORY_SUPPORT,
    min_estimated_count: int = DEFAULT_MIN_ESTIMATED_COUNT,
    min_applied_count: int = DEFAULT_MIN_APPLIED_COUNT,
    relabel_batch_size: int = DEFAULT_RELABEL_BATCH_SIZE,
    model: str = DEFAULT_MODEL,
    taxonomy_model: str = DEFAULT_TAXONOMY_MODEL,
    labeling_model: str = DEFAULT_LABELING_MODEL,
    concurrency: int | None = None,
    domain_context: str = DEFAULT_DOMAIN_CONTEXT,
    target: str = DEFAULT_TARGET,
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    base_url: str | None = None,
    run_id: str | None = None,
):
    configure_paths(run_id)
    if seeds is None:
        seeds = DEFAULT_SEEDS

    print(f"Artifact scope: {DATA_DIR}")
    state, state_path = load_state(labeling_model)
    ensure_labeling_complete(state)

    taxonomy, taxonomy_path = load_taxonomy(taxonomy_model)
    db_path, results_path = get_cache_paths(labeling_model)
    report_path = provider_scoped_path(INTERMEDIATE_DIR / "other_healing_report.json", model)

    print(f"Using taxonomy: {taxonomy_path}")
    print(f"Using labels DB: {db_path}")
    print(f"Using labels JSONL: {results_path}")

    conn = sqlite3.connect(db_path)
    try:
        category_counts = load_category_counts(conn)
        other_rows, ignored_other_other_count = load_other_rows(conn)
        candidates = build_candidate_meta_rows(
            taxonomy,
            other_rows,
            category_counts,
            min_other_count=min_other_count,
            min_other_rate=min_other_rate,
        )

        print(f"Ignored other:other rows: {ignored_other_other_count}")
        print(f"Category-level other rows eligible for healing: {len(other_rows)}")
        if candidates:
            print("Metas selected for Phase 05 healing:")
            for candidate in candidates:
                print_meta_summary(candidate)
        else:
            print("No metas exceed the configured other thresholds.")

        if sample_mode:
            candidates = candidates[:1]
            seeds = seeds[:2]
            n_samples = min(n_samples, 20)
            print(f"=== SAMPLE MODE: {len(candidates)} metas, seeds {seeds}, n_samples={n_samples} ===")

        report: dict[str, Any] = {
            "phase": "05_review_other",
            "model": model,
            "taxonomy_model": taxonomy_model,
            "labeling_model": labeling_model,
            "timestamp": datetime.now().isoformat(),
            "ignored_other_other_count": ignored_other_other_count,
            "thresholds": {
                "min_other_count": min_other_count,
                "min_other_rate": min_other_rate,
                "n_samples": n_samples,
                "seeds": seeds,
                "max_new_categories": max_new_categories,
                "min_category_support": min_category_support,
                "min_estimated_count": min_estimated_count,
                "min_applied_count": min_applied_count,
                "relabel_batch_size": relabel_batch_size,
            },
            "metas": [],
        }

        if not candidates:
            if sample_mode or analyze_only:
                print("No files written (sample/analyze-only mode).")
            else:
                with open(report_path, "w") as f:
                    json.dump(report, f, indent=2, ensure_ascii=False)
                print(f"Saved report: {report_path}")
            return

        client, local = get_client(base_url=base_url)
        concurrency = resolve_concurrency(concurrency, local, model=model)
        semaphore = asyncio.Semaphore(concurrency)
        report_lock = asyncio.Lock()
        report_entries: dict[str, dict[str, Any]] = {}
        print(f"Concurrency: {concurrency} parallel requests")

        async def persist_partial_report(meta: str, entry: dict[str, Any]) -> None:
            async with report_lock:
                report_entries[meta] = entry
                report["metas"] = [report_entries[key] for key in sorted(report_entries)]
                if not sample_mode and not analyze_only:
                    with open(report_path, "w") as f:
                        json.dump(report, f, indent=2, ensure_ascii=False)

        async def process_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
            meta = candidate["meta"]
            meta_report: dict[str, Any] = {
                "meta": meta,
                "other_count_before": candidate["other_count"],
                "total_count_before": candidate["total_count"],
                "other_rate_before": candidate["other_rate"],
                "proposal_runs": [],
                "proposal_errors": [],
                "status": "in_progress",
            }
            print(f"\n=== Phase 05 healing for meta: {meta} ===")
            try:
                proposal_tasks = []
                for seed in seeds:
                    sampled_rows = sample_rows(candidate["rows"], n_samples=n_samples, seed=seed)
                    proposal_tasks.append(
                        run_category_proposal(
                            client,
                            model=model,
                            local=local,
                            taxonomy=taxonomy,
                            meta=meta,
                            meta_description=candidate["meta_description"],
                            existing_categories=candidate["existing_categories"],
                            sampled_rows=sampled_rows,
                            other_count=candidate["other_count"],
                            max_new_categories=max_new_categories,
                            domain_context=domain_context,
                            target=target,
                            semaphore=semaphore,
                            system_prompt=system_prompt,
                            request_timeout_seconds=request_timeout_seconds,
                        )
                    )

                proposal_results = await asyncio.gather(*proposal_tasks, return_exceptions=True)
                proposal_runs: list[dict[str, Any]] = []
                for seed, proposal_result in zip(seeds, proposal_results):
                    if isinstance(proposal_result, Exception):
                        meta_report["proposal_errors"].append(
                            f"seed {seed}: {type(proposal_result).__name__}: {proposal_result}"
                        )
                        continue
                    proposal_run = {
                        "seed": seed,
                        "summary": proposal_result.summary,
                        "remaining_other_estimate": proposal_result.remaining_other_estimate,
                        "proposed_categories": normalize_proposed_categories(proposal_result.proposed_categories),
                    }
                    proposal_runs.append(proposal_run)
                    print(
                        f"  Seed {seed}: proposed {len(proposal_run['proposed_categories'])} categories, "
                        f"remaining_other_estimate={proposal_run['remaining_other_estimate']}"
                    )

                if not proposal_runs:
                    raise RuntimeError(f"No successful proposal runs for meta {meta}")

                consolidation = await run_category_consolidation(
                    client,
                    model=model,
                    local=local,
                    taxonomy=taxonomy,
                    meta=meta,
                    meta_description=candidate["meta_description"],
                    existing_categories=candidate["existing_categories"],
                    runs=proposal_runs,
                    other_count=candidate["other_count"],
                    max_new_categories=max_new_categories,
                    domain_context=domain_context,
                    target=target,
                    semaphore=semaphore,
                    system_prompt=system_prompt,
                    request_timeout_seconds=request_timeout_seconds,
                )

                existing_codes = {category_code(category) for category in candidate["existing_categories"]}
                reserved_codes = build_reserved_code_aliases(set(taxonomy.get("meta_categories", [])) | {target})
                selected_new_categories = select_new_categories(
                    consolidation,
                    existing_codes=existing_codes,
                    reserved_codes=reserved_codes,
                    min_category_support=min_category_support,
                    min_estimated_count=min_estimated_count,
                )
                print(
                    f"  Consolidated tentative new categories for meta {meta}: "
                    f"{[category['code'] for category in selected_new_categories]}"
                )

                allowed_categories = list(candidate["existing_categories"])
                allowed_categories.extend(
                    {"code": category["code"], "description": category["description"]}
                    for category in selected_new_categories
                )
                assignments = await relabel_other_rows(
                    client,
                    model=model,
                    local=local,
                    taxonomy=taxonomy,
                    meta=meta,
                    meta_description=candidate["meta_description"],
                    allowed_categories=allowed_categories,
                    rows=candidate["rows"],
                    relabel_batch_size=relabel_batch_size,
                    domain_context=domain_context,
                    target=target,
                    semaphore=semaphore,
                    system_prompt=system_prompt,
                    request_timeout_seconds=request_timeout_seconds,
                )

                final_new_categories, adjusted_assignments, assignment_counts = finalize_assignments(
                    assignments,
                    tentative_new_categories=selected_new_categories,
                    min_applied_count=min_applied_count,
                )
                print(f"  Finalized new categories for meta {meta}: {[category['code'] for category in final_new_categories]}")
                print(f"  Assignment summary: {assignment_counts}")

                meta_report.update(
                    {
                        "status": "completed",
                        "proposal_runs": proposal_runs,
                        "consolidation": consolidation.model_dump(),
                        "tentative_new_categories": selected_new_categories,
                        "finalized_new_categories": final_new_categories,
                        "assignment_counts": assignment_counts,
                    }
                )
                await persist_partial_report(meta, meta_report)
                return {
                    "meta": meta,
                    "report_entry": meta_report,
                    "finalized_new_categories": final_new_categories,
                    "assignments": adjusted_assignments,
                    "error": None,
                }
            except Exception as exc:
                meta_report.update(
                    {
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                await persist_partial_report(meta, meta_report)
                print(f"  Meta {meta} failed: {exc}")
                return {
                    "meta": meta,
                    "report_entry": meta_report,
                    "finalized_new_categories": [],
                    "assignments": [],
                    "error": meta_report["error"],
                }

        meta_results = await asyncio.gather(
            *[process_candidate(candidate) for candidate in candidates],
            return_exceptions=False,
        )

        finalized_new_categories: dict[str, list[dict[str, Any]]] = {}
        assignments_to_apply: list[dict[str, Any]] = []
        report["meta_errors"] = []
        for meta_result in meta_results:
            if meta_result["error"]:
                report["meta_errors"].append(meta_result["error"])
                continue
            finalized_new_categories[meta_result["meta"]] = meta_result["finalized_new_categories"]
            assignments_to_apply.extend(meta_result["assignments"])

        taxonomy_after_healing = add_new_categories_to_taxonomy(taxonomy, finalized_new_categories)
        taxonomy_hash = compute_taxonomy_hash(taxonomy_after_healing)
        report["taxonomy_changed"] = any(finalized_new_categories.values())
        report["updated_rows"] = len(assignments_to_apply)
        report["metas"] = [report_entries[key] for key in sorted(report_entries)]

        if report["meta_errors"]:
            print(f"\n{len(report['meta_errors'])} metas failed during healing:")
            for error in report["meta_errors"][:5]:
                print(f"  - {error}")

        if sample_mode or analyze_only:
            print("\nNo files written (sample/analyze-only mode).")
            return

        if report["meta_errors"] and not (assignments_to_apply or report["taxonomy_changed"]):
            report["write_applied"] = False
            report["write_skipped_reason"] = "All metas failed; no successful changes were available to apply."
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print("\nNo writes applied because no successful meta changes were available.")
            print(f"Saved report: {report_path}")
            raise RuntimeError(
                f"Phase 05 incomplete: {len(report['meta_errors'])} metas failed and no writes were applied."
            )

        if report["meta_errors"]:
            print("\nApplying partial writes for successful metas; failed metas remain available for rerun.")

        taxonomy_save_path = save_taxonomy(taxonomy_after_healing, taxonomy_model)
        apply_assignments(conn, assignments_to_apply)
        rebuild_results_jsonl(conn, results_path)
        updated_state = rebuild_state(
            conn,
            state,
            taxonomy_hash=taxonomy_hash,
            model_used=str(state.get("model_used") or labeling_model),
        )
        save_state(updated_state, state_path)
        report["write_applied"] = True
        report["write_partial"] = bool(report["meta_errors"])

        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print("\nPhase 05 healing complete")
        print(f"Updated taxonomy: {taxonomy_save_path}")
        print(f"Updated labels DB: {db_path}")
        print(f"Rebuilt labels JSONL: {results_path}")
        print(f"Updated state: {state_path}")
        print(f"Saved report: {report_path}")
        if report["meta_errors"]:
            raise RuntimeError(
                f"Phase 05 incomplete: {len(report['meta_errors'])} metas still failed after partial apply."
            )
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 05: Heal category-level other labels in place")
    parser.add_argument("--sample", action="store_true", help="Run with reduced scope and do not save")
    parser.add_argument("--analyze-only", action="store_true", help="Run the healing logic without writing any changes")
    parser.add_argument("--min-other-count", type=int, default=DEFAULT_MIN_OTHER_COUNT, help="Minimum meta:other count to trigger healing")
    parser.add_argument("--min-other-rate", type=float, default=DEFAULT_MIN_OTHER_RATE, help="Minimum meta:other rate to trigger healing")
    parser.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES, help="Samples per proposal seed")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS, help="Random seeds for proposal sampling")
    parser.add_argument("--max-new-categories", type=int, default=DEFAULT_MAX_NEW_CATEGORIES, help="Maximum new categories per meta")
    parser.add_argument("--min-category-support", type=int, default=DEFAULT_MIN_CATEGORY_SUPPORT, help="Minimum seed support for new categories")
    parser.add_argument("--min-estimated-count", type=int, default=DEFAULT_MIN_ESTIMATED_COUNT, help="Minimum estimated count for new categories")
    parser.add_argument("--min-applied-count", type=int, default=DEFAULT_MIN_APPLIED_COUNT, help="Minimum actual reassigned rows to keep a new category")
    parser.add_argument("--relabel-batch-size", type=int, default=DEFAULT_RELABEL_BATCH_SIZE, help="Rows per targeted relabel batch")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model used for Phase 05 healing")
    parser.add_argument("--taxonomy-model", type=str, default=DEFAULT_TAXONOMY_MODEL, help="Model used to generate the Phase 00-03 taxonomy")
    parser.add_argument("--labeling-model", type=str, default=DEFAULT_LABELING_MODEL, help="Model used for Phase 04 labeling")
    parser.add_argument("--concurrency", type=int, default=None, help="Parallel API calls (defaults: Qwen 27B=8, Qwen 9B=20, otherwise 10; local=1)")
    parser.add_argument("--domain-context", type=str, default=DEFAULT_DOMAIN_CONTEXT, help="Domain context")
    parser.add_argument("--target", type=str, default=DEFAULT_TARGET, help="Target concept")
    parser.add_argument("--request-timeout-seconds", type=float, default=DEFAULT_REQUEST_TIMEOUT_SECONDS, help="Per-attempt timeout for a single request")
    parser.add_argument("--system-prompt", type=str, default=DEFAULT_SYSTEM_PROMPT, help="System prompt for Phase 05 calls")
    parser.add_argument("--base-url", type=str, default=None, help="Custom API base URL (e.g. LM Studio)")
    parser.add_argument("--run-id", type=str, default=None, help="Optional run id for isolated artifacts under data/runs/<run_id>/")

    args = parser.parse_args()
    asyncio.run(main(
        sample_mode=args.sample,
        analyze_only=args.analyze_only,
        min_other_count=args.min_other_count,
        min_other_rate=args.min_other_rate,
        n_samples=args.n_samples,
        seeds=args.seeds,
        max_new_categories=args.max_new_categories,
        min_category_support=args.min_category_support,
        min_estimated_count=args.min_estimated_count,
        min_applied_count=args.min_applied_count,
        relabel_batch_size=args.relabel_batch_size,
        model=args.model,
        taxonomy_model=args.taxonomy_model,
        labeling_model=args.labeling_model,
        concurrency=args.concurrency,
        domain_context=args.domain_context,
        target=args.target,
        request_timeout_seconds=args.request_timeout_seconds,
        system_prompt=args.system_prompt,
        base_url=args.base_url,
        run_id=args.run_id,
    ))
