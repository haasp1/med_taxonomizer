"""
Phase 04: Sample Labeling (Async)

Label all samples using discovered taxonomy from Phase 00-03.
Processes batches in parallel with configurable concurrency.

Usage:
    uv run python scripts/phase_04_labeling.py --sample
    uv run python scripts/phase_04_labeling.py --concurrency 20
    uv run python scripts/phase_04_labeling.py --resume
    uv run python scripts/phase_04_labeling.py --rerun-missing-only
"""

import argparse
import asyncio
import hashlib
import json
import os
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd
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

# --- Configuration ---

EXPERIMENT_DIR = Path(__file__).parent
PROMPTS_DIR = EXPERIMENT_DIR / "prompts"
ARTIFACT_PATHS = resolve_experiment_artifact_paths(EXPERIMENT_DIR)
DATA_DIR = ARTIFACT_PATHS.data_dir
INTERMEDIATE_DIR = ARTIFACT_PATHS.intermediate_dir
CACHE_DIR = ARTIFACT_PATHS.cache_dir
INPUT_DATA = Path("experiments/01_data_preparation/data/texts_sufficient_only.parquet")

DEFAULT_MODEL = "openai/gpt-4o"
DEFAULT_TAXONOMY_MODEL = "openai/gpt-5.2"
DEFAULT_BATCH_SIZE = 10
DEFAULT_OTHER_THRESHOLD = 30
DEFAULT_CONCURRENCY = 10
DEFAULT_DOMAIN_CONTEXT = "Free-text entries from stroke intervention documentation."
DEFAULT_LANGUAGE_NOTE = "German medical terminology."
DEFAULT_REQUEST_TIMEOUT_SECONDS = 180.0
DEFAULT_BATCH_MAX_ATTEMPTS = 4
DEFAULT_SYSTEM_PROMPT = (
    "You are a concise, efficient, decisive assistant. "
    "Think in 2-3 short blocks per sample without repetition or second-guessing, "
    "and then output your answer."
)
QWEN_LABELING_TEMPERATURE = 0.6
QWEN_LABELING_TOP_P = 0.95
QWEN_LABELING_TOP_K = 20
QWEN_LABELING_PRESENCE_PENALTY = 1.5


def configure_paths(run_id: str | None) -> None:
    global ARTIFACT_PATHS, DATA_DIR, INTERMEDIATE_DIR, CACHE_DIR
    ARTIFACT_PATHS = resolve_experiment_artifact_paths(EXPERIMENT_DIR, run_id=run_id)
    DATA_DIR = ARTIFACT_PATHS.data_dir
    INTERMEDIATE_DIR = ARTIFACT_PATHS.intermediate_dir
    CACHE_DIR = ARTIFACT_PATHS.cache_dir


# --- Dynamic Model Creation ---

def load_taxonomy(taxonomy_model: str) -> tuple[list[str], dict[str, list[str]], dict[str, str]]:
    """
    Load taxonomy from Phase 00-03.

    Prefers consolidated files (from multi-seed pipeline) if available,
    falls back to single-run files for backward compatibility.

    Returns:
        meta_codes: List of meta-category codes
        categories_by_meta: Dict mapping meta -> list of category codes
        meta_descriptions: Dict mapping meta -> description
    """
    # Prefer consolidated files, fall back to old single-run files
    meta_file = resolve_provider_path(
        INTERMEDIATE_DIR / "meta_categories_consolidated.json",
        taxonomy_model,
        fallback=DATA_DIR / "meta_categories.json",
    )
    cat_file = resolve_provider_path(
        INTERMEDIATE_DIR / "categories_consolidated.json",
        taxonomy_model,
        fallback=DATA_DIR / "categories.json",
    )
    if not cat_file.exists():
        raise FileNotFoundError("Run Phase 03 first: categories.json or categories_consolidated.json not found")

    meta_data = {}
    if meta_file.exists():
        with open(meta_file) as f:
            meta_data = json.load(f)

    with open(cat_file) as f:
        cat_data = json.load(f)
    print(f"Using taxonomy from: {cat_file.name}")

    meta_codes = cat_data.get("meta_categories") or [m["code"] for m in meta_data.get("meta_categories", [])]
    categories_by_meta = {}
    meta_descriptions = {}

    for meta_code in meta_codes:
        meta_info = cat_data["categories_by_meta"].get(meta_code, {})
        meta_descriptions[meta_code] = meta_info.get("meta_description", "")
        if not meta_descriptions[meta_code]:
            for m in meta_data.get("meta_categories", []):
                if m.get("code") == meta_code:
                    meta_descriptions[meta_code] = m.get("description", "")
                    break
        # Try "name" first (new format), fall back to "code" (old format)
        categories_by_meta[meta_code] = [
            c.get("name", c.get("code", "unknown")) for c in meta_info.get("categories", [])
        ]
        # Ensure "other" is always present
        if "other" not in categories_by_meta[meta_code]:
            categories_by_meta[meta_code].append("other")

    return meta_codes, categories_by_meta, meta_descriptions


def compute_taxonomy_hash(
    meta_codes: list[str],
    categories_by_meta: dict[str, list[str]],
) -> str:
    """Create a stable hash for taxonomy compatibility checks."""
    payload = {
        "meta_codes": meta_codes,
        "categories_by_meta": categories_by_meta,
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_prompt_taxonomy_section(
    meta_codes: list[str],
    categories_by_meta: dict[str, list[str]],
    meta_descriptions: dict[str, str],
) -> tuple[str, str]:
    """Build the taxonomy sections for the prompt."""

    # Meta-categories list
    meta_list = "\n".join(
        f"- **{code}**: {meta_descriptions.get(code, '')}"
        for code in meta_codes
    )

    # Categories per meta
    cat_lines = []
    for meta in meta_codes:
        cats = categories_by_meta.get(meta, ["other"])
        cat_lines.append(f"**{meta}**: {', '.join(cats)}")
    categories_list = "\n".join(cat_lines)

    return meta_list, categories_list


# --- Pydantic Models ---

def create_label_models(
    meta_codes: list[str],
    categories_by_meta: dict[str, list[str]],
) -> tuple[type[BaseModel], type[BaseModel], type[BaseModel]]:
    """
    Create Label, SampleResult, and BatchResult models with Literal types.

    Returns:
        Label, SampleResult, BatchResult model classes
    """
    # Create Literal type for meta
    MetaLiteral = Literal[tuple(meta_codes)]

    # Create Literal type for all categories (union across all metas)
    all_categories = set()
    for cats in categories_by_meta.values():
        all_categories.update(cats)
    CategoryLiteral = Literal[tuple(sorted(all_categories))]

    # Create Label model with Literal types
    Label = create_model(
        'Label',
        meta=(MetaLiteral, Field(description="Meta-category")),
        category=(CategoryLiteral, Field(description="Category within the meta")),
        reasoning=(str, Field(description="Brief explanation (1 sentence)")),
        suggested_category=(str | None, Field(default=None, description="If category is 'other', suggest a new category")),
    )

    # Create SampleResult model
    SampleResult = create_model(
        'SampleResult',
        sample_uuid=(str, Field(description="UUID of the sample from input")),
        labels=(list[Label], Field(description="Labels ordered by importance")),
    )

    # Create BatchResult model
    BatchResult = create_model(
        'BatchResult',
        samples=(list[SampleResult], Field(description="Results for all samples in batch")),
    )

    return Label, SampleResult, BatchResult


class LabelingState(BaseModel):
    """State for resumable labeling."""

    last_processed_index: int = 0
    total_samples: int = 0
    other_samples: list[dict] = Field(default_factory=list)
    category_counts: dict[str, dict[str, int]] = Field(default_factory=dict)
    last_updated: str = ""
    model_used: str = ""
    taxonomy_hash: str = ""


# --- Client Setup ---

def get_client(base_url: str | None = None):
    """Create async OpenAI client. Uses OpenRouter by default, or a local server if base_url is set."""
    return make_openai_client(base_url=base_url, async_client=True)


def load_prompt() -> str:
    """Load the Phase 04 prompt template."""
    prompt_file = PROMPTS_DIR / "phase_04_labeling.md"
    return prompt_file.read_text()


def format_samples(samples: list[tuple[str, int, str]]) -> str:
    """Format samples with UUIDs."""
    return "\n".join(f"[{sample_uuid}] {text}" for sample_uuid, orig_idx, text in samples)


def get_labeling_generation_kwargs(model: str, *, local: bool) -> dict[str, Any]:
    """Return model-specific sampling settings for Phase 04 labeling."""
    if "qwen" not in (model or "").lower():
        kwargs: dict[str, Any] = {
            "temperature": resolve_temperature(model, 0.0),
        }
        if not local:
            kwargs["extra_body"] = {"provider": {"zdr": True}}
        return kwargs

    extra_body: dict[str, Any] = {"top_k": QWEN_LABELING_TOP_K}
    if not local:
        extra_body["provider"] = {"zdr": True}
    return {
        "temperature": QWEN_LABELING_TEMPERATURE,
        "top_p": QWEN_LABELING_TOP_P,
        "presence_penalty": QWEN_LABELING_PRESENCE_PENALTY,
        "extra_body": extra_body,
    }


def format_retry_error(exc: Exception) -> str:
    """Normalize exception text for concise retry logs."""
    message = " ".join(str(exc).split())
    if len(message) > 240:
        return f"{message[:237]}..."
    return message


async def label_batch(
    client: AsyncOpenAI,
    samples: list[tuple[str, int, str]],  # (uuid, orig_idx, text)
    meta_codes: list[str],
    categories_by_meta: dict[str, list[str]],
    meta_descriptions: dict[str, str],
    model: str,
    domain_context: str,
    language_note: str,
    system_prompt: str,
    request_timeout_seconds: float,
    batch_start: int,
    batch_end: int,
    BatchResultModel: type[BaseModel],
    semaphore: asyncio.Semaphore,
    local: bool = False,
    max_attempts: int = DEFAULT_BATCH_MAX_ATTEMPTS,
):
    """Label a batch of samples."""
    prompt_template = load_prompt()
    meta_list, categories_list = build_prompt_taxonomy_section(
        meta_codes, categories_by_meta, meta_descriptions
    )
    samples_formatted = format_samples(samples)
    prompt = (
        prompt_template
        .replace("{domain_context}", domain_context)
        .replace("{language_note}", language_note)
        .replace("{meta_categories_list}", meta_list)
        .replace("{categories_list}", categories_list)
        .replace("{samples}", samples_formatted)
    )
    generation_kwargs = get_labeling_generation_kwargs(model, local=local)
    sent_uuids = {sample_uuid for sample_uuid, _orig_idx, _text in samples}
    uuid_to_orig = {sample_uuid: (orig_idx, text) for sample_uuid, orig_idx, text in samples}

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

        received_uuids = {sample.sample_uuid for sample in result.samples}
        if len(received_uuids) != len(result.samples):
            raise ValidationRetryError(
                "LLM returned duplicate sample_uuid values",
                failed_result=result,
            )
        missing = sent_uuids - received_uuids
        if missing:
            raise ValidationRetryError(
                f"LLM missed {len(missing)} samples: {missing}",
                failed_result=result,
            )
        extra = received_uuids - sent_uuids
        if extra:
            raise ValidationRetryError(
                f"LLM returned {len(extra)} unknown UUIDs: {extra}",
                failed_result=result,
            )

        batch_results = []
        for sample_result in result.samples:
            orig_idx, text = uuid_to_orig[sample_result.sample_uuid]
            batch_results.append(
                {
                    "original_index": orig_idx,
                    "text": text,
                    "labels": [lbl.model_dump() for lbl in sample_result.labels],
                }
            )

        try:
            validate_meta_category_pairs(batch_results, categories_by_meta, batch_start, batch_end)
        except Exception as exc:
            raise ValidationRetryError(str(exc), failed_result=result) from exc
        return batch_results

    return await run_with_validation_repair(
        model=model,
        operation_label=f"batch {batch_start}-{batch_end}",
        run_attempt=run_attempt,
        max_attempts=max_attempts,
    )


def save_state(state: LabelingState, state_file: Path):
    """Save labeling state."""
    state.last_updated = datetime.now().isoformat()
    with open(state_file, "w") as f:
        json.dump(state.model_dump(), f, indent=2, ensure_ascii=False)


def load_state(state_file: Path) -> LabelingState:
    """Load labeling state."""
    if state_file.exists():
        with open(state_file) as f:
            return LabelingState(**json.load(f))
    return LabelingState()


def resolve_artifact_paths(model: str, reuse_existing: bool) -> tuple[Path, Path, Path, Path]:
    """Resolve state, JSONL, DB, and failed-batch manifest paths for a run."""
    resolver = resolve_provider_path if reuse_existing else provider_scoped_path
    return (
        resolver(CACHE_DIR / "labeling_state.json", model),
        resolver(CACHE_DIR / "labels.jsonl", model),
        resolver(CACHE_DIR / "labels.db", model),
        resolver(CACHE_DIR / "failed_batches.json", model),
    )


def save_results(results: list[dict], results_file: Path, mode: str = "a"):
    """Append results to JSONL file."""
    with open(results_file, mode) as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_failed_batch_entry(
    batch_start: int,
    batch_end: int,
    samples: list[tuple[str, int, str]],
    error: str | None = None,
) -> dict[str, Any]:
    """Create a serializable failed-batch manifest entry."""
    return {
        "batch_start": batch_start,
        "batch_end": batch_end,
        "original_indices": [orig_idx for _sample_uuid, orig_idx, _text in samples],
        "sample_count": len(samples),
        "error": error,
        "last_failed_at": datetime.now().isoformat(),
    }


def load_failed_batches_manifest(path: Path) -> list[dict[str, Any]]:
    """Load failed-batch manifest entries if present."""
    if not path.exists():
        return []
    with open(path) as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        batches = payload.get("failed_batches", [])
    else:
        batches = payload
    return [entry for entry in batches if isinstance(entry, dict)]


def save_failed_batches_manifest(path: Path, batches: list[dict[str, Any]]) -> None:
    """Persist the failed-batch manifest."""
    normalized = sorted(batches, key=lambda entry: (entry.get("batch_start", 0), entry.get("batch_end", 0)))
    with open(path, "w") as f:
        json.dump({"failed_batches": normalized}, f, indent=2, ensure_ascii=False)


def rewrite_results_from_database(conn: sqlite3.Connection, results_file: Path) -> None:
    """Rebuild JSONL from the canonical SQLite DB to avoid duplicates after resume/reruns."""
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

    with open(results_file, "w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# --- SQLite Database ---

def init_database(db_path: Path) -> sqlite3.Connection:
    """Initialize SQLite database with schema."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Samples table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS samples (
            id INTEGER PRIMARY KEY,
            original_index INTEGER UNIQUE,
            text TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Labels table (one row per label, samples can have multiple labels)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS labels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sample_id INTEGER NOT NULL,
            meta TEXT NOT NULL,
            category TEXT NOT NULL,
            reasoning TEXT,
            suggested_category TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (sample_id) REFERENCES samples(id)
        )
    """)

    # Indexes for fast queries
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_labels_meta ON labels(meta)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_labels_category ON labels(category)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_labels_sample ON labels(sample_id)")

    conn.commit()
    return conn


def clear_labels(conn: sqlite3.Connection):
    """Clear all labels from database."""
    cursor = conn.cursor()
    cursor.execute("DELETE FROM labels")
    cursor.execute("DELETE FROM samples")
    conn.commit()


def save_to_database(conn: sqlite3.Connection, results: list[dict]):
    """Save batch results to database."""
    cursor = conn.cursor()

    for r in results:
        # Insert or ignore sample (in case of resume)
        cursor.execute("""
            INSERT OR IGNORE INTO samples (original_index, text)
            VALUES (?, ?)
        """, (r["original_index"], r["text"]))

        # Get sample id
        cursor.execute("SELECT id FROM samples WHERE original_index = ?", (r["original_index"],))
        sample_id = cursor.fetchone()[0]

        # Delete old labels for this sample (in case of re-labeling)
        cursor.execute("DELETE FROM labels WHERE sample_id = ?", (sample_id,))

        # Insert labels
        for lbl in r["labels"]:
            cursor.execute("""
                INSERT INTO labels (sample_id, meta, category, reasoning, suggested_category)
                VALUES (?, ?, ?, ?, ?)
            """, (
                sample_id,
                lbl["meta"],
                lbl["category"],
                lbl["reasoning"],
                lbl.get("suggested_category"),
            ))

    conn.commit()


def validate_meta_category_pairs(
    batch_results: list[dict[str, Any]],
    categories_by_meta: dict[str, list[str]],
    batch_start: int,
    batch_end: int,
) -> None:
    """Validate that each category belongs to the selected meta."""
    for record in batch_results:
        for lbl in record.get("labels", []):
            meta = lbl.get("meta")
            category = lbl.get("category")
            allowed = categories_by_meta.get(meta, [])
            if category not in allowed:
                raise ValueError(
                    "Invalid meta/category pair in "
                    f"batch {batch_start}-{batch_end} (index {record.get('original_index')}): "
                    f"{meta}/{category} not in taxonomy"
                )


def get_labeled_original_indices(conn: sqlite3.Connection) -> set[int]:
    """Return all original sample indices persisted in DB."""
    cursor = conn.cursor()
    cursor.execute("SELECT original_index FROM samples")
    return {row[0] for row in cursor.fetchall()}


def compute_contiguous_processed_index(df: pd.DataFrame, seen_original_indices: set[int]) -> int:
    """
    Return the first row-position in df that is not yet covered.

    This preserves resume safety: if there is a gap, resume starts at the gap.
    """
    contiguous = 0
    for pos, original_index in enumerate(df.index):
        if original_index in seen_original_indices:
            contiguous = pos + 1
        else:
            break
    return contiguous


def derive_missing_batches_from_database(
    df: pd.DataFrame,
    seen_original_indices: set[int],
    batch_size: int,
) -> list[dict[str, Any]]:
    """Fallback for failed-only reruns when no manifest exists yet."""
    missing_positions = [
        pos for pos, original_index in enumerate(df.index)
        if original_index not in seen_original_indices
    ]
    batches: list[dict[str, Any]] = []
    chunk: list[int] = []
    for pos in missing_positions:
        chunk.append(pos)
        if len(chunk) == batch_size:
            original_indices = [int(df.index[idx]) for idx in chunk]
            batches.append(
                {
                    "batch_start": chunk[0],
                    "batch_end": chunk[-1] + 1,
                    "original_indices": original_indices,
                    "sample_count": len(original_indices),
                    "error": "Derived from DB coverage gap",
                    "last_failed_at": datetime.now().isoformat(),
                }
            )
            chunk = []
    if chunk:
        original_indices = [int(df.index[idx]) for idx in chunk]
        batches.append(
            {
                "batch_start": chunk[0],
                "batch_end": chunk[-1] + 1,
                "original_indices": original_indices,
                "sample_count": len(original_indices),
                "error": "Derived from DB coverage gap",
                "last_failed_at": datetime.now().isoformat(),
            }
        )
    return batches


def refresh_state_from_database(conn: sqlite3.Connection, state: LabelingState) -> None:
    """Rebuild state aggregates from DB to avoid drift during resume/retries."""
    cursor = conn.cursor()

    category_counts: dict[str, dict[str, int]] = {}
    cursor.execute("""
        SELECT meta, category, COUNT(*)
        FROM labels
        GROUP BY meta, category
    """)
    for meta, category, count in cursor.fetchall():
        category_counts.setdefault(meta, {})[category] = count

    other_samples: list[dict[str, Any]] = []
    cursor.execute("""
        SELECT s.original_index, s.text, l.meta, l.suggested_category, l.reasoning
        FROM labels l
        JOIN samples s ON s.id = l.sample_id
        WHERE l.category = 'other'
        ORDER BY s.original_index ASC, l.id ASC
    """)
    for original_index, text, meta, suggested, reasoning in cursor.fetchall():
        other_samples.append({
            "index": original_index,
            "text": text,
            "meta": meta,
            "suggested": suggested,
            "reasoning": reasoning,
        })

    state.category_counts = category_counts
    state.other_samples = other_samples


async def main(
    sample_mode: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
    resume: bool = False,
    rerun_missing_only: bool = False,
    model: str = DEFAULT_MODEL,
    taxonomy_model: str = DEFAULT_TAXONOMY_MODEL,
    other_threshold: int = DEFAULT_OTHER_THRESHOLD,
    concurrency: int | None = None,
    domain_context: str = DEFAULT_DOMAIN_CONTEXT,
    language_note: str = DEFAULT_LANGUAGE_NOTE,
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    base_url: str | None = None,
    run_id: str | None = None,
):
    """Main labeling function."""

    configure_paths(run_id)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if resume and rerun_missing_only:
        raise ValueError("Use either --resume or --rerun-missing-only, not both")

    reuse_existing = resume or rerun_missing_only
    state_file, results_file, db_file, failed_batches_file = resolve_artifact_paths(model, reuse_existing)
    print(f"Artifact scope: {DATA_DIR}")

    # Load taxonomy
    print("Loading taxonomy from Phase 00-03...")
    try:
        meta_codes, categories_by_meta, meta_descriptions = load_taxonomy(taxonomy_model)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return

    print(f"Meta-categories: {meta_codes}")
    total_cats = sum(len(cats) for cats in categories_by_meta.values())
    print(f"Total categories: {total_cats}")

    # Create dynamic Pydantic models with Literal types
    print("Creating response models with Literal constraints...")
    Label, SampleResult, BatchResult = create_label_models(meta_codes, categories_by_meta)

    # Load data
    print(f"\nLoading data from {INPUT_DATA}...")
    df = pd.read_parquet(INPUT_DATA)
    print(f"Loaded {len(df)} samples")

    taxonomy_hash = compute_taxonomy_hash(meta_codes, categories_by_meta)

    # Load or initialize state
    if reuse_existing and state_file.exists():
        state = load_state(state_file)
        if resume:
            print(f"Resuming from index {state.last_processed_index}")
        else:
            print("Loading existing labeling state for failed-only rerun.")
        if state.taxonomy_hash and state.taxonomy_hash != taxonomy_hash:
            raise RuntimeError(
                "Resume/rerun blocked: taxonomy has changed since previous run. "
                "Run without --resume/--rerun-missing-only or restore the original taxonomy."
            )
        if not state.taxonomy_hash:
            print("WARNING: Legacy state has no taxonomy hash; cannot verify prior taxonomy consistency.")
    elif reuse_existing:
        if resume:
            print("Resume requested, but no state file found. Initializing resume state from database.")
        else:
            print("Failed-only rerun requested, but no state file found. Initializing state from database.")
        state = LabelingState(model_used=model, taxonomy_hash=taxonomy_hash)
    else:
        state = LabelingState(
            total_samples=len(df),
            model_used=model,
            taxonomy_hash=taxonomy_hash,
        )
        if results_file.exists():
            results_file.unlink()

    # Sample mode
    if sample_mode:
        df = df.head(10)
        batch_size = 10
        print(f"=== SAMPLE MODE: {len(df)} samples ===")

    # Initialize client and database
    client, local = get_client(base_url=base_url)
    concurrency = resolve_concurrency(concurrency, local, model=model)
    db_conn = init_database(db_file)

    try:
        # Clear database if not reusing persisted artifacts
        if not reuse_existing:
            clear_labels(db_conn)
            print(f"Database cleared: {db_file}")
            state = LabelingState(
                total_samples=len(df),
                model_used=model,
                taxonomy_hash=taxonomy_hash,
            )
            save_state(state, state_file)
            save_failed_batches_manifest(failed_batches_file, [])
        else:
            print(f"Database initialized: {db_file}")
            state.total_samples = len(df)
            state.model_used = model
            state.taxonomy_hash = taxonomy_hash
            refresh_state_from_database(db_conn, state)
            seen_for_resume = get_labeled_original_indices(db_conn)
            if resume:
                resume_index = compute_contiguous_processed_index(df, seen_for_resume)
                if resume_index != state.last_processed_index:
                    print(
                        "Resume index adjusted from "
                        f"{state.last_processed_index} to {resume_index} based on DB coverage."
                    )
                    state.last_processed_index = resume_index
            else:
                state.last_processed_index = compute_contiguous_processed_index(df, seen_for_resume)
            save_state(state, state_file)

        # Create semaphore for concurrency control and lock for DB writes
        semaphore = asyncio.Semaphore(concurrency)
        db_lock = asyncio.Lock()
        print(f"Concurrency: {concurrency} parallel requests")

        # Prepare all batches
        total = len(df)
        start_idx = min(max(state.last_processed_index, 0), total)

        failed_batches_manifest: dict[int, dict[str, Any]] = {
            int(entry.get("batch_start", 0)): entry
            for entry in load_failed_batches_manifest(failed_batches_file)
        }

        batches = []
        if rerun_missing_only:
            seen_indices = get_labeled_original_indices(db_conn)
            failed_entries = list(failed_batches_manifest.values())
            if not failed_entries:
                failed_entries = derive_missing_batches_from_database(df, seen_indices, batch_size)
                if failed_entries:
                    print("No failed-batch manifest found; derived rerun batches from DB coverage gaps.")
                    failed_batches_manifest = {
                        int(entry["batch_start"]): entry for entry in failed_entries
                    }

            for entry in sorted(failed_entries, key=lambda item: item.get("batch_start", 0)):
                original_indices = [idx for idx in entry.get("original_indices", []) if idx in df.index]
                if not original_indices:
                    failed_batches_manifest.pop(int(entry.get("batch_start", 0)), None)
                    continue
                missing_for_entry = [idx for idx in original_indices if idx not in seen_indices]
                if not missing_for_entry:
                    failed_batches_manifest.pop(int(entry.get("batch_start", 0)), None)
                    continue
                batch_df = df.loc[missing_for_entry]
                samples = [(str(uuid.uuid4()), row.name, row["text"]) for _, row in batch_df.iterrows()]
                batch_start = int(entry.get("batch_start", 0))
                batch_end = int(entry.get("batch_end", batch_start + len(samples)))
                batches.append((batch_start, batch_end, samples))

            save_failed_batches_manifest(failed_batches_file, list(failed_batches_manifest.values()))
            print(f"\nRe-running {len(batches)} failed/missing batches only...")
        else:
            failed_batches_manifest = {}
            save_failed_batches_manifest(failed_batches_file, [])
            for batch_start in range(start_idx, total, batch_size):
                batch_end = min(batch_start + batch_size, total)
                batch_df = df.iloc[batch_start:batch_end]
                samples = [(str(uuid.uuid4()), row.name, row["text"]) for _, row in batch_df.iterrows()]
                batches.append((batch_start, batch_end, samples))

            print(f"\nProcessing {len(batches)} batches...")

        if not batches:
            print("No batches to process.")
            rewrite_results_from_database(db_conn, results_file)
            refresh_state_from_database(db_conn, state)
            save_state(state, state_file)
            save_failed_batches_manifest(failed_batches_file, [])
            return

        completed = [0]  # Mutable counter for progress
        successful_batch_starts: set[int] = set()

        def contiguous_index_from_successful_batches() -> int:
            contiguous = start_idx
            for batch_start, batch_end, _ in batches:
                if batch_start in successful_batch_starts:
                    contiguous = batch_end
                else:
                    break
            return contiguous

        async def process_batch(batch_info: tuple) -> None:
            """Process a single batch and persist immediately."""
            batch_start, batch_end, samples = batch_info

            try:
                batch_results = await label_batch(
                    client, samples, meta_codes, categories_by_meta,
                    meta_descriptions, model, domain_context, language_note, system_prompt,
                    request_timeout_seconds, batch_start, batch_end, BatchResult, semaphore,
                    local=local,
                )
            except Exception as exc:
                async with db_lock:
                    failed_batches_manifest[batch_start] = build_failed_batch_entry(
                        batch_start,
                        batch_end,
                        samples,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    save_failed_batches_manifest(failed_batches_file, list(failed_batches_manifest.values()))
                    completed[0] += 1
                    print(f"  [{completed[0]}/{len(batches)}] FAILED Batch {batch_start}-{batch_end}: {exc}")
                raise

            # Persist immediately with lock
            async with db_lock:
                save_results(batch_results, results_file)
                save_to_database(db_conn, batch_results)

                successful_batch_starts.add(batch_start)
                failed_batches_manifest.pop(batch_start, None)
                save_failed_batches_manifest(failed_batches_file, list(failed_batches_manifest.values()))
                if rerun_missing_only:
                    state.last_processed_index = compute_contiguous_processed_index(
                        df,
                        get_labeled_original_indices(db_conn),
                    )
                else:
                    state.last_processed_index = contiguous_index_from_successful_batches()
                refresh_state_from_database(db_conn, state)
                save_state(state, state_file)

                completed[0] += 1
                print(f"  [{completed[0]}/{len(batches)}] Batch {batch_start}-{batch_end}: {len(batch_results)} samples")

        # Process all batches concurrently
        tasks = [process_batch(b) for b in batches]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check for task-level errors first
        errors = [r for r in results if isinstance(r, Exception)]
        if errors:
            print(f"\n{len(errors)} batches failed:")
            for err in errors[:5]:
                print(f"  - {err}")

        # Final integrity check: persisted coverage must match input coverage
        seen_indices = get_labeled_original_indices(db_conn)
        missing_original_indices = [idx for idx in df.index if idx not in seen_indices]

        state.last_processed_index = compute_contiguous_processed_index(df, seen_indices)
        refresh_state_from_database(db_conn, state)
        rewrite_results_from_database(db_conn, results_file)
        save_state(state, state_file)
        save_failed_batches_manifest(failed_batches_file, list(failed_batches_manifest.values()))

        if errors or missing_original_indices:
            if missing_original_indices:
                print(
                    "Coverage gap detected: "
                    f"{len(missing_original_indices)} samples missing, first examples: "
                    f"{missing_original_indices[:10]}"
                )
            raise RuntimeError(
                "Labeling failed integrity checks. "
                f"Saved resumable state at index {state.last_processed_index}."
            )

        # Final summary
        print("\n" + "=" * 60)
        print("LABELING COMPLETE")
        print("=" * 60)
        print(f"Total samples processed: {state.last_processed_index}")

        print("\nCategory distribution by meta-category:")
        for meta, cats in state.category_counts.items():
            print(f"\n  {meta}:")
            for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
                print(f"    {cat}: {count}")

        if state.other_samples:
            print(f"\n'Other' samples for review: {len(state.other_samples)}")

        print(f"\nResults saved to:")
        print(f"  JSONL: {results_file}")
        print(f"  SQLite: {db_file}")
        print(f"State saved to: {state_file}")
        print(f"Failed batch manifest: {failed_batches_file}")
    finally:
        db_conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 04: Sample Labeling")
    parser.add_argument("--sample", action="store_true", help="Run with 10 samples for testing")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")
    parser.add_argument("--resume", action="store_true", help="Resume from last state")
    parser.add_argument("--rerun-missing-only", action="store_true", help="Process only failed/missing batches from the failed-batch manifest or DB coverage gaps")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model to use")
    parser.add_argument("--taxonomy-model", type=str, default=DEFAULT_TAXONOMY_MODEL, help="Model used to generate the Phase 00-03 taxonomy")
    parser.add_argument("--other-threshold", type=int, default=DEFAULT_OTHER_THRESHOLD, help="Threshold for 'other' review")
    parser.add_argument("--concurrency", type=int, default=None, help="Parallel API calls (defaults: Qwen 27B=8, Qwen 9B=20, otherwise 10; local=1)")
    parser.add_argument("--domain-context", type=str, default=DEFAULT_DOMAIN_CONTEXT, help="Domain context")
    parser.add_argument("--language-note", type=str, default=DEFAULT_LANGUAGE_NOTE, help="Language note")
    parser.add_argument("--request-timeout-seconds", type=float, default=DEFAULT_REQUEST_TIMEOUT_SECONDS, help="Per-attempt timeout for a single batch request")
    parser.add_argument("--system-prompt", type=str, default=DEFAULT_SYSTEM_PROMPT, help="System prompt for labeling")
    parser.add_argument("--base-url", type=str, default=None, help="Custom API base URL (e.g. LM Studio)")
    parser.add_argument("--run-id", type=str, default=None, help="Optional run id for isolated artifacts under data/runs/<run_id>/")

    args = parser.parse_args()
    asyncio.run(main(
        sample_mode=args.sample,
        batch_size=args.batch_size,
        resume=args.resume,
        rerun_missing_only=args.rerun_missing_only,
        model=args.model,
        taxonomy_model=args.taxonomy_model,
        other_threshold=args.other_threshold,
        concurrency=args.concurrency,
        domain_context=args.domain_context,
        language_note=args.language_note,
        request_timeout_seconds=args.request_timeout_seconds,
        system_prompt=args.system_prompt,
        base_url=args.base_url,
        run_id=args.run_id,
    ))
