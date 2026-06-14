"""
Phase 01: Meta-Category Consolidation

Consolidate multiple meta-category discovery runs into a single robust set.

Usage:
    uv run python scripts/phase_01_meta_consolidation.py --sample
    uv run python scripts/phase_01_meta_consolidation.py
    uv run python scripts/phase_01_meta_consolidation.py --max-metas 6
"""

import argparse
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

from lib.experiment_paths import resolve_experiment_artifact_paths
from lib.llm_client import make_openai_client, local_structured_call, local_structured_call_sync, extract_pydantic_json, structured_parse_call_sync
from lib.model_naming import get_provider_name, provider_scoped_path, resolve_temperature

load_dotenv()

EXPERIMENT_DIR = Path(__file__).parent
PROMPTS_DIR = EXPERIMENT_DIR / "prompts"
ARTIFACT_PATHS = resolve_experiment_artifact_paths(EXPERIMENT_DIR)
DATA_DIR = ARTIFACT_PATHS.data_dir
INTERMEDIATE_DIR = ARTIFACT_PATHS.intermediate_dir

DEFAULT_MODEL = "openai/gpt-5.2"
DEFAULT_MAX_METAS = 6
DEFAULT_TARGET = "adverse_events"
DEFAULT_DOMAIN_CONTEXT = "Free-text entries from stroke intervention documentation."
DEFAULT_REQUEST_TIMEOUT_SECONDS = 180.0


def configure_paths(run_id: str | None) -> None:
    global ARTIFACT_PATHS, DATA_DIR, INTERMEDIATE_DIR
    ARTIFACT_PATHS = resolve_experiment_artifact_paths(EXPERIMENT_DIR, run_id=run_id)
    DATA_DIR = ARTIFACT_PATHS.data_dir
    INTERMEDIATE_DIR = ARTIFACT_PATHS.intermediate_dir


class MetaCluster(BaseModel):
    code: str = Field(description="Canonical meta-category code in snake_case")
    description: str = Field(description="Brief description of the concept")
    is_target: bool = Field(description="True if this meta represents the target concept")
    members: list[str] = Field(description="Run-specific members like 'seed_42:complication'")


class MetaConsolidationResult(BaseModel):
    reasoning: str = Field(description="Summary of consolidation decisions")
    clusters: list[MetaCluster]


def get_client(base_url: str | None = None):
    return make_openai_client(base_url=base_url, async_client=False)


def load_prompt() -> str:
    prompt_file = PROMPTS_DIR / "phase_01_meta_consolidation.md"
    return prompt_file.read_text()


def load_runs(input_dir: Path, model: str) -> list[dict]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Meta runs directory not found: {input_dir}")

    provider = get_provider_name(model)
    provider_pattern = re.compile(rf"meta_seed_\d+_{re.escape(provider)}\.json$")
    legacy_pattern = re.compile(r"meta_seed_\d+\.json$")

    provider_paths = sorted(
        path for path in input_dir.iterdir()
        if path.is_file() and provider_pattern.fullmatch(path.name)
    )
    legacy_paths = sorted(
        path for path in input_dir.iterdir()
        if path.is_file() and legacy_pattern.fullmatch(path.name)
    )
    paths = provider_paths or legacy_paths

    runs = []
    for path in paths:
        with open(path) as f:
            data = json.load(f)
        runs.append(data)

    if not runs:
        raise FileNotFoundError(f"No meta runs found in {input_dir}")
    return runs


def format_runs_for_prompt(runs: list[dict]) -> str:
    lines = []
    for run in runs:
        seed = run.get("seed", "unknown")
        run_id = f"seed_{seed}"
        lines.append(f"## {run_id}")
        for mc in run.get("meta_categories", []):
            code = mc["code"]
            desc = mc.get("description", "")
            is_target = mc.get("is_target", False)
            target_marker = " target" if is_target else ""
            lines.append(f"- {run_id}:{code}{target_marker}: {desc}")
        lines.append("")
    return "\n".join(lines).strip()


def consolidate(
    client: OpenAI,
    runs: list[dict],
    target: str,
    domain_context: str,
    model: str,
    max_metas: int,
    local: bool = False,
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
) -> MetaConsolidationResult:
    prompt_template = load_prompt()
    runs_formatted = format_runs_for_prompt(runs)

    prompt = (
        prompt_template
        .replace("{target}", target)
        .replace("{domain_context}", domain_context)
        .replace("{max_metas}", str(max_metas))
        .replace("{runs}", runs_formatted)
    )

    if local:
        return local_structured_call_sync(client, model, prompt, MetaConsolidationResult)

    return structured_parse_call_sync(
        client,
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format=MetaConsolidationResult,
        extra_body={
            "provider": {"zdr": True},
            "reasoning": {"effort": "high"},
        },
        temperature=resolve_temperature(model, 0.6),
        request_timeout_seconds=request_timeout_seconds,
    )


def compute_support(members: list[str]) -> int:
    run_ids = set()
    for m in members:
        run_id = m.split(":", 1)[0].strip()
        if run_id:
            run_ids.add(run_id)
    return len(run_ids)


def main(
    sample_mode: bool = False,
    input_dir: Path = None,
    output_file: Path = None,
    target: str = DEFAULT_TARGET,
    domain_context: str = DEFAULT_DOMAIN_CONTEXT,
    model: str = DEFAULT_MODEL,
    max_metas: int = DEFAULT_MAX_METAS,
    base_url: str | None = None,
    run_id: str | None = None,
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
):
    configure_paths(run_id)
    if input_dir is None:
        input_dir = INTERMEDIATE_DIR / "meta_runs"
    if output_file is None:
        output_file = provider_scoped_path(INTERMEDIATE_DIR / "meta_categories_consolidated.json", model)
    print(f"Artifact scope: {DATA_DIR}")

    runs = load_runs(input_dir, model)

    if sample_mode:
        runs = runs[:2]
        print(f"=== SAMPLE MODE: consolidating {len(runs)} runs ===")

    client, local = get_client(base_url=base_url)
    result = consolidate(
        client,
        runs,
        target,
        domain_context,
        model,
        max_metas,
        local=local,
        request_timeout_seconds=request_timeout_seconds,
    )

    consolidated = []
    for cluster in result.clusters:
        consolidated.append({
            "code": cluster.code,
            "description": cluster.description,
            "is_target": cluster.is_target,
            "support": compute_support(cluster.members),
            "members": cluster.members,
        })

    # Ensure "other" exists
    if "other" not in [c["code"] for c in consolidated]:
        consolidated.append({
            "code": "other",
            "description": "Catch-all for unclassifiable samples",
            "is_target": False,
            "support": 0,
            "members": [],
        })

    if sample_mode:
        print("Sample mode - not saving results")
        return result

    output_data = {
        "target": target,
        "domain_context": domain_context,
        "model": model,
        "max_metas": max_metas,
        "runs": [r.get("seed") for r in runs],
        "reasoning": result.reasoning,
        "meta_categories": consolidated,
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"Saved consolidated meta-categories to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 01: Meta-Category Consolidation")
    parser.add_argument("--sample", action="store_true", help="Sample mode (no save)")
    parser.add_argument("--input-dir", type=Path, default=None, help="Directory with meta runs")
    parser.add_argument("--output-file", type=Path, default=None, help="Output JSON file")
    parser.add_argument("--target", type=str, default=DEFAULT_TARGET, help="Target description")
    parser.add_argument("--domain-context", type=str, default=DEFAULT_DOMAIN_CONTEXT, help="Domain context")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model to use")
    parser.add_argument("--max-metas", type=int, default=DEFAULT_MAX_METAS, help="Max meta-categories")
    parser.add_argument("--base-url", type=str, default=None, help="Custom API base URL (e.g. LM Studio)")
    parser.add_argument("--run-id", type=str, default=None, help="Optional run id for isolated artifacts under data/runs/<run_id>/")
    parser.add_argument("--request-timeout-seconds", type=float, default=DEFAULT_REQUEST_TIMEOUT_SECONDS, help="Per-attempt timeout for a single request")

    args = parser.parse_args()
    main(
        sample_mode=args.sample,
        input_dir=args.input_dir,
        output_file=args.output_file,
        target=args.target,
        domain_context=args.domain_context,
        model=args.model,
        max_metas=args.max_metas,
        base_url=args.base_url,
        run_id=args.run_id,
    )
