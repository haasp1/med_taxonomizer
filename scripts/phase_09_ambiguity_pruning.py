"""
Phase 09: Ambiguity Pruning

Two-step ambiguity pruning for the final taxonomy tree:
1. candidate discovery
2. candidate selection / keep-remove resolution

The phase runs one in-meta scope for the target meta and one cross-meta scope
that includes the target meta plus comparison metas.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
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

load_dotenv()

EXPERIMENT_DIR = Path(__file__).parent
PROMPTS_DIR = EXPERIMENT_DIR / "prompts"
ARTIFACT_PATHS = resolve_experiment_artifact_paths(EXPERIMENT_DIR)
DATA_DIR = ARTIFACT_PATHS.data_dir
INTERMEDIATE_DIR = ARTIFACT_PATHS.intermediate_dir

DEFAULT_MODEL = "openai/gpt-5.2"
DEFAULT_INPUT_MODEL = "openai/gpt-5.2"
DEFAULT_DOMAIN_CONTEXT = "Free-text entries from stroke intervention documentation."
DEFAULT_TARGET = "adverse_events"
DEFAULT_CONCURRENCY = 8
DEFAULT_REQUEST_TIMEOUT_SECONDS = 300.0
DEFAULT_BATCH_MAX_ATTEMPTS = 4
DISCOVERY_PROMPT_NAMES = (
    "phase_09_candidate_discovery.md",
    "phase_09_candidate_discovery_anchor.md",
)
DEFAULT_SYSTEM_PROMPT = (
    "You are a concise, efficient, decisive assistant. "
    "Think in 2-3 short blocks per sample without repetition or second-guessing, "
    "and then output your answer."
)


def configure_paths(run_id: str | None) -> None:
    global ARTIFACT_PATHS, DATA_DIR, INTERMEDIATE_DIR
    ARTIFACT_PATHS = resolve_experiment_artifact_paths(EXPERIMENT_DIR, run_id=run_id)
    DATA_DIR = ARTIFACT_PATHS.data_dir
    INTERMEDIATE_DIR = ARTIFACT_PATHS.intermediate_dir


class CandidateGroup(BaseModel):
    concern: str = Field(description="Short phrase naming the possible ambiguous concern")
    node_ids: list[str] = Field(description="Two or more existing node ids that may compete during labeling")


class CandidateDiscoveryResult(BaseModel):
    candidates: list[CandidateGroup] = Field(description="Possible ambiguity groups")


class CandidateSelectionResult(BaseModel):
    should_prune: bool = Field(description="Whether this candidate should become a pruning decision")
    keep_node_id: str | None = Field(default=None, description="Canonical node id to keep if selected")
    remove_node_ids: list[str] = Field(default_factory=list, description="Competing node ids to remove if selected")

class TaxonomyPruneDecision(BaseModel):
    concern: str = Field(description="Short noun phrase for the competing concern")
    keep_node_id: str = Field(description="Stable node id to retain as canonical owner")
    remove_node_ids: list[str] = Field(description="Stable node ids to prune")


class PromptNodeRef(BaseModel):
    node_id: str
    meta: str
    code: str
    description: str
    path: str
    depth: int
    top_root_code: str
    parent_node_id: str | None
    child_node_ids: list[str]
    descendant_node_ids: list[str]


@dataclass(frozen=True)
class ScopeSpec:
    scope_id: str
    scope_type: str
    metas: tuple[str, ...]
    included_root_codes: tuple[str, ...] | None = None


def get_client(base_url: str | None = None):
    return make_openai_client(base_url=base_url, async_client=True)


def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text()


def load_final_tree(model: str) -> dict[str, Any]:
    tree_file = resolve_provider_path(INTERMEDIATE_DIR / "taxonomy_tree_final.json", model)
    if not tree_file.exists():
        raise FileNotFoundError("taxonomy_tree_final.json not found")
    with open(tree_file) as f:
        return json.load(f)


def get_generation_kwargs(model: str, *, local: bool, default_temperature: float) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "temperature": resolve_temperature(model, default_temperature),
    }
    if not local:
        kwargs["extra_body"] = {"provider": {"zdr": True}}
    return kwargs


def _coerce_path(value: Any) -> str:
    path = str(value or "").strip().strip("/")
    if not path:
        return ""
    parts = path.split("/")
    if any(not part for part in parts):
        return ""
    return "/".join(parts)


def _path_rows_from_nodes_meta(meta_entry: dict[str, Any]) -> list[dict[str, str]]:
    node_map = {
        node["code"]: {
            "description": str(node.get("description") or ""),
            "children": [child for child in (node.get("children") or []) if isinstance(child, str)],
        }
        for node in meta_entry.get("nodes", []) or []
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

    for root in meta_entry.get("roots", []) or []:
        if isinstance(root, str):
            walk(root)
    for code in node_map:
        walk(code)
    return rows


def canonical_path_meta(meta_entry: dict[str, Any]) -> dict[str, Any]:
    if meta_entry.get("paths") is not None:
        return {
            **meta_entry,
            "paths": [row for row in meta_entry.get("paths", []) or [] if isinstance(row, dict)],
        }
    if meta_entry.get("nodes") is not None or meta_entry.get("roots") is not None:
        return {**meta_entry, "paths": _path_rows_from_nodes_meta(meta_entry)}
    return {**meta_entry, "paths": []}


def validate_path_taxonomy_tree(tree: dict[str, Any]) -> None:
    metas = tree.get("metas", [])
    if not isinstance(metas, list) or not metas:
        raise ValueError("Invalid taxonomy tree: missing non-empty metas list")

    for meta_entry in metas:
        if not isinstance(meta_entry, dict):
            raise ValueError("Invalid taxonomy tree: each meta entry must be an object")
        meta = meta_entry.get("meta")
        if not isinstance(meta, str) or not meta:
            raise ValueError("Invalid taxonomy tree: each meta entry needs a non-empty meta")

        path_entry = canonical_path_meta(meta_entry)
        path_rows = path_entry.get("paths", [])
        if not isinstance(path_rows, list) or not path_rows:
            raise ValueError(f"Invalid taxonomy tree for {meta}: missing non-empty paths list")

        paths: list[str] = []
        seen: set[str] = set()
        for row in path_rows:
            if not isinstance(row, dict):
                raise ValueError(f"Invalid taxonomy tree for {meta}: each path entry must be an object")
            path = _coerce_path(row.get("path"))
            if not path:
                raise ValueError(f"Invalid taxonomy tree for {meta}: empty or malformed path")
            if path in seen:
                raise ValueError(f"Duplicate path for {meta}: {path}")
            seen.add(path)
            paths.append(path)

        path_set = set(paths)
        for path in paths:
            if "/" not in path:
                continue
            parent = path.rsplit("/", 1)[0]
            if parent not in path_set:
                raise ValueError(f"Missing parent path for {meta}: {path} -> {parent}")
        if not any("/" not in path for path in paths):
            raise ValueError(f"Invalid taxonomy tree for {meta}: no root paths")


def meta_paths_to_tree(meta_entry: dict[str, Any]) -> dict[str, Any]:
    path_entry = canonical_path_meta(meta_entry)
    path_rows = path_entry.get("paths", [])
    rows_by_path: dict[str, dict[str, str]] = {}
    children_by_parent: dict[str, list[str]] = {}
    roots: list[str] = []

    for row in path_rows:
        path = _coerce_path(row.get("path"))
        if not path:
            continue
        rows_by_path[path] = {"path": path, "description": str(row.get("description") or "")}

    for path in rows_by_path:
        if "/" not in path:
            roots.append(path)
            continue
        parent = path.rsplit("/", 1)[0]
        children_by_parent.setdefault(parent, []).append(path)

    nodes = [
        {
            "code": path,
            "description": row.get("description", ""),
            "children": children_by_parent.get(path, []),
        }
        for path, row in rows_by_path.items()
    ]
    return {**path_entry, "roots": roots, "nodes": nodes}


def normalize_tree_for_phase09(tree: dict[str, Any]) -> dict[str, Any]:
    return {
        **tree,
        "metas": [canonical_path_meta(meta) for meta in tree.get("metas", [])],
    }


def build_prompt_index(
    metas: list[dict[str, Any]],
) -> tuple[dict[str, PromptNodeRef], dict[str, list[str]], dict[str, set[str]]]:
    refs: dict[str, PromptNodeRef] = {}
    ordered_node_ids_by_meta: dict[str, list[str]] = {}
    ids_by_meta: dict[str, set[str]] = {}
    node_counter = 1

    for raw_meta_entry in metas:
        meta_entry = meta_paths_to_tree(raw_meta_entry)
        meta = meta_entry["meta"]
        roots = meta_entry.get("roots", [])
        node_map = {node["code"]: node for node in meta_entry.get("nodes", [])}
        children_map = {code: list(node.get("children", [])) for code, node in node_map.items()}
        ordered_node_ids: list[str] = []
        meta_ids: set[str] = set()

        def walk(
            code: str,
            path_parts: list[str],
            depth: int,
            parent_node_id: str | None,
            top_root_code: str,
        ) -> tuple[str, list[str]]:
            nonlocal node_counter
            node = node_map[code]
            node_id = f"N{node_counter:03d}"
            node_counter += 1
            path = " > ".join(path_parts + [code])
            description = " ".join((node.get("description") or "").split())
            ordered_node_ids.append(node_id)
            meta_ids.add(node_id)

            child_node_ids: list[str] = []
            descendant_node_ids: list[str] = []
            for child_code in children_map.get(code, []):
                child_node_id, child_descendant_node_ids = walk(
                    child_code,
                    path_parts + [code],
                    depth + 1,
                    node_id,
                    top_root_code,
                )
                child_node_ids.append(child_node_id)
                descendant_node_ids.append(child_node_id)
                descendant_node_ids.extend(child_descendant_node_ids)

            refs[node_id] = PromptNodeRef(
                node_id=node_id,
                meta=meta,
                code=code,
                description=description,
                path=path,
                depth=depth,
                top_root_code=top_root_code,
                parent_node_id=parent_node_id,
                child_node_ids=child_node_ids,
                descendant_node_ids=descendant_node_ids,
            )
            return node_id, descendant_node_ids

        for root_code in roots:
            if root_code in node_map:
                walk(root_code, [], 0, None, root_code)

        ordered_node_ids_by_meta[meta] = ordered_node_ids
        ids_by_meta[meta] = meta_ids

    return refs, ordered_node_ids_by_meta, ids_by_meta


def render_scope_tree(
    scope: ScopeSpec,
    refs: dict[str, PromptNodeRef],
    ordered_node_ids_by_meta: dict[str, list[str]],
) -> str:
    lines: list[str] = []
    for meta in scope.metas:
        lines.append(f"### {meta}")
        for node_id in ordered_node_ids_by_meta.get(meta, []):
            ref = refs[node_id]
            if scope.included_root_codes and ref.top_root_code not in scope.included_root_codes:
                continue
            desc_suffix = f" :: {ref.description}" if ref.description else ""
            indent = "  " * ref.depth
            lines.append(f"{indent}- [{ref.node_id}] {ref.code}{desc_suffix}")
        lines.append("")
    return "\n".join(lines).strip()


def allowed_node_ids_for_scope(
    scope: ScopeSpec,
    refs: dict[str, PromptNodeRef],
    ids_by_meta: dict[str, set[str]],
) -> set[str]:
    allowed: set[str] = set()
    for meta in scope.metas:
        allowed.update(ids_by_meta.get(meta, set()))
    if scope.included_root_codes is None:
        return allowed
    return {
        node_id
        for node_id in allowed
        if refs[node_id].top_root_code in scope.included_root_codes
    }
    return allowed


def ancestors_of(node_id: str, parent_by_id: dict[str, str | None]) -> list[str]:
    ancestors: list[str] = []
    current = parent_by_id.get(node_id)
    while current is not None:
        ancestors.append(current)
        current = parent_by_id.get(current)
    return ancestors


def normalize_remove_ids(
    remove_ids: list[str],
    parent_by_id: dict[str, str | None],
) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for node_id in remove_ids:
        if node_id in seen:
            continue
        seen.add(node_id)
        deduped.append(node_id)

    remove_set = set(deduped)
    normalized: list[str] = []
    for node_id in deduped:
        if any(ancestor in remove_set for ancestor in ancestors_of(node_id, parent_by_id)):
            continue
        normalized.append(node_id)
    return normalized


def expand_removed_ids(remove_ids: list[str], refs: dict[str, PromptNodeRef]) -> set[str]:
    removed: set[str] = set()
    for node_id in remove_ids:
        removed.add(node_id)
        removed.update(refs[node_id].descendant_node_ids)
    return removed


def build_parent_by_id(refs: dict[str, PromptNodeRef]) -> dict[str, str | None]:
    return {node_id: ref.parent_node_id for node_id, ref in refs.items()}


def dedupe_candidate_groups(candidates: list[CandidateGroup]) -> list[CandidateGroup]:
    deduped: list[CandidateGroup] = []
    seen_keys: set[tuple[str, ...]] = set()
    for candidate in candidates:
        node_ids = []
        seen_ids: set[str] = set()
        for node_id in candidate.node_ids:
            if node_id in seen_ids:
                continue
            seen_ids.add(node_id)
            node_ids.append(node_id)
        if len(node_ids) < 2:
            continue
        key = tuple(sorted(node_ids))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append(CandidateGroup(concern=candidate.concern, node_ids=node_ids))
    return deduped


def validate_candidate_discovery_result(
    result: CandidateDiscoveryResult,
    allowed_node_ids: set[str],
) -> list[dict[str, Any]]:
    normalized = dedupe_candidate_groups(result.candidates)
    candidate_payloads: list[dict[str, Any]] = []
    for idx, candidate in enumerate(normalized, start=1):
        for node_id in candidate.node_ids:
            if node_id not in allowed_node_ids:
                raise ValidationRetryError(
                    f"Unknown or out-of-scope node id in candidates: {node_id}",
                    failed_result=result,
                )
        candidate_payloads.append(
            {
                "candidate_id": f"C{idx:03d}",
                "concern": candidate.concern,
                "node_ids": candidate.node_ids,
            }
        )
    return candidate_payloads


def dedupe_candidate_payloads(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, ...]] = set()
    for candidate in candidates:
        node_ids = []
        seen_ids: set[str] = set()
        for node_id in candidate["node_ids"]:
            if node_id in seen_ids:
                continue
            seen_ids.add(node_id)
            node_ids.append(node_id)
        if len(node_ids) < 2:
            continue
        key = tuple(sorted(node_ids))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append(
            {
                "candidate_id": candidate["candidate_id"],
                "concern": candidate["concern"],
                "node_ids": node_ids,
            }
        )
    for idx, candidate in enumerate(deduped, start=1):
        candidate["candidate_id"] = f"C{idx:03d}"
    return deduped


def normalize_scope_candidates(
    scope_report: dict[str, Any],
    refs: dict[str, PromptNodeRef],
    target: str,
) -> dict[str, Any]:
    candidates = scope_report.get("candidates", [])
    if scope_report.get("scope_type") != "cross_meta":
        return scope_report

    normalized_candidates: list[dict[str, Any]] = []
    for candidate in candidates:
        target_ids = [node_id for node_id in candidate["node_ids"] if refs[node_id].meta == target]
        other_ids = [node_id for node_id in candidate["node_ids"] if refs[node_id].meta != target]
        if not target_ids or not other_ids:
            continue
        if len(target_ids) == 1:
            normalized_candidates.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "concern": candidate["concern"],
                    "node_ids": [target_ids[0], *other_ids],
                }
            )
            continue
        for split_idx, target_id in enumerate(target_ids, start=1):
            normalized_candidates.append(
                {
                    "candidate_id": f"{candidate['candidate_id']}_T{split_idx}",
                    "concern": candidate["concern"],
                    "node_ids": [target_id, *other_ids],
                }
            )

    return {
        **scope_report,
        "candidate_count": len(normalized_candidates),
        "candidates": normalized_candidates,
    }


def validate_and_normalize_selection_result(
    result: CandidateSelectionResult,
    candidate: dict[str, Any],
    parent_by_id: dict[str, str | None],
) -> TaxonomyPruneDecision | None:
    if not result.should_prune:
        return None
    candidate_ids = set(candidate["node_ids"])
    if result.keep_node_id is None:
        raise ValidationRetryError(
            f"Selected candidate {candidate['candidate_id']} must provide keep_node_id",
            failed_result=result,
        )
    if result.keep_node_id not in candidate_ids:
        raise ValidationRetryError(
            f"keep_node_id {result.keep_node_id} is not part of candidate {candidate['candidate_id']}",
            failed_result=result,
        )
    if not result.remove_node_ids:
        raise ValidationRetryError(
            f"Selected candidate {candidate['candidate_id']} must remove at least one node",
            failed_result=result,
        )
    for node_id in result.remove_node_ids:
        if node_id not in candidate_ids:
            raise ValidationRetryError(
                f"remove_node_id {node_id} is not part of candidate {candidate['candidate_id']}",
                failed_result=result,
            )
    if result.keep_node_id in result.remove_node_ids:
        raise ValidationRetryError(
            f"Selected candidate {candidate['candidate_id']} keeps and removes the same node {result.keep_node_id}",
            failed_result=result,
        )

    normalized_remove_ids = normalize_remove_ids(result.remove_node_ids, parent_by_id)
    if not normalized_remove_ids:
        return None
    return TaxonomyPruneDecision(
        concern=candidate["concern"],
        keep_node_id=result.keep_node_id,
        remove_node_ids=normalized_remove_ids,
    )


def validate_global_decisions(
    decisions: list[TaxonomyPruneDecision],
    refs: dict[str, PromptNodeRef],
    parent_by_id: dict[str, str | None],
) -> list[TaxonomyPruneDecision]:
    deduped: list[TaxonomyPruneDecision] = []
    seen_keys: set[tuple[str, tuple[str, ...]]] = set()
    for decision in decisions:
        key = (decision.keep_node_id, tuple(sorted(decision.remove_node_ids)))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append(decision)

    all_keeps: set[str] = set()
    for decision in deduped:
        if decision.keep_node_id not in refs:
            raise ValidationRetryError(f"Unknown keep_node_id in merged decisions: {decision.keep_node_id}")
        for node_id in decision.remove_node_ids:
            if node_id not in refs:
                raise ValidationRetryError(f"Unknown remove_node_id in merged decisions: {node_id}")
        all_keeps.add(decision.keep_node_id)

    # Prefer more specific kept descendants over broader ancestor removals.
    normalized: list[TaxonomyPruneDecision] = []
    for decision in deduped:
        filtered_remove_ids = [
            node_id
            for node_id in decision.remove_node_ids
            if not any(node_id in ancestors_of(keep_node_id, parent_by_id) for keep_node_id in all_keeps)
        ]
        if not filtered_remove_ids:
            continue
        normalized.append(
            TaxonomyPruneDecision(
                concern=decision.concern,
                keep_node_id=decision.keep_node_id,
                remove_node_ids=filtered_remove_ids,
            )
        )

    keep_dominant: list[TaxonomyPruneDecision] = []
    for decision in normalized:
        filtered_remove_ids = [node_id for node_id in decision.remove_node_ids if node_id not in all_keeps]
        if not filtered_remove_ids:
            continue
        keep_dominant.append(
            TaxonomyPruneDecision(
                concern=decision.concern,
                keep_node_id=decision.keep_node_id,
                remove_node_ids=filtered_remove_ids,
            )
        )

    all_removed_roots = {
        node_id
        for decision in keep_dominant
        for node_id in decision.remove_node_ids
    }
    globally_normalized: list[TaxonomyPruneDecision] = []
    for decision in keep_dominant:
        filtered_remove_ids = [
            node_id
            for node_id in decision.remove_node_ids
            if not any(ancestor in all_removed_roots for ancestor in ancestors_of(node_id, parent_by_id))
        ]
        if not filtered_remove_ids:
            continue
        globally_normalized.append(
            TaxonomyPruneDecision(
                concern=decision.concern,
                keep_node_id=decision.keep_node_id,
                remove_node_ids=filtered_remove_ids,
            )
        )

    all_removed_roots: set[str] = set()
    for decision in globally_normalized:
        all_removed_roots.update(decision.remove_node_ids)

    if all_keeps & all_removed_roots:
        overlap = sorted(all_keeps & all_removed_roots)
        raise ValidationRetryError(f"Node cannot be kept and removed: {', '.join(overlap)}")

    removed_closure = expand_removed_ids(sorted(all_removed_roots), refs)
    invalid_keeps = sorted(node_id for node_id in all_keeps if node_id in removed_closure)
    if invalid_keeps:
        raise ValidationRetryError(
            "Kept node is inside a pruned subtree: " + ", ".join(invalid_keeps)
        )

    return globally_normalized


async def run_structured_operation(
    *,
    client: AsyncOpenAI,
    prompt: str,
    model: str,
    response_model: type[BaseModel],
    operation_label: str,
    semaphore: asyncio.Semaphore,
    local: bool = False,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    max_attempts: int = DEFAULT_BATCH_MAX_ATTEMPTS,
    validate_result: Any | None = None,
) -> Any:
    generation_kwargs = get_generation_kwargs(model, local=local, default_temperature=0.2)
    use_prompt_only_json = local or "qwen" in (model or "").lower()

    async def run_attempt(retry_context):
        async with semaphore:
            if use_prompt_only_json:
                call_coro = local_structured_call(
                    client,
                    model,
                    f"{system_prompt}\n\n{append_validation_feedback_prompt(prompt, retry_context)}",
                    response_model,
                    temperature=generation_kwargs.get("temperature"),
                    extra_body=generation_kwargs.get("extra_body"),
                    max_attempts=1,
                )
                if request_timeout_seconds is not None:
                    result = await asyncio.wait_for(call_coro, timeout=request_timeout_seconds)
                else:
                    result = await call_coro
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
                    max_attempts=1,
                    request_timeout_seconds=request_timeout_seconds,
                )
        if validate_result is not None:
            return validate_result(result)
        return result

    return await run_with_validation_repair(
        model=model,
        operation_label=operation_label,
        run_attempt=run_attempt,
        max_attempts=max_attempts,
    )


async def run_candidate_discovery_scope(
    scope: ScopeSpec,
    *,
    tree_text: str,
    allowed_node_ids: set[str],
    client: AsyncOpenAI,
    model: str,
    semaphore: asyncio.Semaphore,
    local: bool,
    domain_context: str,
    target: str,
    system_prompt: str,
    request_timeout_seconds: float,
) -> dict[str, Any]:
    async def run_discovery_prompt(prompt_name: str) -> list[dict[str, Any]]:
        prompt = (
            load_prompt(prompt_name)
            .replace("{domain_context}", domain_context)
            .replace("{target}", target)
            .replace("{scope_type}", scope.scope_type)
            .replace("{scope_metas}", ", ".join(scope.metas))
            .replace("{tree}", tree_text)
        )
        return await run_structured_operation(
            client=client,
            prompt=prompt,
            model=model,
            response_model=CandidateDiscoveryResult,
            operation_label=f"candidate discovery [{scope.scope_id}:{prompt_name}]",
            semaphore=semaphore,
            local=local,
            system_prompt=system_prompt,
            request_timeout_seconds=request_timeout_seconds,
            validate_result=lambda result: validate_candidate_discovery_result(result, allowed_node_ids),
        )

    candidate_lists = await asyncio.gather(*[run_discovery_prompt(name) for name in DISCOVERY_PROMPT_NAMES])
    candidates = dedupe_candidate_payloads([candidate for group in candidate_lists for candidate in group])
    return {
        "scope_id": scope.scope_id,
        "scope_type": scope.scope_type,
        "scope_metas": list(scope.metas),
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


def format_candidate_for_prompt(
    candidate: dict[str, Any],
    refs: dict[str, PromptNodeRef],
) -> str:
    lines: list[str] = []
    lines.append(f"### {candidate['candidate_id']} :: {candidate['concern']}")
    for node_id in candidate["node_ids"]:
        ref = refs[node_id]
        desc_suffix = f" :: {ref.description}" if ref.description else ""
        lines.append(f"- [{node_id}] {ref.path}{desc_suffix}")
    return "\n".join(lines).strip()


async def run_candidate_selection_scope(
    scope_report: dict[str, Any],
    *,
    tree_text: str,
    refs: dict[str, PromptNodeRef],
    parent_by_id: dict[str, str | None],
    client: AsyncOpenAI,
    model: str,
    semaphore: asyncio.Semaphore,
    local: bool,
    domain_context: str,
    target: str,
    system_prompt: str,
    request_timeout_seconds: float,
) -> dict[str, Any]:
    candidates = scope_report["candidates"]
    if not candidates:
        return {
            **scope_report,
            "decision_count": 0,
            "decisions": [],
        }

    async def review_candidate(candidate: dict[str, Any]) -> tuple[dict[str, Any], TaxonomyPruneDecision | None]:
        prompt = (
            load_prompt("phase_09_candidate_selection.md")
            .replace("{domain_context}", domain_context)
            .replace("{target}", target)
            .replace("{scope_type}", scope_report["scope_type"])
            .replace("{scope_metas}", ", ".join(scope_report["scope_metas"]))
            .replace("{tree}", tree_text)
            .replace("{candidate}", format_candidate_for_prompt(candidate, refs))
        )
        decision = await run_structured_operation(
            client=client,
            prompt=prompt,
            model=model,
            response_model=CandidateSelectionResult,
            operation_label=f"candidate selection [{scope_report['scope_id']}:{candidate['candidate_id']}]",
            semaphore=semaphore,
            local=local,
            system_prompt=system_prompt,
            request_timeout_seconds=request_timeout_seconds,
            validate_result=lambda result: validate_and_normalize_selection_result(result, candidate, parent_by_id),
        )
        return candidate, decision

    reviewed = await asyncio.gather(*[review_candidate(candidate) for candidate in candidates])
    decisions = [decision for _, decision in reviewed if decision is not None]
    decision_payloads = expand_decision_payloads(refs, decisions)
    for payload, (candidate, decision) in zip(
        decision_payloads,
        [(candidate, decision) for candidate, decision in reviewed if decision is not None],
        strict=False,
    ):
        payload["candidate_id"] = candidate["candidate_id"]

    return {
        **scope_report,
        "decision_count": len(decisions),
        "decision_models": decisions,
        "decisions": decision_payloads,
    }


def expand_decision_payloads(
    refs: dict[str, PromptNodeRef],
    decisions: list[TaxonomyPruneDecision],
) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for decision in decisions:
        keep_ref = refs[decision.keep_node_id]
        remove_payloads = []
        for node_id in decision.remove_node_ids:
            ref = refs[node_id]
            remove_payloads.append(
                {
                    "node_id": ref.node_id,
                    "meta": ref.meta,
                    "code": ref.code,
                    "path": ref.path,
                    "description": ref.description,
                    "subtree_removed_count": 1 + len(ref.descendant_node_ids),
                }
            )
        payloads.append(
            {
                "concern": decision.concern,
                "keep": {
                    "node_id": keep_ref.node_id,
                    "meta": keep_ref.meta,
                    "code": keep_ref.code,
                    "path": keep_ref.path,
                    "description": keep_ref.description,
                },
                "remove": remove_payloads,
            }
        )
    return payloads


def apply_pruning_to_tree(
    tree: dict[str, Any],
    refs: dict[str, PromptNodeRef],
    decisions: list[TaxonomyPruneDecision],
    model: str,
) -> tuple[dict[str, Any], set[str]]:
    remove_root_ids: list[str] = []
    for decision in decisions:
        remove_root_ids.extend(decision.remove_node_ids)

    removed_ids = expand_removed_ids(remove_root_ids, refs)
    removed_paths_by_meta: dict[str, set[str]] = {}
    for node_id in removed_ids:
        ref = refs[node_id]
        removed_paths_by_meta.setdefault(ref.meta, set()).add(ref.code)

    pruned_metas: list[dict[str, Any]] = []
    for raw_meta_entry in tree.get("metas", []):
        meta_entry = canonical_path_meta(raw_meta_entry)
        meta = meta_entry["meta"]
        removed_paths = removed_paths_by_meta.get(meta, set())
        paths_out = []
        for row in meta_entry.get("paths", []):
            path = _coerce_path(row.get("path"))
            if not path or path in removed_paths:
                continue
            paths_out.append(
                {
                    "path": path,
                    "description": str(row.get("description") or ""),
                }
            )
        out_entry: dict[str, Any] = {"meta": meta, "paths": paths_out}
        if "reasoning" in meta_entry:
            out_entry["reasoning"] = meta_entry["reasoning"]
        pruned_metas.append(out_entry)

    pruned_tree = {
        "domain_context": tree.get("domain_context"),
        "target": tree.get("target"),
        "model": model,
        "source_tree_model": tree.get("model"),
        "phase": "09_ambiguity_pruning",
        "metas": pruned_metas,
    }
    if "max_children" in tree:
        pruned_tree["max_children"] = tree["max_children"]
    validate_path_taxonomy_tree(pruned_tree)
    return pruned_tree, removed_ids


def build_scope_specs(tree: dict[str, Any], *, sample_mode: bool, target: str) -> list[ScopeSpec]:
    meta_names = [meta["meta"] for meta in tree.get("metas", [])]
    specs: list[ScopeSpec] = [ScopeSpec(scope_id="in_meta", scope_type="in_meta", metas=(target,))]

    if sample_mode:
        comparison_metas = [meta for meta in meta_names if meta in {"clinical_findings"} and meta != target]
    else:
        comparison_metas = [meta for meta in meta_names if meta != target]

    for meta in comparison_metas:
        specs.append(
            ScopeSpec(
                scope_id=f"cross_meta__{meta}",
                scope_type="cross_meta",
                metas=(target, meta),
            )
        )
    return specs


def build_final_report(
    source_tree: dict[str, Any],
    final_tree: dict[str, Any],
    scope_reports: list[dict[str, Any]],
    removed_ids: set[str],
    model: str,
    input_model: str,
) -> dict[str, Any]:
    all_decisions: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, tuple[str, ...]]] = set()
    for report in scope_reports:
        for decision in report.get("decisions", []):
            key = (
                decision["keep"]["node_id"],
                tuple(sorted(item["node_id"] for item in decision["remove"])),
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)
            all_decisions.append(decision)
    removed_root_count = sum(len(decision["remove"]) for decision in all_decisions)
    return {
        "phase": "09_ambiguity_pruning",
        "model": model,
        "input_model": input_model,
        "source_tree_model": source_tree.get("model"),
        "target": source_tree.get("target"),
        "domain_context": source_tree.get("domain_context"),
        "decision_count": len(all_decisions),
        "removed_root_count": removed_root_count,
        "removed_node_count": len(removed_ids),
        "remaining_meta_count": len(final_tree.get("metas", [])),
        "scope_reports": scope_reports,
        "decisions": all_decisions,
    }


def serialize_scope_reports(scope_reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for report in scope_reports:
        item = dict(report)
        if "decision_models" in item:
            item["decision_models"] = [
                decision.model_dump() if hasattr(decision, "model_dump") else decision
                for decision in item["decision_models"]
            ]
        serialized.append(item)
    return serialized


def debug_artifact_path(name: str, model: str, *, sample_mode: bool) -> Path:
    base = INTERMEDIATE_DIR / name
    if sample_mode:
        base = base.with_name(f"{base.stem}_sample{base.suffix}")
    return provider_scoped_path(base, model)


def save_debug_artifact(name: str, payload: dict[str, Any] | list[dict[str, Any]], model: str, *, sample_mode: bool) -> Path:
    path = debug_artifact_path(name, model, sample_mode=sample_mode)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


def print_decision_summary(report: dict[str, Any]) -> None:
    decisions = report.get("decisions", [])
    if not decisions:
        print("No ambiguity pruning decisions returned.")
        return
    print(f"Flagged {len(decisions)} ambiguity-pruning decisions:")
    for decision in decisions:
        keep = decision["keep"]
        remove_codes = ", ".join(item["code"] for item in decision["remove"])
        print(f"  - {decision['concern']}: keep {keep['code']} | remove {remove_codes}")


async def main(
    sample_mode: bool = False,
    domain_context: str = DEFAULT_DOMAIN_CONTEXT,
    target: str = DEFAULT_TARGET,
    model: str = DEFAULT_MODEL,
    input_model: str = DEFAULT_INPUT_MODEL,
    concurrency: int | None = None,
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    base_url: str | None = None,
    run_id: str | None = None,
):
    configure_paths(run_id)
    source_tree = load_final_tree(input_model)
    validate_path_taxonomy_tree(source_tree)
    working_source_tree = normalize_tree_for_phase09(source_tree)
    metas = working_source_tree.get("metas", [])
    if sample_mode:
        preferred_metas = {target, "clinical_findings"}
        metas = [meta for meta in metas if meta.get("meta") in preferred_metas] or metas[:2]
        print(f"=== SAMPLE MODE: {len(metas)} metas ===")
    print(f"Artifact scope: {DATA_DIR}")

    working_tree = dict(source_tree)
    working_tree["metas"] = metas

    refs, ordered_node_ids_by_meta, ids_by_meta = build_prompt_index(working_tree.get("metas", []))
    parent_by_id = build_parent_by_id(refs)
    scopes = build_scope_specs(working_tree, sample_mode=sample_mode, target=target)
    scope_by_id = {scope.scope_id: scope for scope in scopes}

    client, local = get_client(base_url=base_url)
    concurrency = resolve_concurrency(
        concurrency,
        local,
        model=model,
        remote_default=DEFAULT_CONCURRENCY,
        local_default=1,
    )
    semaphore = asyncio.Semaphore(concurrency)

    discovery_tasks = [
        run_candidate_discovery_scope(
            scope,
            tree_text=render_scope_tree(scope, refs, ordered_node_ids_by_meta),
            allowed_node_ids=allowed_node_ids_for_scope(scope, refs, ids_by_meta),
            client=client,
            model=model,
            semaphore=semaphore,
            local=local,
            domain_context=domain_context,
            target=target,
            system_prompt=system_prompt,
            request_timeout_seconds=request_timeout_seconds,
        )
        for scope in scopes
    ]
    discovery_reports = [
        normalize_scope_candidates(report, refs, target)
        for report in await asyncio.gather(*discovery_tasks)
    ]
    for report in discovery_reports:
        print(f"[{report['scope_id']}] discovered {report['candidate_count']} candidates")
    discovery_debug_path = save_debug_artifact(
        "phase_09_discovery_reports.json",
        serialize_scope_reports(discovery_reports),
        model,
        sample_mode=sample_mode,
    )
    print(f"Saved discovery debug: {discovery_debug_path}")

    selection_tasks = [
        run_candidate_selection_scope(
            report,
            tree_text=render_scope_tree(scope_by_id[report["scope_id"]], refs, ordered_node_ids_by_meta),
            refs=refs,
            parent_by_id=parent_by_id,
            client=client,
            model=model,
            semaphore=semaphore,
            local=local,
            domain_context=domain_context,
            target=target,
            system_prompt=system_prompt,
            request_timeout_seconds=request_timeout_seconds,
        )
        for report in discovery_reports
    ]
    scope_reports = await asyncio.gather(*selection_tasks)
    selection_debug_path = save_debug_artifact(
        "phase_09_selection_scope_reports.json",
        serialize_scope_reports(scope_reports),
        model,
        sample_mode=sample_mode,
    )
    print(f"Saved selection debug: {selection_debug_path}")
    all_decision_models = [
        decision
        for report in scope_reports
        for decision in report.get("decision_models", [])
    ]
    merged_decisions = validate_global_decisions(all_decision_models, refs, parent_by_id)
    decision_payloads = expand_decision_payloads(refs, merged_decisions)
    payload_by_key = {
        (payload["keep"]["node_id"], tuple(sorted(item["node_id"] for item in payload["remove"]))): payload
        for payload in decision_payloads
    }

    cleaned_scope_reports: list[dict[str, Any]] = []
    for report in scope_reports:
        report_payloads = []
        for decision in report.get("decision_models", []):
            key = (decision.keep_node_id, tuple(sorted(decision.remove_node_ids)))
            payload = payload_by_key.get(key)
            if payload is None:
                continue
            report_payloads.append(payload)
        cleaned_scope_reports.append(
            {
                "scope_id": report["scope_id"],
                "scope_type": report["scope_type"],
                "scope_metas": report["scope_metas"],
                "candidate_count": report["candidate_count"],
                "candidates": report["candidates"],
                "decision_count": len(report_payloads),
                "decisions": report_payloads,
            }
        )
        if report_payloads:
            print(f"[{report['scope_id']}] selected {len(report_payloads)} decisions")

    final_tree, removed_ids = apply_pruning_to_tree(working_tree, refs, merged_decisions, model)
    report = build_final_report(source_tree, final_tree, cleaned_scope_reports, removed_ids, model, input_model)
    print_decision_summary(report)

    if sample_mode:
        print("Sample mode - not saving results")
        return report

    report_path = provider_scoped_path(INTERMEDIATE_DIR / "taxonomy_tree_final_pruning_report.json", model)
    pruned_tree_path = provider_scoped_path(INTERMEDIATE_DIR / "taxonomy_tree_final_pruned.json", model)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    with open(pruned_tree_path, "w") as f:
        json.dump(final_tree, f, indent=2, ensure_ascii=False)
    print(f"Saved pruning report: {report_path}")
    print(f"Saved pruned tree: {pruned_tree_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 09: Ambiguity Pruning")
    parser.add_argument("--sample", action="store_true", help="Sample mode (no save)")
    parser.add_argument("--domain-context", type=str, default=DEFAULT_DOMAIN_CONTEXT, help="Domain context")
    parser.add_argument("--target", type=str, default=DEFAULT_TARGET, help="Target concept")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model to use")
    parser.add_argument("--input-model", type=str, default=DEFAULT_INPUT_MODEL, help="Model used to generate the Phase 08 final tree")
    parser.add_argument("--concurrency", type=int, default=None, help="Parallel calls (defaults to 8; local=1)")
    parser.add_argument("--request-timeout-seconds", type=float, default=DEFAULT_REQUEST_TIMEOUT_SECONDS, help="Per-attempt timeout for a single request")
    parser.add_argument("--system-prompt", type=str, default=DEFAULT_SYSTEM_PROMPT, help="System prompt for Phase 09 calls")
    parser.add_argument("--base-url", type=str, default=None, help="Custom API base URL (e.g. LM Studio)")
    parser.add_argument("--run-id", type=str, default=None, help="Optional run id for isolated artifacts under data/runs/<run_id>/")

    args = parser.parse_args()
    asyncio.run(main(
        sample_mode=args.sample,
        domain_context=args.domain_context,
        target=args.target,
        model=args.model,
        input_model=args.input_model,
        concurrency=args.concurrency,
        request_timeout_seconds=args.request_timeout_seconds,
        system_prompt=args.system_prompt,
        base_url=args.base_url,
        run_id=args.run_id,
    ))
