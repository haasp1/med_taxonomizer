"""Helpers for resolving per-experiment artifact directories."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class ExperimentArtifactPaths:
    """Resolved artifact directories for an experiment, optionally scoped to a run."""

    data_dir: Path
    intermediate_dir: Path
    cache_dir: Path
    final_dir: Path
    run_id: str | None = None


def normalize_run_id(run_id: str | None) -> str | None:
    """Validate and normalize an optional run identifier."""
    if run_id is None:
        return None

    normalized = run_id.strip()
    if not normalized:
        return None

    if not RUN_ID_PATTERN.fullmatch(normalized):
        raise ValueError(
            "Invalid run_id. Use only letters, numbers, '.', '_' or '-', "
            "and start with a letter or number."
        )
    return normalized


def resolve_experiment_artifact_paths(
    experiment_dir: Path,
    run_id: str | None = None,
) -> ExperimentArtifactPaths:
    """Resolve standard artifact directories for an experiment."""
    base_data_dir = experiment_dir / "data"
    normalized_run_id = normalize_run_id(run_id)

    data_dir = base_data_dir
    if normalized_run_id is not None:
        data_dir = base_data_dir / "runs" / normalized_run_id

    return ExperimentArtifactPaths(
        data_dir=data_dir,
        intermediate_dir=data_dir / "intermediate",
        cache_dir=data_dir / "cache",
        final_dir=data_dir / "final",
        run_id=normalized_run_id,
    )
