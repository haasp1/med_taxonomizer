"""Utilities for provider-scoped artifact names and model-specific defaults."""

from __future__ import annotations

import re
from pathlib import Path

QWEN_DEFAULT_TEMPERATURE = 0.6


def is_qwen_model(model: str) -> bool:
    """Return True when the model identifier refers to a Qwen family model."""
    normalized = (model or "").strip().lower()
    return "qwen" in normalized


def resolve_temperature(model: str, temperature: float | None) -> float | None:
    """Enforce project-wide sampling defaults for specific model families."""
    if is_qwen_model(model):
        return QWEN_DEFAULT_TEMPERATURE
    return temperature


def resolve_remote_concurrency_default(model: str | None, fallback: int = 10) -> int:
    """Return model-aware remote concurrency defaults."""
    normalized = (model or "").strip().lower()
    if not normalized:
        return fallback
    if "qwen" not in normalized:
        return fallback
    if re.search(r"(^|[^0-9])27b([^0-9]|$)", normalized):
        return 8
    if re.search(r"(^|[^0-9])9b([^0-9]|$)", normalized):
        return 20
    return fallback


def get_provider_name(model: str) -> str:
    """Return a stable provider tag for a model identifier."""
    normalized = (model or "").strip().lower()
    if not normalized:
        return "unknown"

    if normalized.startswith("openai/") or "gpt-" in normalized:
        return "openai"
    if is_qwen_model(model):
        return "qwen"

    candidate = normalized.split("/", 1)[0]
    candidate = re.sub(r"[^a-z0-9]+", "_", candidate).strip("_")
    return candidate or "unknown"


def provider_scoped_path(path: Path, model: str) -> Path:
    """Append the provider tag before the file suffix."""
    provider = get_provider_name(model)
    return path.with_name(f"{path.stem}_{provider}{path.suffix}")


def resolve_provider_path(
    preferred: Path,
    model: str,
    fallback: Path | None = None,
) -> Path:
    """
    Resolve a provider-scoped artifact, falling back to legacy filenames.

    Resolution order:
    1. provider-scoped preferred path
    2. legacy preferred path
    3. provider-scoped fallback path
    4. legacy fallback path
    """
    candidates = [provider_scoped_path(preferred, model), preferred]
    if fallback is not None:
        candidates.extend([provider_scoped_path(fallback, model), fallback])

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]
