#!/usr/bin/env python3
"""Prepare a text-only parquet input file for the taxonomy pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_TEXT_COLUMNS = ("text", "free_text", "description", "complication", "note")


def resolve_text_column(df: pd.DataFrame, requested: str | None) -> str:
    if requested:
        if requested not in df.columns:
            raise ValueError(f"Text column not found: {requested}")
        return requested
    for column in DEFAULT_TEXT_COLUMNS:
        if column in df.columns:
            return column
    raise ValueError(
        "No default text column found. Use --text-column. "
        f"Accepted defaults: {', '.join(DEFAULT_TEXT_COLUMNS)}"
    )


def prepare(input_path: Path, output_path: Path, text_column: str | None) -> None:
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(input_path)
    elif suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(input_path)
    else:
        raise ValueError("Input must be CSV or parquet")

    column = resolve_text_column(df, text_column)
    texts = df[column].dropna().astype(str).str.strip()
    texts = texts[texts != ""]
    if texts.empty:
        raise ValueError("No non-empty text rows found")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"text": texts.tolist()}).to_parquet(output_path, index=False)
    print(f"Wrote {len(texts)} text rows to {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a text-only parquet file for the taxonomy pipeline.")
    parser.add_argument("--input", required=True, type=Path, help="CSV or parquet file")
    parser.add_argument("--output", default=Path("data/inputs/texts.parquet"), type=Path, help="Output parquet path")
    parser.add_argument("--text-column", default=None, help="Name of the input free-text column")
    args = parser.parse_args()
    try:
        prepare(args.input, args.output, args.text_column)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
