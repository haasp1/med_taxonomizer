#!/usr/bin/env bash
# Shared settings for the AACT transfer wrappers. Sourced, not executed.

AACT_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
AACT_DIR="paper_runs/aact_transfer"
AACT_TEXT_EXPORT="${AACT_DIR}/phase04_label_exports_2026-05-28/aact_per_sample_source_text_and_predicted_meta_sets.csv"
AACT_INPUT_PARQUET="${AACT_INPUT_PARQUET:-data/inputs/aact_coronary_intervention_1979.parquet}"
AACT_LOG_DIR="${AACT_LOG_DIR:-logs/aact_transfer}"
AACT_EXPECTED_ROWS=1979
AACT_TAXONOMY_RUN_ID="${AACT_TAXONOMY_RUN_ID:-aact_qwen35_27b_20260510_taxonomizer_full1979}"

cd "$AACT_REPO_ROOT"

# Optional: load OPENROUTER_API_KEY from a .env file in the repository root.
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

aact_uses_local_endpoint() {
  local arg
  for arg in "$@"; do
    if [[ "$arg" == "--base-url" ]]; then
      return 0
    fi
  done
  return 1
}

aact_require_api_key() {
  if aact_uses_local_endpoint "$@"; then
    return 0
  fi
  if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
    echo "OPENROUTER_API_KEY is not set (export it or put it in .env)." >&2
    exit 1
  fi
}

# Rebuild the text-only pipeline input (one `text` column, 1,979 rows, in the
# original pipeline order) from the released per-sample export.
aact_prepare_input() {
  if [[ -f "$AACT_INPUT_PARQUET" ]]; then
    echo "input=${AACT_INPUT_PARQUET} (existing)"
  else
    uv run python scripts/prepare_texts.py \
      --input "$AACT_TEXT_EXPORT" \
      --output "$AACT_INPUT_PARQUET" \
      --text-column text
  fi
  uv run python - "$AACT_INPUT_PARQUET" "$AACT_EXPECTED_ROWS" "$AACT_TEXT_EXPORT" <<'PY'
import sys
import pandas as pd

path, expected = sys.argv[1], int(sys.argv[2])
df = pd.read_parquet(path)
if list(df.columns) != ["text"]:
    raise SystemExit(f"Expected only a text column in {path}; got {list(df.columns)}")
if len(df) != expected:
    raise SystemExit(f"Expected {expected} rows in {path}; got {len(df)}")
source = pd.read_csv(sys.argv[3])["text"]
if not df["text"].reset_index(drop=True).equals(source):
    raise SystemExit(f"Input text/order in {path} does not match the frozen AACT export")
print(f"input_rows={len(df)}")
PY
}

# Copy the fixed Qwen Phase 00-03 taxonomy artifacts into a Phase 04 run dir.
aact_copy_taxonomy() {
  local run_id="$1"
  local source_dir="scripts/data/runs/${AACT_TAXONOMY_RUN_ID}/intermediate"
  local target_dir="scripts/data/runs/${run_id}/intermediate"
  local required=(
    meta_categories_consolidated_qwen.json
    categories_consolidated_qwen.json
  )
  local rel
  for rel in "${required[@]}"; do
    if [[ ! -f "${source_dir}/${rel}" ]]; then
      echo "Missing ${source_dir}/${rel}. Run run_aact_taxonomy_full1979_20260510.sh first." >&2
      exit 1
    fi
  done
  if [[ -d "$target_dir" ]]; then
    echo "preserved_existing_intermediate=${target_dir}"
  else
    mkdir -p "$(dirname "$target_dir")"
    cp -R "$source_dir" "$target_dir"
    echo "copied_taxonomy=${source_dir} -> ${target_dir}"
  fi
}
