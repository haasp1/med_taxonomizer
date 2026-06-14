#!/usr/bin/env bash

set -euo pipefail

EXPERIMENT_DIR="scripts"

MODEL="openai/gpt-4o"
INPUT_DATA="data/inputs/texts.parquet"
TAXONOMY_MODEL=""
LABELING_MODEL=""
TREE_MODEL=""
TARGET="adverse_events"
DOMAIN_CONTEXT=""
META_GUIDANCE=""
CONCURRENCY=""
BASE_URL=""
RUN_ID=""
MAX_PHASE_ATTEMPTS=3
START_PHASE="00"
END_PHASE="09"
RUNNER_RESUME=0
COMMON_ARGS=()
PHASE_CMD=()

usage() {
  cat <<'EOF'
Run the full taxonomy pipeline sequentially with phase-level retries.

Usage:
  ./scripts/run_taxonomy_pipeline.sh [options]

Options:
  --model MODEL                Model for phases 00-03, 05-07 (default: openai/gpt-4o)
  --input PATH                 Parquet file with a text column for Phase 00
  --taxonomy-model MODEL       Taxonomy model consumed by phases 04-07 (default: --model)
  --labeling-model MODEL       Labeling model for phase 04/05/06/07 (default: Qwen 27B -> Qwen 9B, otherwise --model)
  --tree-model MODEL           Phase 08/09 tree model (default: --model)
  --target TARGET              Target concept (default: adverse_events)
  --domain-context TEXT        Optional domain context passed to phases that support it
  --meta-guidance TEXT         Optional meta guidance passed to Phase 00
  --resume                     Reuse resumable per-phase artifacts on the first attempt
  --concurrency N              Override phase concurrency
  --base-url URL               Custom API base URL for local serving
  --run-id ID                  Isolate artifacts under data/runs/ID across all phases
  --max-phase-attempts N       Total attempts per phase before stopping (default: 3)
  --start-phase ID             Start at phase ID, e.g. 04 (default: 00)
  --end-phase ID               Stop after phase ID, e.g. 07 (default: 09)
  -h, --help                   Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL="$2"
      shift 2
      ;;
    --input)
      INPUT_DATA="$2"
      shift 2
      ;;
    --taxonomy-model)
      TAXONOMY_MODEL="$2"
      shift 2
      ;;
    --labeling-model)
      LABELING_MODEL="$2"
      shift 2
      ;;
    --tree-model)
      TREE_MODEL="$2"
      shift 2
      ;;
    --target)
      TARGET="$2"
      shift 2
      ;;
    --domain-context)
      DOMAIN_CONTEXT="$2"
      shift 2
      ;;
    --meta-guidance)
      META_GUIDANCE="$2"
      shift 2
      ;;
    --resume)
      RUNNER_RESUME=1
      shift
      ;;
    --concurrency)
      CONCURRENCY="$2"
      shift 2
      ;;
    --base-url)
      BASE_URL="$2"
      shift 2
      ;;
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --max-phase-attempts)
      MAX_PHASE_ATTEMPTS="$2"
      shift 2
      ;;
    --start-phase)
      START_PHASE="$2"
      shift 2
      ;;
    --end-phase)
      END_PHASE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

default_labeling_model() {
  local model="$1"
  if [[ "$model" == *qwen*27b* ]]; then
    echo "${model/27b/9b}"
    return 0
  fi
  echo "$model"
}

if [[ -z "$TAXONOMY_MODEL" ]]; then
  TAXONOMY_MODEL="$MODEL"
fi
if [[ -z "$LABELING_MODEL" ]]; then
  LABELING_MODEL="$(default_labeling_model "$MODEL")"
fi
if [[ -z "$TREE_MODEL" ]]; then
  TREE_MODEL="$MODEL"
fi

PHASE_ORDER=("00" "01" "02" "03" "04" "05" "06" "07" "08" "09")

phase_index() {
  local phase="$1"
  local i
  for i in "${!PHASE_ORDER[@]}"; do
    if [[ "${PHASE_ORDER[$i]}" == "$phase" ]]; then
      echo "$i"
      return 0
    fi
  done
  return 1
}

START_INDEX="$(phase_index "$START_PHASE")" || { echo "Unknown start phase: $START_PHASE" >&2; exit 1; }
END_INDEX="$(phase_index "$END_PHASE")" || { echo "Unknown end phase: $END_PHASE" >&2; exit 1; }

if (( START_INDEX > END_INDEX )); then
  echo "start-phase must be <= end-phase" >&2
  exit 1
fi

build_common_args() {
  COMMON_ARGS=()
  if [[ -n "$CONCURRENCY" ]]; then
    COMMON_ARGS+=(--concurrency "$CONCURRENCY")
  fi
  if [[ -n "$BASE_URL" ]]; then
    COMMON_ARGS+=(--base-url "$BASE_URL")
  fi
}

phase_name() {
  case "$1" in
    "00") echo "Meta Discovery" ;;
    "01") echo "Meta Consolidation" ;;
    "02") echo "Category Discovery" ;;
    "03") echo "Category Consolidation" ;;
    "04") echo "Labeling" ;;
    "05") echo "Review Other" ;;
    "06") echo "Split Audit" ;;
    "07") echo "Subcategory Discovery" ;;
    "08") echo "Tree Consolidation" ;;
    "09") echo "Ambiguity Pruning" ;;
  esac
}

build_phase_command() {
  local phase="$1"
  local attempt="$2"
  build_common_args
  local use_resume=0
  if (( RUNNER_RESUME == 1 || attempt > 1 )); then
    use_resume=1
  fi

  case "$phase" in
    "00")
      PHASE_CMD=(
        uv run python "${EXPERIMENT_DIR}/phase_00_meta_discovery.py"
        --input "$INPUT_DATA"
        --model "$MODEL"
        --target "$TARGET"
      )
      if [[ -n "$DOMAIN_CONTEXT" ]]; then
        PHASE_CMD+=(--domain-context "$DOMAIN_CONTEXT")
      fi
      if [[ -n "$META_GUIDANCE" ]]; then
        PHASE_CMD+=(--meta-guidance "$META_GUIDANCE")
      fi
      if (( ${#COMMON_ARGS[@]} > 0 )); then
        PHASE_CMD+=("${COMMON_ARGS[@]}")
      fi
      if [[ -n "$RUN_ID" ]]; then
        PHASE_CMD+=(--run-id "$RUN_ID")
      fi
      if (( use_resume == 1 )); then
        PHASE_CMD+=(--resume)
      fi
      ;;
    "01")
      PHASE_CMD=(
        uv run python "${EXPERIMENT_DIR}/phase_01_meta_consolidation.py"
        --model "$MODEL"
        --target "$TARGET"
      )
      if [[ -n "$DOMAIN_CONTEXT" ]]; then
        PHASE_CMD+=(--domain-context "$DOMAIN_CONTEXT")
      fi
      if [[ -n "$BASE_URL" ]]; then
        PHASE_CMD+=(--base-url "$BASE_URL")
      fi
      if [[ -n "$RUN_ID" ]]; then
        PHASE_CMD+=(--run-id "$RUN_ID")
      fi
      ;;
    "02")
      PHASE_CMD=(
        uv run python "${EXPERIMENT_DIR}/phase_02_category_discovery.py"
        --model "$MODEL"
      )
      if [[ -n "$DOMAIN_CONTEXT" ]]; then
        PHASE_CMD+=(--domain-context "$DOMAIN_CONTEXT")
      fi
      if (( ${#COMMON_ARGS[@]} > 0 )); then
        PHASE_CMD+=("${COMMON_ARGS[@]}")
      fi
      if [[ -n "$RUN_ID" ]]; then
        PHASE_CMD+=(--run-id "$RUN_ID")
      fi
      if (( use_resume == 1 )); then
        PHASE_CMD+=(--resume)
      fi
      ;;
    "03")
      PHASE_CMD=(
        uv run python "${EXPERIMENT_DIR}/phase_03_category_consolidation.py"
        --model "$MODEL"
        --target "$TARGET"
      )
      if [[ -n "$DOMAIN_CONTEXT" ]]; then
        PHASE_CMD+=(--domain-context "$DOMAIN_CONTEXT")
      fi
      if (( ${#COMMON_ARGS[@]} > 0 )); then
        PHASE_CMD+=("${COMMON_ARGS[@]}")
      fi
      if [[ -n "$RUN_ID" ]]; then
        PHASE_CMD+=(--run-id "$RUN_ID")
      fi
      ;;
    "04")
      PHASE_CMD=(
        uv run python "${EXPERIMENT_DIR}/phase_04_labeling.py"
        --model "$LABELING_MODEL"
        --taxonomy-model "$TAXONOMY_MODEL"
      )
      if [[ -n "$DOMAIN_CONTEXT" ]]; then
        PHASE_CMD+=(--domain-context "$DOMAIN_CONTEXT")
      fi
      if (( ${#COMMON_ARGS[@]} > 0 )); then
        PHASE_CMD+=("${COMMON_ARGS[@]}")
      fi
      if [[ -n "$RUN_ID" ]]; then
        PHASE_CMD+=(--run-id "$RUN_ID")
      fi
      if (( use_resume == 1 )); then
        PHASE_CMD+=(--rerun-missing-only)
      fi
      ;;
    "05")
      PHASE_CMD=(
        uv run python "${EXPERIMENT_DIR}/phase_05_review_other.py"
        --model "$MODEL"
        --taxonomy-model "$TAXONOMY_MODEL"
        --labeling-model "$LABELING_MODEL"
        --target "$TARGET"
      )
      if [[ -n "$DOMAIN_CONTEXT" ]]; then
        PHASE_CMD+=(--domain-context "$DOMAIN_CONTEXT")
      fi
      if (( ${#COMMON_ARGS[@]} > 0 )); then
        PHASE_CMD+=("${COMMON_ARGS[@]}")
      fi
      if [[ -n "$RUN_ID" ]]; then
        PHASE_CMD+=(--run-id "$RUN_ID")
      fi
      ;;
    "06")
      PHASE_CMD=(
        uv run python "${EXPERIMENT_DIR}/phase_06_split_audit.py"
        --model "$MODEL"
        --taxonomy-model "$TAXONOMY_MODEL"
        --labeling-model "$LABELING_MODEL"
        --target "$TARGET"
      )
      if [[ -n "$DOMAIN_CONTEXT" ]]; then
        PHASE_CMD+=(--domain-context "$DOMAIN_CONTEXT")
      fi
      if (( ${#COMMON_ARGS[@]} > 0 )); then
        PHASE_CMD+=("${COMMON_ARGS[@]}")
      fi
      if [[ -n "$RUN_ID" ]]; then
        PHASE_CMD+=(--run-id "$RUN_ID")
      fi
      if (( use_resume == 1 )); then
        PHASE_CMD+=(--resume)
      fi
      ;;
    "07")
      PHASE_CMD=(
        uv run python "${EXPERIMENT_DIR}/phase_07_subcategory_discovery.py"
        --model "$MODEL"
        --taxonomy-model "$TAXONOMY_MODEL"
        --labeling-model "$LABELING_MODEL"
        --target "$TARGET"
      )
      if [[ -n "$DOMAIN_CONTEXT" ]]; then
        PHASE_CMD+=(--domain-context "$DOMAIN_CONTEXT")
      fi
      if (( ${#COMMON_ARGS[@]} > 0 )); then
        PHASE_CMD+=("${COMMON_ARGS[@]}")
      fi
      if [[ -n "$RUN_ID" ]]; then
        PHASE_CMD+=(--run-id "$RUN_ID")
      fi
      if (( use_resume == 1 )); then
        PHASE_CMD+=(--resume)
      fi
      ;;
    "08")
      PHASE_CMD=(
        uv run python "${EXPERIMENT_DIR}/phase_08_tree_consolidation.py"
        --model "$TREE_MODEL"
        --input-model "$MODEL"
        --target "$TARGET"
      )
      if [[ -n "$DOMAIN_CONTEXT" ]]; then
        PHASE_CMD+=(--domain-context "$DOMAIN_CONTEXT")
      fi
      if (( ${#COMMON_ARGS[@]} > 0 )); then
        PHASE_CMD+=("${COMMON_ARGS[@]}")
      fi
      if [[ -n "$RUN_ID" ]]; then
        PHASE_CMD+=(--run-id "$RUN_ID")
      fi
      ;;
    "09")
      PHASE_CMD=(
        uv run python "${EXPERIMENT_DIR}/phase_09_ambiguity_pruning.py"
        --model "$TREE_MODEL"
        --input-model "$TREE_MODEL"
        --target "$TARGET"
      )
      if [[ -n "$DOMAIN_CONTEXT" ]]; then
        PHASE_CMD+=(--domain-context "$DOMAIN_CONTEXT")
      fi
      if (( ${#COMMON_ARGS[@]} > 0 )); then
        PHASE_CMD+=("${COMMON_ARGS[@]}")
      fi
      if [[ -n "$RUN_ID" ]]; then
        PHASE_CMD+=(--run-id "$RUN_ID")
      fi
      ;;
  esac
}

run_phase() {
  local phase_id="$1"
  local attempt=1

  while (( attempt <= MAX_PHASE_ATTEMPTS )); do
    local phase_title
    phase_title="$(phase_name "$phase_id")"
    PHASE_CMD=()
    build_phase_command "$phase_id" "$attempt"

    echo
    echo "============================================================"
    echo " Phase ${phase_id}: ${phase_title} (attempt ${attempt}/${MAX_PHASE_ATTEMPTS})"
    echo "============================================================"

    if "${PHASE_CMD[@]}"; then
      echo
      echo "Phase ${phase_id} complete."
      return 0
    else
      local exit_code=$?
      if (( exit_code == 130 || exit_code == 143 )); then
        echo
        echo "Phase ${phase_id} interrupted. Stopping sequence." >&2
        return "$exit_code"
      fi

      if (( attempt >= MAX_PHASE_ATTEMPTS )); then
        echo
        echo "Phase ${phase_id} failed after ${MAX_PHASE_ATTEMPTS} attempts. Stopping sequence." >&2
        return "$exit_code"
      fi

      echo
      echo "Phase ${phase_id} incomplete on attempt ${attempt}/${MAX_PHASE_ATTEMPTS}. Rerunning..." >&2
      attempt=$((attempt + 1))
    fi
  done
}

for i in "${!PHASE_ORDER[@]}"; do
  if (( i < START_INDEX || i > END_INDEX )); then
    continue
  fi

  phase="${PHASE_ORDER[$i]}"
  run_phase "$phase" || exit $?
done

echo
echo "taxonomy pipeline complete."
