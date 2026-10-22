#!/usr/bin/env bash
# AACT transfer, step 1: full taxonomy pipeline (Phases 00-09) with Qwen 3.5 27B
# on the 1,979 AACT coronary-intervention text fragments.
# Original run: 2026-05-10 to 2026-05-11 (OpenRouter, zero-data-retention routing).
#
# Usage (from anywhere):
#   paper_runs/aact_transfer/run_aact_taxonomy_full1979_20260510.sh [extra runner args]
# Extra args are passed to scripts/run_taxonomy_pipeline.sh, e.g. --resume,
# --start-phase 09, or --base-url http://localhost:1234/v1 for a local server.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

RUN_ID="$AACT_TAXONOMY_RUN_ID"
MODEL="${AACT_TAXONOMY_MODEL:-qwen/qwen3.5-27b}"
DOMAIN_CONTEXT="Free-text entries from public clinical trial documentation."
mkdir -p "$AACT_LOG_DIR"
LOG="${AACT_LOG_DIR}/${RUN_ID}.log"

aact_require_api_key "$@"

{
  echo "AACT taxonomy full run started at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "run_id=${RUN_ID} model=${MODEL} target=adverse_events"
  aact_prepare_input
  ./scripts/run_taxonomy_pipeline.sh \
    --input "$AACT_INPUT_PARQUET" \
    --run-id "$RUN_ID" \
    --model "$MODEL" \
    --target adverse_events \
    --domain-context "$DOMAIN_CONTEXT" \
    --concurrency 10 \
    --max-phase-attempts 3 \
    "$@"
  echo "AACT taxonomy full run finished at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} 2>&1 | tee -a "$LOG"
