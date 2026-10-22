#!/usr/bin/env bash
# AACT transfer, step 2b: cardiology-aware Phase 04 relabeling with GPT-5.4 on
# the fixed Qwen 3.5 27B taxonomy from step 1 (prompt-context sensitivity run).
# Original run: 2026-05-11 (OpenRouter, zero-data-retention routing).
#
# Usage: paper_runs/aact_transfer/run_aact_phase04_cardio_gpt54_20260511.sh [extra phase-04 args]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

RUN_ID="aact_gpt54_cardio_indication_phase04_20260511"
mkdir -p "$AACT_LOG_DIR"
LOG="${AACT_LOG_DIR}/${RUN_ID}.log"

aact_require_api_key "$@"

{
  echo "AACT Phase 04 cardiology-aware GPT-5.4 run started at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "run_id=${RUN_ID} taxonomy_run=${AACT_TAXONOMY_RUN_ID}"
  aact_prepare_input
  aact_copy_taxonomy "$RUN_ID"
  uv run python scripts/phase_04_labeling_cardio_indication.py \
    --input "$AACT_INPUT_PARQUET" \
    --run-id "$RUN_ID" \
    --model openai/gpt-5.4 \
    --taxonomy-model qwen/qwen3.5-27b \
    --domain-context "Coronary artery disease / acute coronary syndrome / myocardial infarction / PCI / stenting / angioplasty / coronary revascularization trial text fragments. Conditions should be limited to baseline cardiology indications, eligibility diagnoses, or pre-existing comorbidities; prefer adverse_events for plausible incident complications." \
    --language-note none \
    --batch-size 10 \
    --concurrency 10 \
    --request-timeout-seconds 300 \
    "$@"
  echo "AACT Phase 04 cardiology-aware GPT-5.4 run finished at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} 2>&1 | tee -a "$LOG"
