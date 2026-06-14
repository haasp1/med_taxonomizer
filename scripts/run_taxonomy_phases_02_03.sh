#!/usr/bin/env bash
# Run Phase 02 (Category Discovery) then Phase 03 (Category Consolidation) sequentially.
# Usage: bash experiments/taxonomy_taxonomy_labeling/run_phases_02_03.sh [--base-url URL] [--model MODEL]
#
# Example (local):
#   bash experiments/taxonomy_taxonomy_labeling/run_phases_02_03.sh --base-url http://192.168.0.70:1234/v1 --model unsloth/qwen3.5-27b
#
# Example (OpenRouter, default):
#   bash experiments/taxonomy_taxonomy_labeling/run_phases_02_03.sh

set -euo pipefail

EXTRA_ARGS=("$@")

echo "========================================="
echo " Phase 02: Category Discovery (Ensemble)"
echo "========================================="
uv run python experiments/taxonomy_taxonomy_labeling/phase_02_category_discovery.py "${EXTRA_ARGS[@]}"

echo ""
echo "========================================="
echo " Phase 03: Category Consolidation"
echo "========================================="
uv run python experiments/taxonomy_taxonomy_labeling/phase_03_category_consolidation.py "${EXTRA_ARGS[@]}"

echo ""
echo "Done. Phases 02 + 03 complete."
