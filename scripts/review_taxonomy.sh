#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if command -v uv >/dev/null 2>&1; then
  uv run python scripts/review_taxonomy.py "$@"
elif [[ -x .venv/bin/python ]]; then
  .venv/bin/python scripts/review_taxonomy.py "$@"
else
  python3 scripts/review_taxonomy.py "$@"
fi
