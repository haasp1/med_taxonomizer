# AACT public transfer demonstration

Run wrappers and derived exports for the public transfer demonstration on
coronary-intervention trial text from the Aggregate Analysis of
ClinicalTrials.gov (AACT). Everything here comes from public ClinicalTrials.gov
text; no registry or patient data is included.

AACT is used here to show that the workflow runs end to end on a different
public corpus. It is not an external validation of the GSR-ET stroke
complication taxonomy.

## Contents

| Path | What it is |
|---|---|
| `run_aact_taxonomy_full1979_20260510.sh` | Step 1: full taxonomy pipeline, Phases 00-09, Qwen 3.5 27B (Phase 04 labeling with Qwen 3.5 9B, the runner default) |
| `run_aact_phase04_neutral_gpt54_20260526.sh` | Step 2a: source-blind Phase 04 relabeling with GPT-5.4 on the fixed Qwen taxonomy |
| `run_aact_phase04_cardio_gpt54_20260511.sh` | Step 2b: cardiology-aware Phase 04 relabeling with GPT-5.4 (prompt-context sensitivity) |
| `common.sh` | Shared helpers: input preparation, API-key check, taxonomy copy |
| `taxonomy_tree_exports_2026-05-28/` | Tree artifacts from step 1 (Phase 07-09 JSON and flattened CSV); see its README |
| `phase04_label_exports_2026-05-28/` | Per-sample Phase 04 labels from steps 1, 2a and 2b, plus source-type matrices; see its README |
| `../../scripts/phase_04_labeling_cardio_indication.py` | Phase 04 script for step 2b; identical to `phase_04_labeling.py` except that it loads the prompt below |
| `../../scripts/prompts/phase_04_labeling_cardio_indication.md` | Phase 04 prompt with an added cardiology-indication rule |

## Corpus

Coronary-intervention trials were selected by intersecting trials whose
conditions referenced myocardial infarction, acute coronary syndrome (STEMI,
NSTEMI), coronary artery or heart disease, or unstable angina with trials whose
interventions referenced PCI, stenting, angioplasty, or coronary
revascularization (1,747 trials). Unique text fragments were taken from five
AACT tables: `reported_events`, `outcomes` and `design_outcomes`,
`interventions`, `conditions`, and `drop_withdrawals` (15,601 unique
fragments). The transfer sample has 1,979 fragments: 1,000 adverse-event,
300 condition, 300 intervention, 300 outcome, and all 79 drop/withdrawal
fragments. The source table is kept as a provenance label and is never shown to
a model.

The model input is the `text` column of
`phase04_label_exports_2026-05-28/aact_per_sample_source_text_and_predicted_meta_sets.csv`
in row order (`original_index` 0-1978). The wrappers rebuild the pipeline input
from it with `scripts/prepare_texts.py`. The AACT extraction and sampling
script is not part of this repository.

## Re-running

Prerequisites: `uv sync --extra dev` and an OpenRouter key
(`export OPENROUTER_API_KEY=...` or a `.env` file in the repository root). All
hosted calls send `extra_body={"provider": {"zdr": True}}`
([docs/data-handling.md](../../docs/data-handling.md)).

```bash
# Step 1: taxonomy induction (Phases 00-09), Qwen 3.5 27B / 9B via OpenRouter
paper_runs/aact_transfer/run_aact_taxonomy_full1979_20260510.sh

# Step 2a: neutral GPT-5.4 relabeling on the step-1 taxonomy
paper_runs/aact_transfer/run_aact_phase04_neutral_gpt54_20260526.sh

# Step 2b: cardiology-aware GPT-5.4 relabeling on the step-1 taxonomy
paper_runs/aact_transfer/run_aact_phase04_cardio_gpt54_20260511.sh
```

The wrappers can be started from any directory. They:

1. write `data/inputs/aact_coronary_intervention_1979.parquet` (override with
   `AACT_INPUT_PARQUET`) and check that it has exactly 1,979 rows and one
   `text` column matching the frozen export in content and order;
2. for steps 2a and 2b, copy the step-1 Phase 00-03 artifacts from
   `scripts/data/runs/aact_qwen35_27b_20260510_taxonomizer_full1979/intermediate/`
   into the new run directory, so labeling uses the fixed Qwen taxonomy
   (50 categories in six meta domains);
3. run the pipeline and append output to `logs/aact_transfer/<run-id>.log`.

Extra arguments are passed through. Examples: `--resume` or `--start-phase 09`
for step 1, `--rerun-missing-only` for steps 2a/2b, or
`--base-url http://localhost:1234/v1` to use a local OpenAI-compatible server
(see [docs/qwen-configuration.md](../../docs/qwen-configuration.md)).

Outputs go to `scripts/data/runs/<run-id>/` (not tracked by Git). The exported
files in this directory were derived from `intermediate/taxonomy_tree_*_qwen.json`
and `cache/labels_*.jsonl` of those runs.

LLM sampling is stochastic, so a rerun gives a similar but not identical
taxonomy and labels. The exports here are the artifacts reported in the
manuscript.

## Run settings and dates

| Step | Run id | Models | Key settings | Run date (UTC) |
|---|---|---|---|---|
| 1 | `aact_qwen35_27b_20260510_taxonomizer_full1979` | `qwen/qwen3.5-27b` (Phases 00-03, 05-09), `qwen/qwen3.5-9b` (Phase 04) | `--target adverse_events`, `--domain-context "Free-text entries from public clinical trial documentation."`, `--concurrency 10`, `--max-phase-attempts 3` | Phases 00-08: 2026-05-10 to 2026-05-11; Phase 09 re-run alone on 2026-05-11 after three failed validation attempts |
| 2a | `aact_gpt54_neutral_phase04_20260526` | `openai/gpt-5.4`, taxonomy `qwen/qwen3.5-27b` | prompt `phase_04_labeling.md`, domain context "Public AACT coronary-intervention clinical-trial text fragments.", `--language-note none`, batch 10, concurrency 10, timeout 300 s | 2026-05-26 |
| 2b | `aact_gpt54_cardio_indication_phase04_20260511` | `openai/gpt-5.4`, taxonomy `qwen/qwen3.5-27b` | prompt `phase_04_labeling_cardio_indication.md`, coronary-trial domain context (see wrapper), `--language-note none`, batch 10, concurrency 10, timeout 300 s | 2026-05-11 |

All three runs used OpenRouter with zero-data-retention routing. Qwen sampling
parameters per phase are listed in
[docs/qwen-configuration.md](../../docs/qwen-configuration.md). Exports were
generated on 2026-05-28.

## Differences from the original run wrappers

The original wrappers were written for the authors' machine. The released
versions keep every model, prompt, and run parameter the same and change only
the local glue:

- no absolute paths or machine-specific `.env` locations; the repository root is
  found from the script location, and an optional `.env` in the root is loaded;
- the input is passed with `--input` (added to Phases 00, 02 and 04 and to the
  runner) instead of being copied to a fixed legacy path;
- the input is rebuilt from the released per-sample export instead of a local
  AACT extract;
- the step-2b wrapper copies the Qwen Phase 00-03 artifacts explicitly, as the
  step-2a wrapper already did (the original step-2b run used a run directory
  prepared beforehand with the step-1 Qwen taxonomy; its log records the same
  50 categories);
- logs go to `logs/aact_transfer/` and are also printed to the terminal.

The pipeline code on `main` is the project state synced after these runs; the
per-request timeout default was changed to 180 s during step 1, before
Phase 03.
