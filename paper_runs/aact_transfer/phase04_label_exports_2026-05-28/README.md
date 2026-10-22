# AACT Phase 04 label exports (exported 2026-05-28)

Compact exports of the Phase 04 label caches from three labeling runs over the
same 1,979 AACT coronary-intervention text fragments. The full run caches
(`scripts/data/runs/<run-id>/cache/labels_*.jsonl`) are not tracked by Git and
contain model reasoning; these exports keep only what is needed to rebuild the
source-type alignment matrices.

These files support the AACT source type by predicted meta domain comparison
(neutral and cardiology-aware GPT-5.4 runs).

## Runs

| Column prefix / `run` value | Run id | Labeling model | Prompt |
|---|---|---|---|
| `qwen_original` | `aact_qwen35_27b_20260510_taxonomizer_full1979` | `qwen/qwen3.5-9b` (runner default for a 27B taxonomy run) | `scripts/prompts/phase_04_labeling.md` |
| `gpt54_neutral` | `aact_gpt54_neutral_phase04_20260526` | `openai/gpt-5.4` | `scripts/prompts/phase_04_labeling.md` (source-blind) |
| `gpt54_cardio_aware` | `aact_gpt54_cardio_indication_phase04_20260511` | `openai/gpt-5.4` | `scripts/prompts/phase_04_labeling_cardio_indication.md` |

All three runs label against the same Qwen 3.5 27B taxonomy (Phases 00-03 of
the `qwen_original` run).

## Files

- `aact_per_sample_source_and_predicted_meta_sets.csv`: one row per fragment
  (`original_index` 0-1978); source type plus the predicted meta set,
  categories, and label count for each run. No text, no reasoning.
- `aact_per_sample_source_text_and_predicted_meta_sets.csv`: the same with the
  public AACT fragment text. Rows are in pipeline input order, so the `text`
  column can be used to rebuild the text-only pipeline input (see
  `../README.md`).
- `aact_<run>_source_type_by_predicted_meta_record_counts.csv`: source type by
  predicted meta record-count matrices. Each fragment counts at most once per
  meta domain but can appear in several columns if it received several
  meta-level labels.
- `aact_phase04_assignment_long_no_reasoning.csv`: one row per label
  assignment across the three runs; no model reasoning.

## Source-type values

`adverse_event` (AACT `reported_events`, n=1,000), `condition` (`conditions`,
n=300), `intervention` (`interventions`, n=300), `outcome` (`outcomes` and
`design_outcomes`, n=300), `study_conduct` (`drop_withdrawals`, n=79).

## Interpretation guardrail

AACT source type is source-field provenance, not adjudicated semantic ground
truth. Source type was withheld from all model prompts. Use these files for
provenance alignment, discordance, and prompt-sensitivity analyses; do not
report the matrices as accuracy against a gold standard.
