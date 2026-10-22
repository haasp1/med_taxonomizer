# Changelog

## 1.1.0 - 2026-10-22

### Added

- `paper_runs/aact_transfer/`: run wrappers, taxonomy-tree exports, and
  per-sample Phase 04 label exports for the public AACT coronary-intervention
  transfer demonstration.
- `scripts/phase_04_labeling_cardio_indication.py` and
  `scripts/prompts/phase_04_labeling_cardio_indication.md`: the cardiology-aware
  Phase 04 variant used in the AACT prompt-sensitivity run.
- `docs/qwen-configuration.md` and `configs/qwen35_27b_{openrouter,local}.yaml`:
  the Qwen 3.5 27B/9B settings used in the study (model ids per endpoint,
  per-phase sampling, concurrency, batching, retries, validation, ZDR flag, and
  local-server usage).
- `docs/data-handling.md`: what is sent to models, OpenRouter zero-data-retention
  routing, which models ran locally; linked from the README and website.
- Tests that every phase script starts when run directly and that the runner
  passes `--input` to the phases that read text.

### Fixed

- Phase scripts failed with `ModuleNotFoundError: No module named 'lib'` when
  run as `python scripts/<phase>.py`; added the `scripts/lib -> ../lib` link.
- Phases 02 and 04 read a hard-coded legacy input path; they now accept
  `--input` (default `data/inputs/texts.parquet`), and
  `run_taxonomy_pipeline.sh` passes `--input` to Phases 00, 02 and 04.
- `scripts/run_taxonomy_phases_02_03.sh` pointed to a non-existent directory.
- Replaced a private network address in examples with `http://localhost:1234/v1`.
- `.gitignore` now ignores `logs/` and `notes/`, as the README already stated.

## 1.0.0 - 2026-06-16

- First public release (Zenodo DOI 10.5281/zenodo.21275129).
