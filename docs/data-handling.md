# Data handling

This page describes how data moved through the models in the study that
Med Taxonomizer was built for, and what the released code does by default.

## What is sent to a model

- The pipeline reads a parquet file with a single `text` column
  (`scripts/prepare_texts.py` writes it). Prompts are built only from these
  text values.
- In the study, only the content of the free-text fields was sent. No patient
  ID, case ID, centre, date, or other structured variable was included in any
  prompt.
- Inside the pipeline, texts are tracked by their row position
  (`original_index`) and, in Phase 04, by random per-request UUIDs. Linkage from
  row position back to source records was kept locally, outside this
  repository, and was never sent to a model.
- Source-type or other provenance labels are not part of the model input. In
  the AACT transfer demonstration the source table of each fragment was
  withheld from all prompts and used only afterwards.

## Hosted models: OpenRouter with zero data retention

All hosted calls go through OpenRouter's OpenAI-compatible API
(`https://openrouter.ai/api/v1`, `lib/config.py`). Every remote request built by
this repository includes the zero-data-retention routing flag:

```python
client.beta.chat.completions.parse(
    model=model,
    messages=messages,
    response_format=ResponseModel,
    extra_body={"provider": {"zdr": True}, ...},
)
```

- Phase scripts (`scripts/phase_*.py`) add `"provider": {"zdr": True}` to
  `extra_body` whenever no `--base-url` is given.
- `lib/llm_client.OpenRouterClient` and `AsyncOpenRouterClient` default to
  `zero_data_retention=True` and add the same block; `OpenRouterClient.embed`
  sends `"provider": {"zdr": True}` in the JSON body.

With this flag OpenRouter routes a request only to upstream providers that
have a zero-data-retention policy. The code does not pin a specific upstream
provider.

In the study, every proprietary model (GPT-5.4, GPT-4o, GPT-4o-mini for the
embedding-experiment translations, Claude Opus 4.6, Gemini 3.1 Pro, and the
other hosted comparators) and the hosted Qwen 3.5 runs (`qwen/qwen3.5-27b`,
`qwen/qwen3.5-9b`: taxonomy induction, taxonomy-phase labeling, benchmark
classification) were called this way.

## Local models

- MedGemma 27B (`alibayram/medgemma:27b`) ran locally through Ollama.
- Qwen 3.5 27B (`unsloth/qwen3.5-27b`, GGUF Q8_0) ran through LM Studio for the 300-entry held-out comparison. Exploratory taxonomy iterations also used a local LM Studio server.

For local runs the client connects only to the given `--base-url`, sends a
placeholder key, and sends no `provider` block.

## Endpoint-agnostic design

Every phase accepts `--base-url`. The same pipeline, prompts and validation run
against OpenRouter or any OpenAI-compatible server (LM Studio, vLLM, llama.cpp
server, Ollama), so an institution can keep all inference on its own hardware.
See [qwen-configuration.md](qwen-configuration.md) for the local command line.

## What stays out of the repository

`.gitignore` excludes `data/`, `scripts/data/` (run artifacts, caches, SQLite
label databases), `logs/`, `notes/`, `.env` files, and spreadsheet and database
formats. Source
datasets, identifiers, run logs, model caches, and reviewer decisions should not
be committed. The only data files in the repository are the synthetic example
(`examples/synthetic/`) and derived exports of public ClinicalTrials.gov (AACT)
text (`paper_runs/aact_transfer/`).

## Your own deployment

Whether hosted inference is acceptable for your data depends on your approvals
and data-protection rules, not on this code. If it is not, use a local
endpoint. If it is, keep the ZDR flag and send only the text column.
