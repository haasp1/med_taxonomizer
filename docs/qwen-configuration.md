# Qwen 3.5 configuration

This page lists the Qwen 3.5 settings used in the study and in the released
code. Values are taken from the code (file and constant names are given so they
can be checked). Settings not recorded for exploratory runs are identified below.

Machine-readable copies: [`configs/qwen35_27b_openrouter.yaml`](../configs/qwen35_27b_openrouter.yaml)
and [`configs/qwen35_27b_local.yaml`](../configs/qwen35_27b_local.yaml). These
files document the settings; the pipeline does not read them.

## Model identifiers

| Model | Endpoint | Model id as sent | Used for |
|---|---|---|---|
| Qwen 3.5 27B | OpenRouter (`https://openrouter.ai/api/v1`) | `qwen/qwen3.5-27b` | taxonomy phases 00-03 and 05-09; benchmark classification |
| Qwen 3.5 9B | OpenRouter | `qwen/qwen3.5-9b` | Phase 04 flat labeling inside the taxonomy pipeline; benchmark classification |
| Qwen 3.5 27B | local LM Studio server (OpenAI-compatible) | `unsloth/qwen3.5-27b` | 300-entry held-out classification; exploratory taxonomy iterations |

The taxonomy-discovery, AACT and 1,000-entry benchmark runs used OpenRouter with the zero-data-retention routing flag. The 300-entry held-out comparison used local Qwen 3.5 27B (Unsloth GGUF Q8_0) through LM Studio: temperature 0.6, batch size 25 and three runs. Its classifier is maintained in the study repository, separately from the taxonomy pipeline documented here.

The 9B labeling model is chosen by `scripts/run_taxonomy_pipeline.sh`: when
`--model` contains `qwen` and `27b`, `--labeling-model` defaults to the same id
with `27b` replaced by `9b`. The run state of the GSR-ET Phase 04 labeling
and the AACT transfer run log both record `qwen/qwen3.5-9b` as the labeling
model. Phases 05-07 read the Phase 04 labels but call the `--model` (27B).

## Hosted calls: zero data retention

Every remote (OpenRouter) request built by this repository includes

```python
extra_body={"provider": {"zdr": True}}
```

(`scripts/phase_*.py`; `lib/llm_client.py`, `OpenRouterClient` and
`AsyncOpenRouterClient` default to `zero_data_retention=True`; embedding calls
send `"provider": {"zdr": True}` in the JSON body). OpenRouter then routes the
request only to providers with a zero-data-retention policy. Local calls
(`--base-url`) do not send the `provider` block.

## Sampling parameters by phase

`lib/model_naming.resolve_temperature()` forces `temperature=0.6` for any model
id containing `qwen`. Phases 04-08 bypass it for Qwen and use their own
constants. The values below apply to both `qwen/qwen3.5-27b` and
`qwen/qwen3.5-9b`.

| Phase | Script | temperature | top_p | top_k | presence_penalty | reasoning (remote only) |
|---|---|---|---|---|---|---|
| 00 meta discovery | `phase_00_meta_discovery.py` | 0.6 | not sent | not sent | not sent | `{"effort": "high"}` |
| 01 meta consolidation | `phase_01_meta_consolidation.py` | 0.6 | not sent | not sent | not sent | `{"effort": "high"}` |
| 02 category discovery | `phase_02_category_discovery.py` | 0.6 | not sent | not sent | not sent | `{"effort": "high"}` |
| 03 category consolidation | `phase_03_category_consolidation.py` | 0.6 | not sent | not sent | not sent | `{"effort": "high"}` |
| 04 labeling | `phase_04_labeling.py` (`QWEN_LABELING_*`) | 0.6 | 0.95 | 20 | 1.5 | not sent |
| 05 review "other" | `phase_05_review_other.py` (`QWEN_*`) | 0.6 | 0.95 | 20 | 1.5 | not sent |
| 06 split audit | `phase_06_split_audit.py` (`QWEN_*`) | 0.8 | 0.95 | 20 | 1.5 | not sent |
| 07 subcategory discovery | `phase_07_subcategory_discovery.py` (`QWEN_*`) | 0.8 | 0.95 | 20 | 1.5 | not sent |
| 08 tree consolidation | `phase_08_tree_consolidation.py` (`QWEN_*`) | 0.8 | 0.95 | 20 | 1.5 | `{"effort": "xhigh"}` |
| 09 ambiguity pruning | `phase_09_ambiguity_pruning.py` | 0.6 | not sent | not sent | not sent | not sent |

`top_k` is not an OpenAI parameter, so it goes in `extra_body` (`{"top_k": 20}`)
next to the `provider` block. `reasoning` is also sent in `extra_body`. On local
servers the same temperature, top_p, presence_penalty and `top_k` values are
sent; `provider` and `reasoning` are not.

Other generation settings:

- `max_tokens=25000` on remote structured calls (`lib/llm_client.structured_parse_call`).
  This default was added to the public package; the study repository's client
  used for the GSR-ET runs sent no `max_tokens`.
- Phase 04 system prompt: `DEFAULT_SYSTEM_PROMPT` in `phase_04_labeling.py`
  ("You are a concise, efficient, decisive assistant. ...").
- Phase 00/02 sampling: 200 texts per seed; Phase 00 uses seeds
  `42, 123, 456, 789, 2024, 4096`, Phase 02 uses `42, 123, 456` per meta
  category. Phases 05-07 use their own `DEFAULT_N_SAMPLES` / `DEFAULT_SEEDS`.

### Benchmark classification (study repository, not in this package)

The 1,000-entry classification benchmark was run with the study's
`05_model_benchmark.py`. For Qwen it sent `temperature=0.6` (through
`resolve_temperature`), no top_p/top_k/presence_penalty, no reasoning
parameter, and `extra_body={"provider": {"zdr": True}}`. Script defaults are
batch size 25, 3 runs per sample and up to 3 attempts per call; the run caches
record concurrency 6 for all OpenRouter models.

## Concurrency and batching

| Setting | Default | Source |
|---|---|---|
| Remote concurrency, Qwen 27B | 8 | `lib/model_naming.resolve_remote_concurrency_default` |
| Remote concurrency, Qwen 9B | 20 | same |
| Remote concurrency, other models | 10 | same |
| Remote concurrency, Phase 09 | 8 | `phase_09_ambiguity_pruning.DEFAULT_CONCURRENCY` |
| Local concurrency (`--base-url` set) | 1 | `lib/llm_client.resolve_concurrency` |
| `--concurrency N` | overrides all of the above | every phase script |
| Phase 04 batch size | 10 samples per request | `phase_04_labeling.DEFAULT_BATCH_SIZE` |
| Phase 05 relabel batch size | 25 | `phase_05_review_other.DEFAULT_RELABEL_BATCH_SIZE` |

The AACT transfer run passed `--concurrency 10` to every phase
(`paper_runs/aact_transfer/run_aact_taxonomy_full1979_20260510.sh`).

## Retries, timeouts and output validation

- Structured output: remote calls use the OpenAI SDK `beta.chat.completions.parse`
  with a Pydantic response model. Local calls append the JSON schema to the
  prompt and extract the first valid JSON object from the reply, after
  removing `<think>...</think>` blocks (`lib/llm_client.extract_pydantic_json`).
- Phases 00-03: up to 3 attempts per call (`STRUCTURED_OUTPUT_MAX_ATTEMPTS`),
  retried on timeouts, length-limit stops, JSON/Pydantic validation errors and
  empty parses, with a linear back-off of 1 s x attempt number.
- Phases 04-09: up to 4 attempts per batch (`DEFAULT_BATCH_MAX_ATTEMPTS`,
  `run_with_validation_repair`). The last 2 attempts include a repair message
  with the validation error and the previous parsed response.
- Phase 04 validation: every sample UUID sent must come back exactly once (no
  missing, duplicate or unknown UUIDs), and every meta/category pair must exist
  in the taxonomy. Category names are also constrained through `Literal` types
  in the dynamically built response model. Batches that still fail are written
  to a failed-batch manifest and can be rerun with `--rerun-missing-only`.
- Request timeout: 180 s per attempt (`DEFAULT_REQUEST_TIMEOUT_SECONDS`), 300 s
  in Phase 09. The AACT GPT-5.4 relabeling runs used
  `--request-timeout-seconds 300`. Local clients are created without an HTTP
  timeout.
- Phase-level retries: `run_taxonomy_pipeline.sh --max-phase-attempts`
  (default 3). Attempts after the first run in resume mode.

## Running against a hosted endpoint (OpenRouter)

```bash
export OPENROUTER_API_KEY=...
./scripts/run_taxonomy_pipeline.sh \
  --input data/inputs/texts.parquet \
  --run-id qwen27b_openrouter \
  --model qwen/qwen3.5-27b \
  --target adverse_events \
  --domain-context "Clinical free-text fragments for taxonomy induction." \
  --resume
```

This uses `qwen/qwen3.5-9b` for Phase 04 unless `--labeling-model` is given.

## Running against a local OpenAI-compatible server

Any server that implements `/v1/chat/completions` works (LM Studio, vLLM,
llama.cpp server, Ollama's OpenAI endpoint). Pass its base URL and the model id
the server exposes:

```bash
./scripts/run_taxonomy_pipeline.sh \
  --input data/inputs/texts.parquet \
  --run-id qwen27b_local \
  --base-url http://localhost:1234/v1 \
  --model unsloth/qwen3.5-27b \
  --labeling-model unsloth/qwen3.5-27b \
  --target adverse_events \
  --resume
```

Notes:

- `--base-url` is forwarded to every phase. No OpenRouter key is needed; the
  client sends the placeholder key `lm-studio`.
- Set `--labeling-model` explicitly. The automatic 27B-to-9B substitution only
  rewrites the id string, and the local server may not serve a 9B model.
- Concurrency defaults to 1 locally; raise it with `--concurrency` if the
  server can handle parallel requests.
- Artifact file names use a provider tag derived from the model id (`qwen` for
  any id containing `qwen`), so local and hosted runs should use different
  `--run-id` values.

## Scope of recorded settings

- Quantisation of the local exploratory taxonomy runs was not recorded; the held-out classification used GGUF Q8_0.
- Hardware (GPU/VRAM), LM Studio version and context-length setting for the
  local exploratory runs.
- Server-side sampling defaults in LM Studio for those runs (the client sends
  the values above, but a server can override them).
- The upstream inference provider(s) that OpenRouter selected for each
  `qwen/qwen3.5-*` request, and the provider-side quantisation. The code does
  not pin a provider; it only requires ZDR.
