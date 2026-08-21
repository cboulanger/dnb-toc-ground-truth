# Pluggable inference endpoints for KISSKI/MPCDF/etc.

## 1. Problem

`evaluation/scripts/generate_dnb_toc_ground_truth.py` and
`evaluation/refresh_llm_cache.py` both talk to KISSKI (Academic Cloud)
exclusively -- `evaluation/kisski.py`'s discovery/selection helpers, and
an `AsyncOpenAI(base_url=DEFAULT_KISSKI_BASE_URL, api_key=os.environ["KISSKI_API_KEY"])`
client constructed ad hoc in each script. KISSKI's per-response token
limit is restrictive enough to be a real constraint on the dnb-toc-only
vision extraction gate (`evaluation/experiments/dnb-toc-ground-truth.md`'s
`max_tokens` escalation-on-truncation logic exists specifically because of
this), and its shared-pool rate limits (`evaluation/experiments/dnb-toc-ground-truth.md`'s
day/hour/minute quota-exhaustion write-ups) have repeatedly stalled batch
runs for hours.

MPCDF's LLM Inference Service (`https://llm.mpcdf.mpg.de`) lets a user
spin up a dedicated vLLM or Ollama endpoint on Viper-GPU/DAIS for up to 8
hours -- both frameworks speak the OpenAI-compatible chat completions API,
the same shape both scripts already call. Neither script can use it today
without hardcoding a second, parallel code path.

## 2. Goal

Let both scripts run against any OpenAI-compatible endpoint -- KISSKI
(today's default, unchanged), one or more MPCDF sessions, or any future
provider -- selected per invocation via a CLI flag, with secrets supplied
through environment variables (never CLI args, which are visible via
`ps`). No script-specific code should be needed to add a new provider:
if it speaks the OpenAI chat completions API, it works.

## 3. Non-goals

- **Not a `Provider` class hierarchy.** KISSKI's discovery/demand-aware
  selection (`fetch_kisski_models`, `select_top5`/`select_gap_fill`/
  `select_full_regen`, the vision-model pattern matching in
  `generate_dnb_toc_ground_truth.py`) has no MPCDF equivalent -- MPCDF has
  no shared pool, no "demand," and no discovery endpoint; you pick and
  deploy exactly one model per session yourself. Forcing MPCDF through a
  `fetch_models()`-shaped interface that always answers "one model,
  demand 0" would be an abstraction with no real second implementation.
  `kisski.py` stays as is and remains the default path; everything else
  is a flat `(base_url, api_key, model_id)` triple.
- **Not automatic provider fallback or failover.** The user decides which
  endpoint(s) are live for a given run (an MPCDF session either exists
  right now or it doesn't) and says so explicitly via `--endpoint`.
- **Not new handling for session expiry.** An MPCDF endpoint disappearing
  mid-run surfaces as ordinary request failures, already handled by each
  script's existing per-book catch-log-continue behavior and (for
  `generate_dnb_toc_ground_truth.py`) `_still_needs_a_decision`'s skip
  logic on the next invocation. See §6.

## 4. Architecture

### 4.1 New module: `evaluation/inference_endpoints.py`

```python
@dataclass(frozen=True)
class ModelEndpoint:
    label: str        # e.g. "MPCDF_A" -- the alias itself, used in log/print output
    model_id: str
    client: AsyncOpenAI


def resolve_endpoint_from_env(alias: str, *, timeout: float = 90.0) -> ModelEndpoint:
    """Reads `<ALIAS>_BASE_URL`, `<ALIAS>_API_KEY`, `<ALIAS>_MODEL` from
    the environment and builds a ModelEndpoint. Raises a clear
    (KeyError-derived) error naming exactly which env var is missing and
    the three-variable convention, rather than a bare KeyError."""
```

`timeout` defaults to the same 90.0s already hardcoded for KISSKI's
`AsyncOpenAI` client in `generate_dnb_toc_ground_truth.py`
(`evaluation/experiments/dnb-toc-ground-truth.md`'s "second stall" -- long
enough for a real vision call, short enough to bound a retry loop's worst
case). Not made configurable per alias for now -- MPCDF's dedicated GPU
should if anything be less prone to the KISSKI stuck-connection failure
mode this value was chosen to bound; revisit only if it proves wrong in
practice.

This is the entire new module. No `KisskiProvider`/`MpcdfProvider`
classes -- an alias pointing at a manually-picked KISSKI model (bypassing
`fetch_kisski_models` discovery entirely) resolves through the exact same
function as an MPCDF alias.

### 4.2 CLI convention (both scripts)

```
--endpoint ALIAS   # repeatable
```

Each `ALIAS` must have `<ALIAS>_BASE_URL`, `<ALIAS>_API_KEY`,
`<ALIAS>_MODEL` set in the environment, e.g.:

```bash
export MPCDF_A_BASE_URL="https://<mpcdf-session-host>/v1"
export MPCDF_A_API_KEY="..."
export MPCDF_A_MODEL="Qwen/Qwen3-VL-30B-A3B-Instruct"
```

Omitting `--endpoint` entirely preserves every existing default: KISSKI
discovery/selection runs exactly as it does today. This is a pure
addition, not a breaking change to either script's current CLI.

### 4.3 `generate_dnb_toc_ground_truth.py`

The one real structural change. Today `_run_book`/`_run_all` take one
shared `client` plus `models: tuple[str, str]`, because both vision
models sit behind KISSKI's single base URL. That assumption breaks once
the two independent models can live behind two different endpoints (e.g.
two separate MPCDF sessions, or one MPCDF + one KISSKI) -- so both
functions' signatures change to take `endpoints: tuple[ModelEndpoint,
ModelEndpoint]` instead, and the per-model call inside `_run_book` uses
`endpoint.client`/`endpoint.model_id` rather than a shared `client` +
loop variable `model`.

`_generate` resolves `endpoints` one of two ways:
- `args.endpoint` has exactly 2 values → `tuple(resolve_endpoint_from_env(a) for a in args.endpoint)`.
- `args.endpoint` is empty → today's path (`_pick_models` +
  `fetch_kisski_models`), each of the two selected model ids wrapped in a
  `ModelEndpoint` sharing one KISSKI client, preserving current behavior
  and output exactly.
- `args.endpoint` has 1 or 3+ values → argument error: the two-independent-
  model gate design requires exactly two.

Nothing else in the gate/arbitration path changes -- `vision_extract_toc_entries`,
`gate_book`, caching (keyed on `model_id`, unaffected by which endpoint
served it) are untouched.

### 4.4 `refresh_llm_cache.py`

Smaller change: `_OpenAICompatibleLLMClient` already takes `(model,
base_url, api_key)` generically. Add `--endpoint ALIAS` (repeatable,
`argparse` mutually exclusive with `--mode`). When given, `_main` skips
`fetch_kisski_models`/`select_*` entirely and runs each resolved
`ModelEndpoint` once over the corpus, reusing `_run_book_for_model`'s
existing per-model worker and cache-write path (`model_id` from the
`ModelEndpoint` substitutes for `KisskiModel.id` wherever the worker
reads it). `--mode`'s top5/fill-gaps/full sweep semantics don't apply
here at all -- MPCDF has nothing to sweep, you're always running exactly
the endpoint(s) you named.

## 5. Data flow example

```bash
# two MPCDF sessions running different models, independence preserved
export MPCDF_A_BASE_URL=... MPCDF_A_API_KEY=... MPCDF_A_MODEL=...
export MPCDF_B_BASE_URL=... MPCDF_B_API_KEY=... MPCDF_B_MODEL=...
uv run python evaluation/scripts/generate_dnb_toc_ground_truth.py \
  --endpoint MPCDF_A --endpoint MPCDF_B --limit 100

# mixed pair: one MPCDF session + one manually-picked KISSKI model
export KISSKI_VISION_BASE_URL="https://chat-ai.academiccloud.de/v1"
export KISSKI_VISION_API_KEY=$KISSKI_API_KEY
export KISSKI_VISION_MODEL="qwen3.6-35b-a3b"
uv run python evaluation/scripts/generate_dnb_toc_ground_truth.py \
  --endpoint MPCDF_A --endpoint KISSKI_VISION --limit 100

# no --endpoint: unchanged today's-default KISSKI auto-select behavior
uv run python evaluation/scripts/generate_dnb_toc_ground_truth.py --limit 100
```

## 6. Error handling / session ephemerality

An MPCDF session's 8-hour lifetime (or an endpoint not yet up) surfaces
as ordinary connection/timeout/5xx failures on individual requests --
indistinguishable, from the calling script's point of view, from any
other transient failure `_call_with_retry` already handles via blind
exponential backoff (KISSKI's rate-limit-header-aware retry path,
`_binding_rate_limit_window`, simply won't find those headers on an MPCDF
response and falls through to the same blind backoff). A book whose
retries are exhausted mid-session is reported as a failure, not cached;
`generate_dnb_toc_ground_truth.py`'s existing `_still_needs_a_decision`
skip-logic naturally retries it on the next invocation once a fresh
session is up. No new resilience code is needed.

## 7. Testing

- `resolve_endpoint_from_env`: unit tests for all-vars-present (returns a
  correctly-populated `ModelEndpoint`), each single missing var (clear
  error naming that var), and confirming the returned client's
  `base_url`/`api_key` match the env values.
- `generate_dnb_toc_ground_truth.py`: extend the existing test style in
  `tests/test_generate_dnb_toc_ground_truth.py` -- `_run_book` given two
  `ModelEndpoint`s backed by distinct mock `AsyncOpenAI` clients, asserting
  each side's call reaches its own client, not the other's; `_generate`'s
  endpoint-count validation (0, 1, 2, 3 values for `--endpoint`).
- `refresh_llm_cache.py`: a test that `--endpoint` bypasses
  `fetch_kisski_models` entirely (mock it and assert it's never called)
  and drives `_run_book_for_model` with the resolved endpoint's
  `model_id`.

## 8. Documentation

- `evaluation/scripts/README.md`: document `--endpoint` for both scripts,
  including the three-env-var convention and the exactly-two-endpoints
  requirement for `generate_dnb_toc_ground_truth.py`.
- `evaluation/experiments/dnb-toc-ground-truth.md`: once this is used for
  a real MPCDF-backed run, record the outcome there (same "Current
  status" / "History" convention as every other run in that file) --
  not part of this design, just noting where that write-up belongs.

## 9. Open questions

None -- scope confirmed via brainstorming session on 2026-08-18: shared
abstraction covering both `generate_dnb_toc_ground_truth.py` and
`refresh_llm_cache.py`; MPCDF confirmed able to run two endpoints
(different models) simultaneously within one 8h window, so the
two-independent-vision-model gate design carries over unchanged;
provider/endpoint selection is via an explicit repeatable `--endpoint`
CLI flag per invocation, not a config file or auto-detection; MPCDF
endpoints are config-driven (no model-discovery call), unlike KISSKI's
real discovery/demand logic which is left untouched.
