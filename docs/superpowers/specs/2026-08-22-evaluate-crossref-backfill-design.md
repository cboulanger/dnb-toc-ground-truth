# Backfilling missing llm-cache entries in evaluate_crossref.py

Status: proposed
Date: 2026-08-22

## Context

`cli/evaluate_crossref.py --model <model> [--all-models]` scores a
model's raw llm-cache extraction (`data/corpus/pilot/llm-cache/v2/`)
against the committed Crossref evaluation corpus
(`dnb_toc_ground_truth.crossref_evaluation.evaluate_model_corpus`). A
book with no cache entry for that model is reported in
`keys_with_no_cache_entry` and simply excluded from scoring -- there is
currently no way to populate those entries except running the full
`cli/generate_ground_truth.py` bulk-gate pipeline, which deliberately
*excludes* every eval-tier book
(`_still_needs_a_decision`/`eval_tier_ids.json`) and requires at least
two models to agree, neither of which is wanted here: this is a
single-model, eval-tier-only, no-gating raw-extraction run purely to
measure one model's (here: `numind/NuExtract3`, freshly wired up per
`docs/superpowers/specs/2026-08-22-nuextract-template-mode-integration-design.md`)
extraction quality against the Crossref evaluation corpus's ~54 books.

## Goals

- `evaluate_crossref.py --model numind/NuExtract3 --backfill` populates
  every missing `<key>.numind__NuExtract3.json` llm-cache entry for the
  Crossref-evaluation-corpus books, then scores as today.
- Reuses `evaluate_model_corpus`'s own book-selection logic (its
  `no_cache` list) rather than re-deriving which books need a cache
  entry -- one source of truth for "which books are in scope."
- `--backfill` combines with multiple `--model` invocations, backfilling
  each independently.

## Non-goals

- No rate-limit-aware retry/backoff machinery
  (`generate_ground_truth.py`'s `_call_with_retry`) -- this is a manual,
  interactive, one-off utility run over ~54 books for one model, not an
  unattended batch job. One attempt per book; a failure is printed and
  skipped, not retried.
- No text-mode (`--use-text`-style OCR'd-input) backfill support --
  every model this has ever been run against (Pixtral, Qwen3-Omni,
  NuExtract3) is vision-input. Add it later if a text-only model
  actually needs backfilling.
- No shared extraction-dispatch module factored out of
  `generate_ground_truth.py`'s `_run_book`/`_call`. The
  `extraction_api == "nuextract"` branch (4 lines: pick
  `nuextract_vision_extract_toc_entries` vs `vision_extract_toc_entries`)
  is small enough to duplicate inline in the new backfill function,
  matching this project's existing tolerance for small parallel
  duplication across `vision.py`/`ocr.py`/`nuextract.py` (each keeps its
  own `_MAX_TOKENS`/`_MAX_TOKENS_RETRY` constants rather than sharing
  them).

## Design

### 1. `src/dnb_toc_ground_truth/crossref_evaluation.py`

New function, placed after `evaluate_model_corpus`:

```python
async def backfill_model_cache(
    model: str, endpoint: ModelEndpoint, cache_directory: Path,
) -> tuple[list[str], list[str]]:
    """For every Crossref-evaluation-corpus book missing a llm-cache
    entry for `model` (reuses evaluate_model_corpus's own no_cache list
    -- the exact same book-selection logic scoring already trusts),
    extracts once via `endpoint` and writes the cache entry. Returns
    (succeeded_keys, failed_keys) -- a failure (missing PDF, network
    error, unparseable response) is printed and skipped, not retried;
    this is a manual one-off utility run, not generate_ground_truth.py's
    unattended batch job. `model` and `endpoint.model_id` are expected
    to match (the caller resolved `endpoint` FOR this `model`); kept as
    two separate parameters rather than reading `endpoint.model_id`
    directly so the cache is written under exactly the model id the
    caller/CLI asked to backfill, not whatever string the endpoints file
    happened to resolve it to."""
    _, missing_keys = evaluate_model_corpus(model)
    succeeded, failed = [], []
    for key in missing_keys:
        pdf_path = corpus.pdf_path(key)
        if not pdf_path.exists():
            print(f"[backfill] {key}: skipped, no PDF at {pdf_path}")
            failed.append(key)
            continue
        try:
            if endpoint.extraction_api == "nuextract":
                entries = await nuextract_vision_extract_toc_entries(
                    pdf_path, endpoint.model_id, endpoint.client,
                    use_instructions=endpoint.extraction_instructions,
                )
            else:
                entries = await vision_extract_toc_entries(pdf_path, endpoint.model_id, endpoint.client)
        except Exception as exc:  # noqa: BLE001 -- one book's failure must not abort the whole backfill
            print(f"[backfill] {key}: failed -- {type(exc).__name__}: {exc}")
            failed.append(key)
            continue
        if entries:
            vision.write_cached_llm_entries(cache_directory, key, model, entries, kind="vision")
            succeeded.append(key)
        else:
            print(f"[backfill] {key}: extraction returned no entries, not cached")
            failed.append(key)
    return succeeded, failed
```

Imports gained by this module: `ModelEndpoint` (from `.inference`,
type-only), `nuextract_vision_extract_toc_entries` (from `.nuextract`),
`vision_extract_toc_entries` (from `.vision` -- `vision` itself is
already imported).

### 2. `cli/evaluate_crossref.py`

New flags:
- `--backfill` (store_true) -- only meaningful alongside `--model`.
- `--endpoints-file` (default `.endpoints`, or config file's
  `"endpoints_file"` key -- same convention/default as
  `generate_ground_truth.py`).

In `main()`, before the existing scoring loop: if `args.backfill` and
`args.model`, for each requested model id: resolve it against
`--endpoints-file` via `inference.load_endpoint_entries` +
`inference.resolve_model_endpoints([model], "vision", entries)`
(raises `ValueError` naming the model if no endpoint matches -- same
error-naming convention every other endpoint-resolution call site in
this project already follows, not caught/swallowed here), then
`asyncio.run(backfill_model_cache(model, endpoint, corpus.llm_cache_dir()))`,
printing a one-line summary (`f"[backfill] {model}: {len(succeeded)} written, {len(failed)} failed"`).
Scoring then proceeds exactly as today, now seeing the freshly-written
cache entries.

## Error handling

- A model name passed to `--backfill` with no matching `--endpoints-file`
  entry: `resolve_model_endpoints` raises `ValueError` naming the model
  id, propagating up and aborting the whole `evaluate_crossref.py`
  invocation before any extraction call -- consistent with every other
  script's endpoint-resolution failure behavior (fail loud and early,
  not partially).
- Any single book's extraction failure (missing PDF, network error,
  unparseable response, empty result) is printed and skipped -- the
  backfill continues with the remaining books, and scoring afterward
  simply reports that book under `keys_with_no_cache_entry` again, same
  as it does today for any never-backfilled book.

## Testing

Follows `tests/test_crossref_evaluation.py`'s existing conventions:

- `backfill_model_cache`: a fake `AsyncOpenAI`-shaped client (same
  `MagicMock`/`AsyncMock` pattern as `tests/test_vision.py`/
  `tests/test_nuextract.py`), a small fixture set of
  `data/corpus/pilot/evaluation/<key>.expected.json` + PDF files with
  one book already cached (must be skipped, not re-extracted) and one
  book missing (must be extracted and cached). Separate test asserting
  a missing PDF is reported in `failed` without raising. Separate test
  asserting `endpoint.extraction_api == "nuextract"` routes through
  `nuextract_vision_extract_toc_entries` (mocked/patched, asserting
  `use_instructions` was passed through), and the empty-`extraction_api`
  case routes through the ordinary `vision_extract_toc_entries`.
- CLI wiring in `cli/evaluate_crossref.py`: one test asserting
  `--backfill` without a matching endpoint raises before any scoring
  happens; not more than that -- the actual backfill logic is already
  covered at the library level above.
