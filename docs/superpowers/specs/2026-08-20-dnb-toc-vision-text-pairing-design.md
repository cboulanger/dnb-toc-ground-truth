# Pairing a vision model with a text-only model for dnb-toc-only ground truth

Status: proposed
Date: 2026-08-20

## Context

`generate_dnb_toc_ground_truth.py`'s two-vision-model agreement gate
(`evaluation/dnb_toc_matching.py`'s `gate_book`) needs two *independent*
extractions of the same book's TOC pages to trust an automatic pass. Since
switching to MPCDF as a second inference source alongside KISSKI
(`docs/superpowers/specs/2026-08-18-inference-endpoint-abstraction-design.md`),
this session tried four candidate second vision models to pair with
`Qwen/Qwen3-Omni-30B-A3B-Instruct` -- `InternVL2_5-38B` (frequent
word-level misreadings), `Pixtral-12B-2409` (confirmed via direct
page-image spot-check to confidently fabricate plausible-but-wrong text on
dense small-font TOCs, not just garble it), `GLM-4.5V` (never testable --
its ~108B MoE checkpoint's graph-capture time exceeds an 8h session),
`GLM-4.1V-9B-Thinking` (its fixed `<think>...</think><answer>` format
breaks the pipeline's JSON extraction, and even a direct code-level fix
was unreliable at temperature=0.0 and risks the context budget on dense
books) -- see `evaluation/experiments/dnb-toc-ground-truth.md`'s
2026-08-19/20 entries for the full writeups. Every candidate failed a
*different* one of three independent constraints: architectural
independence from Qwen (most open VLMs, including InternVL, are
Qwen-backbone-derived), OCR-quality fit (labs with genuinely independent
lineages mostly aren't optimized for dense document transcription), and
MPCDF-launcher infrastructure friction (old default vLLM version, a
CLI-arguments field that strips all quote characters, large-checkpoint
startup cost).

Text-only LLMs sidestep the first two constraints at once: the pool of
architecturally-independent, strong text models (Llama, Mistral, DeepSeek,
GLM-text, ...) is far larger than the pool of vision-capable *and*
document-OCR-quality models, and a text-only model has no vision-encoder
graph-capture overhead, so MPCDF startup cost stops being a filter too.
The tradeoff, accepted explicitly: a text-only read depends on this
project's own OCR step being correct, so a disagreement can now mean "the
model got it wrong" *or* "the OCR was wrong" -- unlike two vision reads of
the same image. This is judged an acceptable, even favorable, trade:
unlike Pixtral's fluent fabrication (indistinguishable from a correct read
without checking the source image), an OCR-caused disagreement is easy to
diagnose by eye (open the image, check whether the OCR'd text matches) --
see "Arbitration UX" below.

## Goals

- `generate_dnb_toc_ground_truth.py` can pair one vision-model endpoint
  with one text-only-model endpoint, MPCDF-sourced or otherwise, and gate
  their outputs exactly like two vision models today.
- The existing all-vision path (`--endpoint`/`--config-file` used alone,
  or KISSKI auto-discovery) is completely unchanged.
- `gate_book`/`diff_toc_entries` (`evaluation/dnb_toc_matching.py`) require
  no changes -- they already only operate on two `list[TocEntry]`.

## Non-goals

- No chapter-segmentation integration. Names and module boundaries are
  kept generic where that's free (e.g. the `LLMClient` adapter already has
  zero corpus-specific knowledge), but nothing here wires into
  `src/chapter_segmentation/segmentation.py`'s own heuristic pipeline or
  `refresh_llm_cache.py`'s multi-corpus cache refresh.
- No two-text-only-endpoints mode. Requiring at least one side to read the
  actual page image preserves this design's core epistemic value; two
  text reads would silently reintroduce the OCR-dependent-on-both-sides
  design this project already moved away from once (the original
  regex-heuristic + text-LLM pairing superseded by
  `docs/superpowers/specs/2026-08-16-dnb-toc-uniform-ocr-design.md`).
- No change to `_LLM_TOC_EXTRACTION_PROMPT`/`llm_extract_toc_entries`
  (`src/chapter_segmentation/segmentation.py:542,685`) -- those stay
  exactly as used by the main heuristic pipeline's own LLM fallback. The
  new text-extraction prompt for this pipeline is a separate constant (see
  below), not a shared/modified version of that one.

## Design

### 1. CLI interface

`generate_dnb_toc_ground_truth.py` gains two new flags, parallel to the
existing `--endpoint`/`--config-file` (which stay vision-only, unchanged
default behavior):

- `--text-endpoint ALIAS` -- resolves `<ALIAS>_BASE_URL`/`_API_KEY`/`_MODEL`
  from the environment, exactly like `--endpoint`, via the existing
  `resolve_endpoint_from_env` (`evaluation/inference_endpoints.py`) --
  unchanged, no text-specific logic needed there.
- `--text-config-file [PATH]` -- same pasted-session-table format and
  `resolve_endpoints_from_config_file` machinery as `--config-file`,
  defaulting to `.mpcdf-sessions.txt` when the path is omitted, same as
  today.

**Valid combinations** (validated in `_resolve_vision_endpoints`'s
replacement, see below):

| Vision side | Text side | Result |
|---|---|---|
| none | none | KISSKI auto-discovery, both vision (today's default, unchanged) |
| `--endpoint` x2, or `--config-file` (2 blocks) | none | two vision endpoints (today's MPCDF path, unchanged) |
| `--endpoint` x1, or `--config-file` (1 block) | `--text-endpoint` x1, or `--text-config-file` (1 block) | **new**: one vision + one text |
| anything else (e.g. 1 vision only, 2 text, mismatched block counts) | -- | `ValueError`, same style as today's existing endpoint-count errors -- names exactly what's wrong |

Vision and text sides may use different sourcing mechanisms freely (e.g.
vision via `--endpoint`, text via `--text-config-file`) -- each resolves
independently.

`resolve_endpoints_from_config_file`'s current hard "exactly 2 blocks"
assertion loosens to "at least 1 block, caller validates the total
shape" -- the file itself may contain 1 or 2 pasted session tables; which
count is valid depends on whether it's supplied via `--config-file`
(needs 1 when paired with a text side, 2 when standalone) or
`--text-config-file` (always needs exactly 1). This shifts the "exactly
2" check from `resolve_endpoints_from_config_file` itself up into
`generate_dnb_toc_ground_truth.py`'s own combination-validation logic,
next to the equivalent `--endpoint` count checks.

### 2. `evaluation/inference_endpoints.py`

Relocate `_OpenAICompatibleLLMClient` here from `evaluation/refresh_llm_cache.py:114`
(made non-private, e.g. `OpenAICompatibleLLMClient`) -- it already has zero
KISSKI/MPCDF-specific knowledge per its own docstring ("callers construct
the client themselves... this class has no provider-specific knowledge at
all"), so this is a pure move, not a behavior change.
`refresh_llm_cache.py` imports it from its new location instead of
defining it. It implements `chapter_segmentation.llm.LLMClient`
(`.generate(prompt, *, max_tokens, temperature, is_valid=None) -> str`),
which is exactly what `text_extract_toc_entries` needs to call
`llm_extract_toc_entries`-shaped logic against a `ModelEndpoint`.

### 3. New module: `evaluation/dnb_toc_ocr.py`

Structurally parallel to `evaluation/dnb_toc_vision.py`, reusing its
caching functions directly (`cache_path`/`load_cached_llm_entries`/
`write_cached_llm_entries`/`versioned_cache_dir` are already
model-id-keyed and provider-agnostic -- no cache-schema version bump
needed for the entries themselves, but see the `"kind"` field addition
below).

**`ocr_pages_to_rows(pdf_path: Path, *, pdfalto_bin: str | None = None) -> list[str]`**
(new code -- the row-reconstruction step was only ever a one-off
investigation script in the 2026-08-16 uniform-ocr-design spec §1b, never
committed). `pdfalto_bin` is passed straight through to
`pdfalto_runner.resolve_pdfalto_binary` (existing,
`evaluation/scripts/pdfalto_runner.py:12`) -- `None` (the CLI's default
when `--pdfalto-bin` isn't given) resolves via `PDFALTO_BIN` then a bare
`pdfalto` on `PATH`, same convention as every other pdfalto call site in
this project (`pdfalto` is a sibling checkout, not on `PATH` by default --
see `CLAUDE.local.md`/`evaluation/CLAUDE.md`'s pdfalto notes):
1. Runs `ocrmypdf --force-ocr -l deu+eng` on `pdf_path` unconditionally
   (per the "always force fresh OCR" decision -- consistent behavior
   regardless of whether the PDF already has an embedded text layer; cheap
   here since dnb-toc-only PDFs are pre-filtered to 1-3 TOC pages).
2. Runs `pdfalto_runner.ensure_alto_xml` (existing,
   `evaluation/scripts/pdfalto_runner.py:27`) on the OCR'd output to get
   per-word `HPOS`/`VPOS`/`WIDTH` coordinates.
3. Parses the ALTO XML's `<String>` elements per page, clusters by `VPOS`
   (8px tolerance, per the original investigation's finding) into rows,
   sorts by `HPOS` within each row, joins into one text block per page.
4. Returns one string per page, in page order -- same per-page-list shape
   `render_pages_to_images` (`evaluation/dnb_toc_vision.py:170`) already
   returns for images, so the two extraction paths stay visually parallel
   in any calling code.

**`_TEXT_TOC_EXTRACTION_PROMPT`** -- a new prompt constant, NOT a reuse of
`_LLM_TOC_EXTRACTION_PROMPT` (`src/chapter_segmentation/segmentation.py:542`),
which predates the 2026-08-17 verbatim-per-line/`skip`-flag standard (it
says "skip acknowledgements, bibliography, index, and part-divider pages"
-- the old omission-based approach, and has no `skip` field in its JSON
schema at all). Reusing it here would make the text side systematically
disagree with the vision side's now-standard output for reasons having
nothing to do with either side's actual accuracy -- the same class of bug
the Pixtral numbering-prefix fix addressed
(`evaluation/dnb_toc_vision.py`, commit `9bea40a`). The new prompt is
`_VISION_TOC_EXTRACTION_PROMPT` (`evaluation/dnb_toc_vision.py:109`)
adapted for text input: same JSON schema, same verbatim-per-line/`skip`
semantics, same sub-point/two-line-title guidance, with the
image-reading-specific framing ("photographed/scanned page images...read
the images directly") replaced by text-reading framing that also warns the
input is machine-OCR'd and may contain scanning artifacts (misrecognized
characters, stray dot-leader fragments) to read past rather than transcribe
literally.

**`text_extract_toc_entries(pdf_path: Path, model: str, client: Any, *, pdfalto_bin: str = "pdftoppm") -> list[TocEntry]`**
-- structurally parallel to `vision_extract_toc_entries`
(`evaluation/dnb_toc_vision.py:189`): calls `ocr_pages_to_rows`, builds
the prompt with the joined per-page text blocks, wraps `client` in
`OpenAICompatibleLLMClient` (from `inference_endpoints.py`, see above) and
calls `.generate(...)`, parses via the same `parse_json_array`/
`_toc_items_to_entries` pipeline. Same `_VISION_MAX_TOKENS`/
`_VISION_MAX_TOKENS_RETRY`-style escalation on a parse failure (text
responses have no image tokens to inflate the prompt, so budget pressure
here is milder than vision, but the escalation costs nothing to keep for
consistency). Same "raises on failure rather than swallowing" contract as
`vision_extract_toc_entries`, for the same reason (the caller's retry
wrapper does real work).

### 4. Cache schema: a `"kind"` field

`write_cached_llm_entries`/`load_cached_llm_entries`
(`evaluation/dnb_toc_vision.py:66,85`) gain an optional `"kind":
"vision"|"text"` field in the cached JSON, written by both extraction
paths' callers. Absent on any pre-existing cache file, which
`load_cached_llm_entries` treats as `"vision"` (every cache file to date
came from `vision_extract_toc_entries`) -- fully backward compatible, no
migration needed. This is the data source for the arbitration-UX label
below; without it, `arbitrate_dnb_toc.py` would have no reliable way to
know a given cached model id's entries came from an OCR'd-text read
rather than a direct image read.

### 5. `generate_dnb_toc_ground_truth.py`

`_resolve_vision_endpoints` (currently: exactly 2 endpoint aliases in,
exactly 2 `ModelEndpoint`s out) is replaced by a function returning a
small `(vision_endpoint: ModelEndpoint, second_endpoint: ModelEndpoint,
second_kind: Literal["vision", "text"])` result, implementing the
combination-validation table in §1. `_run_book`'s two extraction calls
become: always `vision_extract_toc_entries` for the vision endpoint;
`vision_extract_toc_entries` or `text_extract_toc_entries` for the second
endpoint, dispatched on `second_kind`.

### 6. Arbitration UX

`format_book_report` (`evaluation/scripts/arbitrate_dnb_toc.py:88`) labels
each side by its cached `"kind"` (§4) when formatting a disagreement dump
-- e.g. `"vision: Qwen/Qwen3-Omni-30B-A3B-Instruct"` vs. `"text (OCR'd):
meta-llama/Llama-3.3-70B-Instruct"`. This directly serves the "fails
legibly" property motivating this whole design: a human arbitrating a
mixed-source disagreement knows immediately to also check the OCR'd text
itself (not just re-read the page image cold) as a likely root cause, the
same way a `bulk_gate` vs. `claude_arbitration` `"source"` field already
tells a reader how an entry was decided.

### 7. `gate_book`/`diff_toc_entries`: unchanged

No code changes to `evaluation/dnb_toc_matching.py`. They already operate
on two `list[TocEntry]` with no knowledge of where either list came from
(confirmed by the 2026-08-16 uniform-ocr-design spec §3.1, which made the
same point when the gate's inputs first became two vision reads instead of
a heuristic+text-LLM pair). Same 0.90 threshold, same any-mismatched-pair
rejection rule, for both the vision+vision and vision+text cases.

### Out of scope

- `refresh_llm_cache.py` gets no `--text-endpoint`/`--text-config-file`
  flags in this pass -- it already shares `--endpoint`/`--config-file` for
  its own, separate multi-corpus cache-refresh purpose, unrelated to this
  pipeline's two-source gate.
- No chapter-segmentation wiring, per Non-goals.

## Error handling

- OCR failure (`ocrmypdf`/`pdfalto` crash, e.g. a corrupted PDF) --
  `ocr_pages_to_rows` raises, propagating up through
  `text_extract_toc_entries` exactly like any other extraction failure;
  the caller's existing retry/skip-and-report logic in
  `_run_book`/`_generate` needs no changes to handle it, since it already
  treats any raised exception from either extraction call uniformly.
- A book with genuinely poor scan quality -- the OCR-derived read
  disagrees badly with the vision read, `gate_book` rejects it below
  threshold (or on the any-mismatch rule) exactly as today, and
  arbitration surfaces it with the new source label so a human's first
  instinct is to suspect OCR quality, not silently assume both models
  simply disagree. No special-cased "OCR looked bad, auto-reject" path --
  consistent with the "no gate change" decision, and keeping the human
  arbitration step as the actual quality backstop.

## Testing

Follows existing conventions in `tests/test_dnb_toc_vision.py` and
`tests/test_generate_dnb_toc_ground_truth.py`:

- `ocr_pages_to_rows`: unit tests against a small fixture PDF (or a
  fixture ALTO XML, to avoid a real `ocrmypdf` dependency in fast unit
  tests) covering correct row ordering (multi-column input, the same
  `9783518585306.pdf` dot-leader case documented in the 2026-08-16 spec).
- `_TEXT_TOC_EXTRACTION_PROMPT`/`text_extract_toc_entries`: mirrors
  existing `vision_extract_toc_entries` tests (fake `LLMClient` returning
  canned JSON, parse-retry-escalation behavior, raises-on-failure
  contract).
- CLI combination validation: one test per row of the §1 table, including
  every invalid combination, asserting the exact `ValueError` message
  names what's wrong (matching this project's existing convention for
  endpoint-resolution errors).
- `format_book_report`'s new source labels: a test asserting a `"kind":
  "text"` cache entry renders differently from a `"kind": "vision"` (or
  absent-key, defaulting to vision) one.
