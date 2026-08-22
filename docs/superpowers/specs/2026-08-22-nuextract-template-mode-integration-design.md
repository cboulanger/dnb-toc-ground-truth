# Integrating NuExtract-family template-mode extraction

Status: proposed
Date: 2026-08-22

## Context

Every vision/text model this project calls today (`vision.py`,
`ocr.py`) is driven the same way: a long free-text prompt describing the
desired JSON output shape and the extraction rules, sent as an ordinary
chat message. `numind/NuExtract3` (a 4B vision-language model
specifically trained for structured extraction, now confirmed launchable
on MPCDF's Viper-GPU service -- see
`docs/llm-inference-providers.md`) does not follow that pattern well: ad
hoc testing against the live endpoint (`https://llm.mpcdf.mpg.de/8eofdywb22uz9iyr/v1`,
`numind/NuExtract3`, single-GPU, `rocm7.13.0_gfx94X-dcgpu_..._vllm_0.19.1`
image) found that a free-text extraction prompt sent as ordinary chat
content is effectively ignored -- the model instead falls back to its own
trained default behavior (document-to-markdown conversion, or a
degenerate single-entry response), regardless of what the prompt asks
for.

NuExtract3 performs well only through its own bespoke API convention: an
explicit JSON **template** (a schema-shaped object whose leaf values are
type names, e.g. `"verbatim-string"`, `"boolean"`) and an optional prose
**instructions** string, both passed via
`extra_body={"chat_template_kwargs": {...}}` -- not the OpenAI
`response_format`/`json_schema` mechanism, which is a generic,
model-agnostic constrained-decoding feature unrelated to NuExtract's own
trained template-following behavior. There is no converged standard
across extraction-tuned models for this; NuExtract's
`chat_template_kwargs` shape is specific to its own Jinja chat template.

Empirical validation (single sample book, `0121475018.pdf`, page 1 -- 24
TOC entries covering dividers, front/back matter, multi-author lines,
roman-numeral front matter):

| Mode | Result |
|---|---|
| free-text prompt, `content` mode, or no `chat_template_kwargs` at all | Ignores the prompt; returns a markdown table or a single degenerate entry |
| `template` mode, explicit JSON schema, **no** `instructions` | All 24 entries correct, but no semantic understanding of `skip` (needs the concept explained) |
| `template` mode + `instructions` carrying this project's extraction rules (subtitle-merging, sub-point handling, `skip` semantics, verbatim roman numerals), **image** input | All 24 entries correct, `skip` correctly applied to dividers/front/back matter, roman numerals preserved verbatim (`"v"`, `"vii"`) |
| Same template + instructions, **OCR'd text** input (`ocr_pages_to_rows` output) instead of an image | Meaningfully worse: dropped `"Part I."`/`"Part II."` divider labels (violates an explicit instruction), one genuine chapter's page number came back `null`, one title absorbed a stray OCR-noise fragment, author names duplicated into the title instead of being cleanly split out |

Despite the weaker text-mode result, this design still wires up both
paths: a finetuned, vision-less NuExtract2-family checkpoint is a
concrete near-term follow-up, and NuExtract2's own API is known not to
accept a separate `instructions` field at all, so the `instructions`
step must already be optional by construction.

There is no reliable way to detect "this endpoint speaks the NuExtract
template API" from the model id alone -- a finetuned checkpoint can be
named anything -- so this is declared explicitly per endpoint, not
sniffed from `framework_args` or the model id.

## Goals

- `--use-vision`/`--use-text` can resolve to a NuExtract-family
  (`numind/NuExtract3` today, a finetuned NuExtract2 later) endpoint and
  get correct `TocEntry` extraction from it, via template mode.
- Every existing model/endpoint (free-text-prompt path) is completely
  unchanged -- this is purely additive.
- `numind/NuExtract3` specifically works with **zero `.endpoints` edits**
  for the entry already in place (see "Known-model convenience default"
  below) -- every other NuExtract-family endpoint (the future finetune)
  must declare itself explicitly.

## Non-goals

- No attempt to fix or work around NuExtract3's weaker text-mode
  behavior in this pass -- documented as a known limitation (see
  Context table above), not engineered around. The finetuned NuExtract2
  checkpoint's own quality is untested and out of scope until it exists.
- No change to `parse_json_array`/`_toc_items_to_entries`
  (`toc_entry.py`) -- NuExtract's `{"entries": [...]}` response shape
  already parses correctly through the existing bracket-extraction logic
  (`parse_json_array` takes the substring from the first `[` to the last
  `]`, which discards the outer `{"entries": ...}` object wrapper for
  free), confirmed against real responses during ad hoc testing.
- No change to the OpenAI `response_format`/`json_schema` mechanism or
  any other model's extraction path.
- No fix for the trailing-period artifact occasionally seen in
  NuExtract-extracted author names (e.g. `"P. N. JOHNSON-LAIRD."`) --
  minor, noted for a future pass.

## Design

### 1. Endpoint declaration (`.endpoints`)

Two new optional fields on an endpoint entry, read by both the
JSON-array and plain-text pasted-session-table parsers in `inference.py`:

- `extraction_api` (string, default `""`) -- `"nuextract"` selects the
  template-mode extraction path for this endpoint; empty (the default,
  and every existing entry's implicit value) means today's free-text
  path, unchanged.
- `extraction_instructions` (string `"true"`/`"false"`, default
  `"true"`) -- whether to send the `instructions` field alongside the
  template. Only meaningful when `extraction_api == "nuextract"`.

**Known-model convenience default:** while parsing an entry (either
format), if `extraction_api` was not explicitly set in the file AND the
resolved model id is exactly `"numind/NuExtract3"`, `_EndpointEntry`
defaults `extraction_api` to `"nuextract"` and `extraction_instructions`
to `True` -- so the `.endpoints` entry already running today (added
before this design existed) works without manual editing. An explicit
`extraction_api` value in the file always wins over this default,
including an explicit empty-string override if someone ever needs to
force the old free-text path for a `numind/NuExtract3` endpoint. This
convenience default applies to `"numind/NuExtract3"` only, by exact
string match -- it is a one-off carve-out for the endpoint that already
exists, not a general pattern-matching mechanism (see Context for why
name-pattern matching was rejected as the general mechanism).

`_EndpointEntry` gains `extraction_api: str = ""` and
`extraction_instructions: bool = True`. `ModelEndpoint` gains the same
two fields, threaded through unchanged by `resolve_model_endpoints`.

### 2. New module: `src/dnb_toc_ground_truth/nuextract.py`

Structurally parallel to `vision.py`/`ocr.py`. Two public async
functions:

- `nuextract_vision_extract_toc_entries(pdf_path: Path, model: str, client: Any, *, use_instructions: bool = True, pdftoppm_bin: str = "pdftoppm") -> list[TocEntry]`
- `nuextract_text_extract_toc_entries(pdf_path: Path, model: str, client: Any, *, use_instructions: bool = True, pdfalto_bin: str | None = None) -> list[TocEntry]`

Both:
- Reuse `render_pages_to_images` (`vision.py`) / `ocr_pages_to_rows`
  (`ocr.py`) for input construction -- no duplicated
  rendering/OCR logic, and the same `_MAX_VISION_PAGES`/`_MAX_TEXT_PAGES`-
  style page-count guard as their free-text counterparts.
- Build the same template for both: `{"entries": [{"title":
  "verbatim-string", "authors": ["string"], "printed_page_number":
  "verbatim-string", "skip": "boolean"}]}`.
- Build the same prose `_INSTRUCTIONS` constant (module-level, shared by
  both functions) -- ported from `_VISION_TOC_EXTRACTION_PROMPT`'s rules
  (subtitle-merging, sub-point handling, `skip` semantics, verbatim
  roman numerals, leading-label-part-of-title), stripped of the
  "return ONLY a JSON array" boilerplate since template mode already
  governs output shape.
- Send via `extra_body={"chat_template_kwargs": {"template":
  json.dumps(_TEMPLATE), "enable_thinking": False, **({"instructions":
  _INSTRUCTIONS} if use_instructions else {})}}`.
- Vision variant's message content: image(s) plus a short fixed user
  text (`"Extract every table-of-contents entry from this page."`),
  matching the pattern confirmed working in ad hoc testing. Text
  variant's message content: the joined per-page OCR'd text block, as a
  plain string (matching `text_extract_toc_entries`'s existing
  `pages_block` construction).
- Parse the response via the existing `parse_json_array` +
  `_toc_items_to_entries` pipeline -- no new parsing code.
- Same max-tokens escalation (`_VISION_MAX_TOKENS`/`_VISION_MAX_TOKENS_RETRY`-
  shaped retry on an unparseable/truncated response) and
  raises-on-failure contract (never swallows an exception) as
  `vision_extract_toc_entries`/`text_extract_toc_entries`.

### 3. Dispatch (`generate_ground_truth.py`)

`_run_book`'s per-endpoint call gains an `extraction_api` branch before
the existing `kind` branch:

```python
async def _call(ep=endpoint):
    async with semaphore:
        if ep.extraction_api == "nuextract":
            fn = nuextract_text_extract_toc_entries if ep.kind == "text" else nuextract_vision_extract_toc_entries
            return await fn(pdf_path, ep.model_id, ep.client, use_instructions=ep.extraction_instructions)
        if ep.kind == "text":
            return await text_extract_toc_entries(pdf_path, ep.model_id, ep.client)
        return await vision_extract_toc_entries(pdf_path, ep.model_id, ep.client)
```

### 4. Caching: unchanged

`load_cached_llm_entries`/`write_cached_llm_entries`/`load_cached_kind`
(`vision.py`) are already keyed by `(book key, model_id)` plus a
`"vision"`/`"text"` `kind` field, agnostic to which function produced
the entries. A NuExtract-produced `TocEntry` list is cached and read
back exactly like any other model's, no schema change needed.

## Error handling

- Same as every existing extraction path: a malformed/truncated response
  triggers the existing max-tokens escalation retry; a still-unparseable
  response after that raises, propagating through `_call_with_retry` and
  `_run_book`'s existing catch-and-report-per-book handling -- no new
  error paths.
- An `.endpoints` entry with `extraction_api` set to anything other than
  `"nuextract"` or `""` is treated the same as `""` (today's free-text
  path) -- not validated/rejected, consistent with every other
  currently-ignored extra field in the endpoints file (`framework`,
  `gpus`, `job_id`, ...).

## Testing

Follows existing conventions (`tests/test_vision.py`,
`tests/test_ocr.py`, `tests/test_inference.py`,
`tests/test_generate_ground_truth.py`):

- `inference.py`: new tests covering `extraction_api`/
  `extraction_instructions` parsing in both the JSON-array and
  plain-text formats, explicit-value-wins-over-default, and the
  `numind/NuExtract3` convenience default (present when unset, absent
  for every other model id).
- `nuextract.py`: mirrors `test_vision.py`/`test_ocr.py`'s existing
  shape -- a fake client returning canned `{"entries": [...]}` JSON,
  asserting correct `TocEntry` construction, the `use_instructions=False`
  case omits the `instructions` key from `extra_body`, parse-retry
  escalation, and raises-on-failure.
- `generate_ground_truth.py`: one test asserting `_run_book` dispatches
  to the NuExtract functions when `extraction_api == "nuextract"` for
  both `kind` values, and to the existing functions otherwise.
