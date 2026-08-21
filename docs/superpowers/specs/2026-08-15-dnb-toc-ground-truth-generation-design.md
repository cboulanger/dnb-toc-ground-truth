# Structured ground truth generation for dnb-toc-only

Status: approved for planning
Date: 2026-08-15

Detailed follow-up to §2 of
`docs/superpowers/specs/2026-08-15-dnb-toc-ground-truth-and-consumers-design.md`
("Structured ground truth for dnb-toc-only, and its three consumers"). That
document scoped this piece as "build first, everything else depends on it";
this spec designs it concretely enough to plan and implement.

## 1. Goal

Produce `evaluation/corpus/dnb-toc-only/<id>.expected.json` for as much of
the (grown, ~1000-book) `dnb-toc-only` corpus as can be trusted without a
full human pass over every book, plus a smaller, fully hand-verified,
held-out subset — per the two-tier split the parent spec already settled
(§2.2-§2.4 there). This spec covers the mechanics: exact extractors, the
agreement/gating algorithm, the eval-tier sampling method, the manual
transcription workflow, and the precision spot-check.

## 2. Schema (unchanged from the parent spec, restated for completeness)

```json
{
  "entries": [
    {"title": "Einleitung", "authors": [], "printed_page_number": "9"},
    {"title": "Zur Soziologie des Rechts", "authors": ["Max Mustermann"], "printed_page_number": "17"}
  ],
  "verified": false
}
```

`title`/`authors`/`printed_page_number` mirror `TocEntry`'s own field shapes
(`src/chapter_segmentation/segmentation.py:209`) exactly — both extractors
below already produce `TocEntry` objects, so serializing one to this schema
is a direct field copy, no translation layer.

## 3. The two extractors — reused as-is, not reimplemented

Both are existing functions in `src/chapter_segmentation/segmentation.py`
and both already return `list[TocEntry]`, confirmed identical in shape:

- **Heuristic**: `find_toc_candidates(pages: list[str], ...) -> list[TocEntry]`
  (line 340). Call it directly on the DNB scan's own page texts — every
  page of a `dnb-toc-only` PDF *is* TOC content by construction (per the
  2026-08-14 acquisition spec), so no front/back-fraction scanning
  heuristic is needed beyond what the function already does by default.
  Page texts come from `extract_page_texts_for_analysis(content: bytes) ->
  tuple[list[str], bool]` (line 286), the same extraction path production
  and every other corpus already uses.
- **LLM**: `llm_extract_toc_entries(pages: list[str], llm_client: LLMClient)
  -> list[TocEntry]` (line 586, `async`). Takes an injected `LLMClient`
  (Protocol, `src/chapter_segmentation/llm.py:13`) with no hardcoded
  provider — wire it to KISSKI the same way `evaluation/refresh_llm_cache.py`
  already does (`_OpenAICompatibleLLMClient`, `KISSKI_API_KEY` env var,
  lines 90-109/257 there). This function makes a live call every
  invocation and has no caching of its own (confirmed) — the new script
  owns caching.

**New script**: `evaluation/scripts/generate_dnb_toc_ground_truth.py`.
Structure mirrors `refresh_llm_cache.py`'s existing pattern (already
validated at ~1000-book-adjacent scale): `asyncio.Semaphore`-bounded
concurrency (default 4, same as the existing `--concurrency` convention),
`_call_with_retry`-style exponential backoff (3 attempts, 1s/2s/4s), one
LLM call per book (not per-model — the concurrency/cost profile here is far
smaller than `refresh_llm_cache.py`'s 5-models-per-book fan-out). LLM output
cached at `evaluation/corpus/dnb-toc-only/llm-cache/<id>.json` — gitignored
(transient intermediate toward `.expected.json`, nothing downstream reads it
directly, unlike other corpora's committed `llm-cache/`, which
`generate_report.py` reads for its own report).

**Skip condition**: run `pages_need_ocr(pages)` (line 268) before either
extractor. DNB's own digitization already embeds an OCR text layer, so this
should rarely trip — for the (hopefully small) fraction where it does, skip
the book entirely from the bulk tier rather than routing through the
Kreuzberg OCR sidecar (avoids pulling that infra dependency in for what
should be a minority case). Report the skip rate in the results write-up;
revisit only if it turns out non-trivial.

## 4. Agreement algorithm and the whole-book gate

### 4.1 Matching

New function, `_align_toc_entries(a: list[TocEntry], b: list[TocEntry]) ->
list[tuple[int, int]]`, adapted from two existing pieces rather than written
from scratch:

- The matching *rule* from `evaluation/nuextract_baseline.py:83`'s
  `match_toc_entries` — page-number-exact-match first (disqualifies on
  mismatch regardless of title), then `rapidfuzz.fuzz.token_sort_ratio`
  on lowercased titles, threshold `_ALIGN_SCORE_THRESHOLD = 70.0`
  (`src/chapter_segmentation/evidence/fusion.py:13`).
- The *greedy, order-preserving scan* from `fusion._align`
  (`fusion.py:16-36`) — for each item in `a`, search `b` only from
  `last_j + 1` onward, keep the highest-scoring match ≥ threshold, advance
  `last_j`. Same "TOC order is book order" monotonicity assumption both
  existing functions already rely on.

Unlike `match_toc_entries` (which returns only an integer count),
`_align_toc_entries` returns the matched **index pairs** — needed for the
union step below, not just a score. This is the one new piece of matching
logic this spec introduces; everything else is direct reuse.

### 4.2 Whole-book threshold gate

```
tp = len(_align_toc_entries(heuristic_entries, llm_entries))
agreement_rate = tp / max(len(heuristic_entries), len(llm_entries))
```

- **`agreement_rate >= 0.90` → book passes.** Final `entries` = the matched
  pairs (preferring the heuristic `TocEntry`'s field values when both sides
  match, since its title/author split comes from structured regex capture
  rather than LLM reformatting) **unioned with every singleton entry either
  extractor found alone**, inserted in printed-page-number order. This is
  deliberate, not an oversight: once a book clears the trust bar, a line
  only one method caught is far likelier a real entry the other missed
  (OCR noise, an unusual title format) than a hallucination — trimming it
  out would silently understate the page's real content, which is exactly
  the "incomplete training target" failure mode this design is meant to
  avoid for §3 (NuExtract fine-tuning) downstream.
- **`agreement_rate < 0.90` → book dropped from the bulk tier entirely.**
  No `.expected.json` is written; the PDF and manifest entry are untouched
  (available for the corpus's other uses, e.g. as a free layout-classifier
  positive-page example, which needs none of this). Not queued for manual
  review as part of this spec's automation — same "leave it out rather
  than force a fix" stance `evaluation/CLAUDE.md`'s redaction section
  already takes for a structurally similar problem. A human can pick up
  the dropped set later the same way any `pending/` book is picked up; this
  spec's `--report` output should print the dropped-book count and IDs so
  that queue is visible, not lost.

### 4.3 Output

For every book that passes and isn't in the eval-tier exclusion list
(§5): write `<id>.expected.json` with `"verified": false`.

## 5. Eval tier: stratified selection

New mode of the same script, or a small sibling
(`evaluation/scripts/select_dnb_toc_eval_sample.py` — final split decided
during planning, not load-bearing here): read each candidate book's
`.lobid-cache/<id>.lobid.json` for `publication[0].startDate` (bucket into
decades) and the manifest's `language` field, draw a stratified sample of
~50-100 IDs proportional to the decade/language spread actually present in
the corpus (confirmed fields exist on disk today — see research notes
below). Write the selected IDs to a new, committed
`evaluation/corpus/dnb-toc-only/eval_tier_ids.json` (flat list of manifest
keys). The bulk-tier script (§3-4) always excludes IDs in this file — an
eval-tier book never gets bulk-tier (`verified: false`) treatment, since it
is meant to be transcribed by hand instead (§6) and must never be drafted by
either extractor it will later help evaluate.

## 6. Eval tier: manual transcription workflow

Documented as a new subsection of `evaluation/README.md` (permanent
"what/how" reference, alongside the existing `dnb-toc-only` corpus
description — not `RESULTS.md`, since this describes a repeatable procedure
rather than a dated snapshot):

1. Open the book's `<id>.pdf` (1-3 pages, the TOC scan itself — no
   chapter-locate search needed, unlike the full-book workflow
   `evaluation/CLAUDE.md` documents, since the target page *is* the whole
   PDF).
2. View it directly (`Read` tool, `pages` param).
3. Transcribe every entry the page actually prints (§2's schema) —
   including lines a full-book `.expected.json` would mark `skip: true`
   (bibliography, index headers, part dividers): this file measures
   extraction fidelity against what's printed, not "which of these are
   real chapters," so nothing gets filtered out here the way it is for
   `open-access`/`copyrighted-scans`.
4. Save as `<id>.expected.json` with `"verified": true`.

Meaningfully cheaper per book than the existing corpora's ground-truth
workflow — one page-image read replaces a multi-page content search through
a separate full book.

## 7. Spot-check: measuring the bulk tier's real precision

`--spot-check N` mode on the same script: sample N books from the *passing*
bulk tier (excluding the eval tier, which already has independent
verification), render each one's scan pages, and walk through an
Accept/Reject prompt per book against the rendered image (terminal-driven —
reuses the same visual-read pattern as §6 rather than building new UI; the
review-app extension mentioned in the parent spec's §8 stays deferred).
Reports a measured precision figure for the `agreement_rate >= 0.90` gate,
directly satisfying the parent spec's §7 decision criterion ("a spot-check
of the bulk tier's auto-accepted entries against real scan images reports a
measured precision, not assumed"). Record the result in `RESULTS.md`, same
convention as every other one-off measurement in this project.

## 8. File layout

```
evaluation/corpus/dnb-toc-only/
  manifest.json              # unchanged
  eval_tier_ids.json          # NEW, committed -- the held-out sample (§5)
  <id>.expected.json          # NEW, committed -- {"entries": [...], "verified": bool}
  <id>.pdf                    # unchanged, gitignored
  .lobid-cache/<id>.lobid.json  # unchanged
  llm-cache/<id>.json          # NEW, gitignored -- transient LLM-extraction cache (§3)
```

`<id>.expected.json` is committed for both tiers, same convention as the
other corpora's ground truth — it holds only parsed titles/authors/page
numbers (already legible directly from the CC0-licensed scan), no
copyrighted running text.

## 9. Testing

`_align_toc_entries` (§4.1) and the whole-book gate (§4.2) are pure
functions over `TocEntry` lists — straightforward to unit-test with
synthetic data (perfect agreement, partial disagreement above/below the
0.90 threshold, empty lists, out-of-order input) without needing real PDFs
or network access. New test module,
`tests/scripts/test_generate_dnb_toc_ground_truth.py` (or colocated with
the script per this project's existing test-layout convention — confirm
during planning). Per this project's standing convention, write these
tests before the implementation (test-driven-development).

## 10. Decision criteria

- The bulk-tier pass produces `.expected.json` for a clear majority of the
  (grown) corpus, with the dropped-book count/IDs printed and reviewable.
- `--spot-check` reports a measured precision number for the
  `agreement_rate >= 0.90` gate against real scan images, not an assumed
  one.
- `eval_tier_ids.json` holds ~50-100 IDs stratified across the corpus's
  actual decade/language spread, each with a hand-transcribed,
  `verified: true` `.expected.json`, none overlapping the bulk tier.
- `pages_need_ocr` skip rate is reported; if non-trivial (no numeric bar
  set here — a judgment call once the real rate is known), that becomes a
  follow-up rather than blocking this spec.

## 11. Out of scope

- Routing `pages_need_ocr`-flagged books through the Kreuzberg OCR sidecar
  (§3's stated limitation — revisit only if the skip rate is large).
- A manual-review queue/UI for books that fail the whole-book gate (§4.2) —
  they're dropped and reported, not queued, in this pass.
- Any of §3-§5 from the parent spec (NuExtract fine-tuning, the heuristic
  line-parsing harness, the layout-classifier pilot check) — each gets its
  own follow-up spec once this one's output exists.
- The review-app UI extension mentioned in the parent spec's §8 — the
  terminal-driven spot-check (§7) and manual transcription (§6) workflows
  here don't need it.

## Research notes (facts this design relies on, confirmed by direct
    inspection 2026-08-15)

- `find_toc_candidates(pages: list[str], max_front_fraction: float = 0.15,
  max_back_fraction: float = 0.05) -> list[TocEntry]` —
  `src/chapter_segmentation/segmentation.py:340`.
- `llm_extract_toc_entries(pages: list[str], llm_client: LLMClient) ->
  list[TocEntry]` — `segmentation.py:586`; no caching, no hardcoded
  provider.
- `match_toc_entries(predicted: list[dict], expected: list[dict]) -> int` —
  `evaluation/nuextract_baseline.py:83`; page-number-first then
  `token_sort_ratio` title match, greedy/order-preserving, returns a count
  only (this spec's `_align_toc_entries` is a pair-returning variant).
- `_align(list_a, list_b) -> list[tuple[int, int]]` —
  `src/chapter_segmentation/evidence/fusion.py:16-36`; the pair-returning
  greedy-scan pattern `_align_toc_entries` follows.
- `_ALIGN_SCORE_THRESHOLD = 70.0` — `fusion.py:13`.
- `refresh_llm_cache.py`'s concurrency (`asyncio.Semaphore`, default 4) and
  retry (`_call_with_retry`, 3 attempts, exponential backoff from 1s) —
  lines 181-201 and 160-178 respectively.
- `evaluation/corpus/dnb-toc-only/manifest.json` book-entry fields:
  `filename`, `title`, `language`, `doi`, `toc_download_url`, `license`,
  `license_source`, `lobid_url`. `.lobid-cache/<id>.lobid.json` files exist
  on disk today and carry `publication` (a list with `startDate`,
  `publishedBy`, etc.) and `language`, confirming §5's stratification is
  feasible without any new network fetch.
- `list_corpora(include_toc_only: bool = False)` —
  `evaluation/harness.py:41` — already implemented, not a stale forward
  reference from the 2026-08-14 spec.
