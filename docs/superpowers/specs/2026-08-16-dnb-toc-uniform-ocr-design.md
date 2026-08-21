# Vision-LLM TOC extraction for dnb-toc-only

Status: proposed
Date: 2026-08-16 (revised same day — see §2 for the pivot)

Follow-up to `docs/superpowers/specs/2026-08-15-dnb-toc-ground-truth-generation-design.md`
(the GT-generation script this modifies). Triggered by two real findings
from smoke-testing that script against the live corpus:

1. `needs_ocr` skips ~30% of sampled books outright — their embedded text
   layer is absent or degenerate, so neither extractor ever runs.
2. Even where a text layer exists, quality is inconsistent across the
   corpus (different scanning/digitization sources), so heuristic and LLM
   extraction quality varies for reasons unrelated to either extractor's
   logic.

The original brief was to fix this with a uniform, project-controlled OCR
pass. That investigation (§1) found a working fix for the *layout*
problem (column/block misassociation) but ran into a harder, only
partially fixable *dot-leader garbage* problem. A follow-up experiment
(§2) tested KISSKI's vision-capable models reading page images directly,
bypassing OCR text entirely — results were strong enough (§2.1) that the
design now replaces the OCR pipeline rather than building it (§3).

## 1. OCR-pipeline investigation (superseded by §2-3, kept for the record)

Four things were tested directly against the corpus before proposing the
original OCR design, using book `9783518585306.pdf` (the known "shredded",
one-word-per-line worst case) as the running example.

**a) Plain `ocrmypdf --force-ocr -l deu+eng` (tesseract's own reading
order).** Fixes the one-word-per-line fragmentation, but introduces a worse
problem on this book's two-column-ish layout: tesseract's block
segmentation groups *all* title lines into one block, then *all* page
numbers into a separate block that follows, e.g. the reconstructed page
text ends with a disconnected `"...33\n81\n100\n117\n155\n174\n203"` list.
`find_toc_candidates`'s regex requires title and number on the same
physical line, so this produces **zero** heuristic entries. Confirms the
original concern: naive OCR does not solve column-based TOCs.

**b) `pdfalto` word-position reconstruction.** `pdfalto` (sibling checkout
at `/Users/cboulanger/Code/pdfalto/pdfalto`, already used elsewhere in this
project — `evaluation/scripts/pdfalto_runner.py`) runs cleanly against an
`ocrmypdf`-produced PDF and emits per-word `HPOS`/`VPOS` coordinates.
Clustering `<String>` tokens by `VPOS` (8px tolerance) into rows and sorting
by `HPOS` within each row reconstructs correct reading order regardless of
tesseract's own block segmentation:

```
240 1. Wozu noch Philosophie? .........000e ee eeee 33
260 . Die Philosophie als Platzhalter und Interpret
279 3. Was Theorien leisten können - und was nicht.
300 Ein Interview ss m onen een ee eee eee ees 81
```

Title and page number now land on the same reconstructed line every time —
this directly fixes problem (a). Confirmed on the full page: 20/20 rows
reconstructed correctly, in printed order.

**c) `tessdata_best` vs. the installed `tessdata_fast` German/English
models.** Hypothesis: a stronger OCR model would produce cleaner dot-leader
text (`.........000e ee eeee` above is tesseract mis-reading a run of dots
as garbage words). Downloaded `tessdata_best/deu.traineddata` (8.6MB vs.
the installed 1.5MB fast model) and re-ran. Result: a **mixed, marginal**
improvement — some lines got cleanly recognized dot runs
(`Vorwort zur Studienausgabe ...................`), but others stayed
garbled differently (`Einleitung .2..0..000000000000 0100`), and the best
model introduced a *new* digit misread it didn't have before
(`155` → `I55`, capital-I for `1`). Not a reliable fix on its own, and it
adds a real-world dependency (downloading ~9-15MB per language from
`tessdata_best` at setup time — not bundled with the `tesseract-lang`
Homebrew formula) for an inconsistent payoff. **Rejected** as the primary
fix for dot-leader garbage; not adopted.

**d) Whether dot-leader garbage actually breaks anything, and how to fix
it independently of OCR quality.** `_TOC_LINE_RE`
(`src/chapter_segmentation/segmentation.py:76`) is permissive about what it
swallows into the title group (`.{3,120}?` matches any character), so
garbled dot-leader text does **not** prevent page-number extraction — it
just pollutes the extracted title string:
`"Ein Interview ss m onen een ee eee eee ees"` instead of `"Ein Interview"`.
That pollution *does* break the agreement gate: `align_toc_entries`
(`evaluation/dnb_toc_matching.py`) scores title similarity with
`fuzz.token_sort_ratio`, which drops to 38.9-47.3 on these polluted/clean
pairs — well under the 70.0 threshold — because token-sort compares the
*entire* token multiset and a handful of garbage tokens dominates a short
real title. Testing `fuzz.partial_ratio` (best-matching contiguous
substring) on the same pairs scores a clean **100.0** on all of them, since
the garbage is a trailing addition rather than an interleaved corruption of
the real text. A negative-control check against four genuinely different
title pairs confirmed `partial_ratio` doesn't inflate false positives
either (max 48.8, still well under threshold) — expected, since alignment
is already gated on an *exact* page-number match before title similarity
is even scored, so `partial_ratio`'s looser substring tolerance only needs
to discriminate between candidates that already share a page number.

Also checked whether ALTO's per-token geometry could drive a targeted
"strip the dot-leader run" cleanup instead: found overlapping/nonsensical
`HPOS`/`WIDTH` boxes on the garbled tokens (e.g. a 4-character token
`"onen"` reported 201px wide — 7x the per-character width of neighboring
real words), meaning tesseract's own bounding boxes for hallucinated
dot-leader "words" aren't reliable enough to key a geometric cleanup off
of. Confirms text-level tolerance (partial_ratio) is the right layer to fix
this at, not more geometry.

**e) Runtime cost.** `ocrmypdf --force-ocr -l deu+eng` + `pdfalto` on the
2-page test book: 8.4s wall-clock — a workable cost, but see §2.2 for how
it compares to the vision alternative.

## 2. Pivot: vision-model extraction

The OCR/ALTO pipeline in §1 fixes the layout problem but only partially
fixes dot-leader garbage, and still depends on `ocrmypdf` producing a
usable text layer at all. Since `find_toc_candidates`/`llm_extract_toc_entries`
only need the TOC's *content*, not literally its text layer, the next
question was whether a vision-capable LLM reading the page image directly
could skip text extraction altogether — no OCR, no ALTO, no row
reconstruction, no dot-leader garbling because there's no text layer to
garble.

### 2.1 Experiment

Rendered page images directly from the original PDFs with
`pdftoppm -r 200 -png` (no OCR step at all) and sent them to KISSKI's
`qwen3-omni-30b-a3b-instruct` (`demand=0` at test time) via a single
OpenAI-compatible chat completion with `image_url` content blocks, using a
prompt adapted from `_LLM_TOC_EXTRACTION_PROMPT`
(`src/chapter_segmentation/segmentation.py:488`) for image input. Two
books, chosen for different failure modes:

| Book | Layout | Result | Time |
| --- | --- | --- | --- |
| `9783518585306.pdf` (the §1 "shredded"/dot-leader book, 2 pages) | numbered chapters, dotted leaders | **18/18 entries**, every title clean (verified against the rendered image by eye), every page number correct | 6.9s |
| `3110139642.pdf` (a `needs_ocr` book — no usable embedded text layer at all, 3 pages) | edited-volume TOC, author name in caps on its own line above each title, right-aligned page numbers, *no* dot leaders | **18/18 entries**, titles and page numbers correct, **and** every all-caps author name correctly parsed into `authors` | 10.5s |

The second result matters beyond raw accuracy: `find_toc_candidates`
can only ever recover authors when a marker word (`par`/`by`/`et`,
`_TOC_AUTHOR_MARKER_RE` at `segmentation.py:155`) appears on the line —
this book's layout has no marker word, so the regex heuristic would return
zero authors here regardless of text quality. The vision model recovered
them anyway, from typographic cues (a distinct all-caps line) a
text-only extractor structurally can't see.

Two more KISSKI models were spot-checked for vision support as a second
independent signal (needed because the whole-book agreement gate requires
two *independent* extractions, not one extractor plus a rubber stamp):
`gemma-4-31b-it` (accurate on a partial read of the first test book, 15.3s)
and `qwen3.6-27b` (accurate but slower — 71-76s — and applied its own
judgment call to omit the back-matter entries `Textnachweise`/`Register`/
`Gesamtinhaltsverzeichnis`, plus left numbering prefixes like `"1. "` on
titles the other models stripped). Both confirm vision support exists
beyond a single model, though `qwen3-omni-30b-a3b-instruct` is clearly the
stronger and faster of the three tested.

### 2.2 Why this replaces the OCR pipeline rather than supplementing it

- It fixes the `needs_ocr` cases directly — `3110139642.pdf` above has no
  usable text layer today and would have needed the full §1 OCR pipeline
  to produce any text at all; vision extraction doesn't care, since it
  never reads the text layer.
- It fixes the dot-leader garbage problem at the source rather than
  compensating for it downstream (§1d's `partial_ratio` fix becomes a
  nice-to-have robustness improvement rather than a load-bearing fix — see
  §3.2).
- It fixes an accuracy gap (`authors` recovery on marker-word-free layouts)
  the OCR pipeline was never going to fix, since that's a limitation of
  the regex heuristic's own logic, not of its input text quality.
- It is simpler to build: no new `ocrmypdf`/`pdfalto`/row-reconstruction
  module, no new cache layer for intermediate OCR text, no row-clustering
  tolerance tuning (§1's open question about 8px being font-size-specific
  disappears entirely). What's needed instead — page-image rendering and a
  vision-capable chat completion call — is less code than §1's design, and
  reuses the existing async/retry/cache machinery in
  `generate_dnb_toc_ground_truth.py` almost as-is (see §3).
- Per-book cost is comparable or better: 6.9-10.5s for one vision call
  vs. 8.4s for OCR+ALTO *alone*, before a text-LLM call still had to run
  on top of that in the old design.

The tradeoff: this corpus (`dnb-toc-only`) is specifically PDFs
pre-filtered to just their TOC pages during acquisition, so "render every
page as an image" is cheap and bounded (1-3 pages typically — confirmed by
spot-checking the manifest). This finding does not generalize to sending
whole-book PDFs as images; it works here because the input is already
small.

## 3. Design

### 3.1 Two-vision-model agreement gate replaces the heuristic/LLM gate

`gate_book`/`align_toc_entries` (`evaluation/dnb_toc_matching.py`) are
already generic over any two `list[TocEntry]` — nothing about them assumes
one side came from regex matching. The gate's two inputs become two
independent vision-model extractions instead of
(`find_toc_candidates` output, `llm_extract_toc_entries` output):

- **New function** `vision_extract_toc_entries(pdf_path: Path, model: str, llm_client) -> list[TocEntry]`
  in `src/chapter_segmentation/segmentation.py` (alongside the existing
  `llm_extract_toc_entries`, same return shape) — renders every page of
  `pdf_path` via `pdftoppm -r 200 -png` to a temp dir, builds one chat
  completion with the adapted prompt (§2.1) plus one `image_url` block per
  page, parses the JSON response the same way `_extract_with_retry`
  already does (reused as-is — it's response-shape-agnostic).
- `_run_book` in `generate_dnb_toc_ground_truth.py` calls
  `vision_extract_toc_entries` twice, once per model, instead of calling
  `find_toc_candidates` once and `llm_extract_toc_entries` once. Both
  calls go through the existing cache
  (`_load_cached_llm_entries`/`_write_cached_llm_entries`), keyed by
  `(book, model)` as it already is — no schema change needed there, since
  the cache is already model-keyed.
- **Model selection.** KISSKI's `/models` endpoint doesn't expose a
  "supports vision" flag (`fetch_kisski_models` only returns
  `id`/`name`/`demand`), so vision-capable models must be identified by a
  curated allowlist, the same pattern `_PREFERRED_MODEL_PATTERNS`
  (`generate_dnb_toc_ground_truth.py`) already uses for the text-LLM
  model, but a separate list since not every strong text model is
  vision-capable: `_VISION_MODEL_PATTERNS = (re.compile(r"^qwen\d+-omni"),
  re.compile(r"^gemma-\d+-"))`, in that preference order (matches §2.1's
  finding that the omni model is faster and more consistent than
  `gemma-4-31b-it`). `_select_best_model`'s existing logic is reused, but
  needs two picks instead of one — extract a `_select_best_models(models,
  patterns, count=2) -> list[str]` that walks the same preference-ordered
  loop but keeps collecting until it has `count` distinct model ids
  (falling through if a pattern doesn't resolve to two candidates
  eventually) instead of returning on the first hit. If only one
  vision-capable model is reachable at run time (e.g. the other is fully
  down), fail loudly rather than silently gating against a single model
  called twice — the whole point of the gate is two *independent* reads,
  and calling one model twice measures its self-consistency, not
  agreement.
- `find_toc_candidates` and `llm_extract_toc_entries` (text-based) are
  **not called** in this pipeline anymore. They remain untouched, in
  production use elsewhere (`analyze_attachment_with_llm_fallback` and
  friends) — this change is scoped to `dnb-toc-only` GT generation only,
  same boundary the §1 design already had.
- `needs_ocr`/`pages_need_ocr` and the whole `extract_page_texts_for_analysis`
  call in `_run_book` are **deleted** from this script — nothing in the
  new path reads the embedded text layer at all, so there's nothing left
  to check it for.

### 3.2 Title-matching robustness fix (kept from the original design)

Still worth doing even though vision output is far cleaner than
OCR-garbled text: change `align_toc_entries`'s scoring from
`fuzz.token_sort_ratio(...)` to `max(fuzz.token_sort_ratio(...),
fuzz.partial_ratio(...))`. §2.1's `qwen3.6-27b` run showed a live example
of why it's still useful even post-vision — its titles kept a `"1. "`
numbering prefix the other model stripped, which is exactly the kind of
"real match, extra leading tokens" case `partial_ratio` handles and
`token_sort_ratio` doesn't. Small, independent, no dependency on anything
else in this spec — land it first.

### 3.3 Caching and cost

Vision responses are cached exactly like today's text-LLM responses (same
`llm_cache_dir`/JSON-per-book-per-model file, no new cache directory
needed — this removes the `.ocr-cache/` addition §1's design would have
needed). Per-book cost: two sequential-per-model but concurrent-per-book
vision calls, ~7-15s each for the faster model pairing tested. At the
existing script's `--concurrency` pattern (already used for the text-LLM
calls) this should clear the full ~1251-book corpus in well under the
"significant time cost" the user already accepted for this work — no
OCR/ALTO subprocess overhead is added on top, unlike §1's design.

## 4. Testing

- `vision_extract_toc_entries`: integration-tested against 3-5 real corpus
  books (including the two from §2.1, since their correct output is now
  known), not unit-tested with a mocked vision response — matches how
  `llm_extract_toc_entries` itself is tested today (real KISSKI calls in
  its test suite) and how `pdfalto_runner.py` is tested elsewhere in this
  project.
- `_select_best_models`: unit-tested against a fabricated model list
  covering "both patterns resolve", "only one pattern resolves, need a
  second from a lower-preference pattern", and "fewer than 2 vision models
  available anywhere → raises" — mirroring the existing
  `_select_best_model` test structure.
- The §3.2 `partial_ratio` change: unit-tested directly in
  `tests/test_dnb_toc_matching.py` with the exact garbled/clean pairs
  measured in §1d, plus the existing negative-control pairs already
  verified there.
- Re-run the existing 60-book smoke test after integration and compare
  pass-rate/skip-reason breakdown against the last recorded run (6/60
  passed, 35 below_threshold, 19 needs_ocr) to quantify the actual
  improvement before declaring this done.

## 5. Open questions for review

- §2.1's two-book sample is promising but small — worth widening to
  10-20 books spanning different eras/layouts before fully trusting
  `qwen3-omni-30b-a3b-instruct` + `gemma-4-31b-it` as the standing model
  pair, ideally as part of task 1 of the implementation plan rather than a
  separate up-front step, since the real integration test doubles as that
  wider sample.
- `qwen3.6-27b`'s judgment call to skip `Textnachweise`/`Register`/
  `Gesamtinhaltsverzeichnis` raises a prompt-wording question: is
  "skip acknowledgements, bibliography, index" (carried over from the
  text prompt) making some models over-eager to drop legitimate back-matter
  entries with generic-sounding titles? Worth checking whether the chosen
  model pair agrees on this category consistently, or whether the prompt
  needs a clarifying example.
- No image-count/size ceiling has been chosen yet for
  `vision_extract_toc_entries` — this corpus's PDFs are short today, but
  the function should probably still cap or warn past some page count
  (e.g. 20) rather than silently building an arbitrarily large multi-image
  request if an outlier book slips through acquisition filtering.
