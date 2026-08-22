# Crossref evaluation corpus (revision of 2026-08-21 crossref cross-validation)

## Why this revision

Running `cli/backfill_crossref.py` + `cli/evaluate_crossref.py` for real
against the live corpus (2026-08-21) surfaced two things the original
design's mocked unit tests couldn't:

1. **Two real bugs** in how Crossref data was compared against ground
   truth -- a page-number-glued-onto-title artifact in some publishers'
   Crossref metadata, and `matching.diff_toc_entries`'s greedy alignment
   assuming both sides are already in page order (true for two
   independent TOC-page reads, false for Crossref's registration-order
   API response). Both are now fixed in `crossref.py`/`evaluate_crossref.py`
   directly (see their own commit history) and are orthogonal to this
   revision.
2. **A architecture gap**: comparing directly against
   `.crossref-cache/<isbn>.crossref.json` (private, gitignored, holds
   every Crossref-registered item including ones with no page number at
   all) meant the evaluation script had to re-derive "is this book even
   usable for evaluation" logic every run, the comparison couldn't be
   inspected/reviewed without re-running the tool, and books whose
   Crossref registration has no page data at all produced a misleading
   "0% agreement" instead of being correctly excluded as unevaluable.

This revision introduces a **committed, filtered evaluation corpus**
(`data/corpus/pilot/evaluation/<key>.expected.json`) as the actual input
to comparison, derived from (but distinct from) the raw
`.crossref-cache/` fetch cache, which continues to exist unchanged as the
private cache `fetch_crossref_book` reads/writes.

## What changes

### 1. A new committed corpus directory: `data/corpus/pilot/evaluation/`

Not gitignored (unlike `.crossref-cache/`) -- this is real, reviewable
corpus data, the same tracked-file discipline as
`data/corpus/pilot/ground-truth/`. `src/dnb_toc_ground_truth/corpus.py`
gains:

```python
def evaluation_dir() -> Path:
    return CORPUS_DIR / "evaluation"

def evaluation_json_path(key: str) -> Path:
    return evaluation_dir() / f"{key}.expected.json"
```

### 2. A new `crossref.py` function: write the filtered file

```python
_DEFAULT_MIN_CHAPTERS_FOR_EVAL = 3

def write_evaluation_entry(
    isbn: str,
    crossref_data: CrossrefBookData,
    eval_dir: Path,
    min_chapters: int = _DEFAULT_MIN_CHAPTERS_FOR_EVAL,
) -> bool:
    """Filters crossref_data.chapters to those with a non-None
    printed_page_number (a chapter Crossref registered with no page data
    at all can never be matched by matching.diff_toc_entries's
    page-then-title alignment -- see its own docstring's "A None-page
    entry_a never matches a known-page entry_b" rule -- so including it
    would only ever inflate only_in_crossref noise, never contribute a
    real match). Writes <eval_dir>/<isbn>.expected.json in the same
    {"entries": [...]} shape ground-truth files use (each entry via
    toc_entry_to_gt_dict-compatible fields), plus a top-level
    "source": "crossref" (no "verified" field -- this is evaluation-only
    data, not a ground-truth candidate, so "verified" doesn't apply),
    IFF at least min_chapters chapters survive the page-data filter.
    Returns whether the file was written. Overwrites any existing file
    for this isbn unconditionally (this is a derived artifact, not
    something hand-edited -- always safe to regenerate from the cache)."""
```

Entry shape written (mirrors `toc_entry_to_gt_dict`, `skip` always
`False` since every included item is a genuine `book-chapter`-typed
Crossref record):

```json
{
  "entries": [
    {"title": "...", "authors": [...], "printed_page_number": "49", "skip": false}
  ],
  "source": "crossref",
  "fetched_at": "2026-08-22T..."
}
```

### 3. Both write paths call it

- `cli/backfill_crossref.py`'s `backfill()`: after
  `fetch_crossref_book()`, calls `crossref.write_evaluation_entry(isbn,
  data, corpus.evaluation_dir(), min_chapters)` unconditionally (whether
  or not a doi was found -- independent concerns, same as chapter
  caching already is). Gains a `--min-chapters` CLI flag (default 3).
- `cli/fetch_corpus.py`'s `_acquire_record()`: same call, right after
  the existing `fetch_crossref_book()` call. Gains a `--min-chapters`
  CLI flag (default 3), threaded the same way `--contact-email` already
  is.

### 4. `cli/evaluate_crossref.py` -- rewritten to read the evaluation corpus, report precision/recall/F1

No longer touches `.crossref-cache/` or `CrossrefBookData` at all --
scoping and comparison both now come from paired committed files:

```python
def evaluate_corpus() -> tuple[list[BookMetrics], list[str]]:
    """Returns (results, keys_with_no_evaluation_data) for every manifest
    book that has BOTH a .expected.json (ground truth) AND an
    evaluation/<key>.expected.json (Crossref evaluation corpus)."""
```

For each such book: load ground truth, filter to `not skip`; load the
evaluation-corpus file's entries directly as `TocEntry` (already
`skip=False`, already page-filtered -- no further filtering needed);
`matching.diff_toc_entries(gt_real, crossref_entries)` (unchanged, still
completely unmodified, still sorted by page first per the 2026-08-21
ordering fix). From `(matched, only_in_gt, only_in_crossref)`:

```
TP = len(matched)
FN = len(only_in_gt)         # a real GT chapter Crossref didn't register/match
FP = len(only_in_crossref)   # a Crossref chapter with no GT match
precision = TP / (TP + FP) if (TP + FP) else 0.0
recall    = TP / (TP + FN) if (TP + FN) else 0.0
f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
```

Reports per-book precision/recall/F1, plus an aggregate: the **macro
average** (mean of each book's own precision/recall/F1, every book
weighted equally regardless of chapter count) -- matches the existing
tool's "mean agreement" precedent from the 2026-08-21 design, and is
explicitly documented as a stated methodology constraint (see README
section below), not silently chosen. `--min-agreement` (unchanged
option, now gating on mean F1 instead of the old ad-hoc agreement rate)
remains available for eventual CI use, still off by default.

### 5. README methodology section

`README.md` gains a "Crossref evaluation" section (placed after
"Methodology", before "Setup") documenting: what this checks and why
(an independent cross-check against a second, non-LLM source), the
`min_chapters`/page-data filter and why it exists (a book with too few
Crossref-registered, page-numbered chapters isn't a meaningful sample),
how to run it (`backfill_crossref.py` then `evaluate_crossref.py`), and
its constraints (macro-averaged across books; only checks books
Crossref happens to have adequately registered, which skews toward
larger/more prominent publishers and is not a random sample of the
corpus; a Crossref "miss" doesn't necessarily mean the ground truth is
wrong -- Crossref's own chapter registration can itself be incomplete or
differently-scoped, e.g. a handbook's many short entries registered as
one part rather than individually).

## Non-goals (unchanged from the 2026-08-21 design, still apply)

No automatic ground-truth correction; no chapter data pulled for
non-ISBN-keyed manifest entries.

## Migration

The existing 58 already-cached `.crossref-cache/*.crossref.json` files
(from the 2026-08-21 real run) are NOT retroactively migrated by any
script change alone -- `write_evaluation_entry` only ever gets called
from the two write paths above (`backfill_crossref.py` re-run,
`fetch_corpus.py` future acquisitions). Re-running
`cli/backfill_crossref.py --force` after this lands regenerates
`.crossref-cache/` AND populates `data/corpus/pilot/evaluation/` for
every book still eligible (has ground truth, no doi -- see the
2026-08-21 spec's own note that a book which already found a doi is
permanently outside `_needs_backfill`'s filter, a known, unchanged
limitation carried forward from that design). The resulting
`data/corpus/pilot/evaluation/*.expected.json` files this produces
should be committed as real corpus data, the same as the 87 backfilled
DOIs were.
