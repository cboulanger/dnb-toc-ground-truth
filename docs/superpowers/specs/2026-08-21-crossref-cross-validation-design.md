# Crossref cross-validation for the DNB-toc ground truth

## Problem

This corpus's ground truth (`data/corpus/pilot/ground-truth/*.expected.json`)
comes entirely from independent LLM reads of scanned TOC images
(`matching.py`'s two-model gate, or agent arbitration) -- there is no
independent, human-curated data source it's ever been checked against.
Crossref registers per-chapter metadata (title, authors, page range) for a
large fraction of edited volumes, keyed by the same ISBN this corpus already
tracks. Where both exist for a book, comparing them is a free, independent
correctness signal this corpus has never had.

chapter-segmentation already has this exact capability --
`src/chapter_segmentation/evidence/crossref_strategy.py` fetches and caches
a book's Crossref-registered chapter list by ISBN -- built for a different
purpose (as one candidate-metadata source feeding chapter-segmentation's own
strategy pipeline). This design ports the fetch/cache mechanism into this
repo (adapted to its own conventions: sync `httpx.Client`, `TocEntry`
output, `.config.json`-driven contact email) and adds a book-level DOI
lookup alongside it, then uses this repo's own `matching.py` (already built
to diff two independently-produced `TocEntry` lists) to compute agreement
against the ground truth, unmodified.

## Scope

1. For every manifest entry that already has a `.expected.json` but no
   `doi`, look up its ISBN on Crossref; if Crossref has a DOI for the book,
   write it into `manifest.json` -- regardless of whether Crossref also has
   usable chapter data for that book.
2. For every manifest entry Crossref has chapter data for, pull and cache
   it locally, keyed by ISBN.
3. Wire the same fetch+cache into `cli/fetch_corpus.py` so a newly acquired
   book gets its Crossref data immediately, without waiting for a separate
   backfill run.
4. A new evaluation script measures agreement, per book and in aggregate,
   between each book's ground-truth chapters (`"skip": false` only) and its
   cached Crossref chapter list, on title (chapter-number-prefix and
   capitalization normalized) and first page -- reusing `matching.py`'s
   existing diff logic exactly as built for two independent LLM reads.

## Module 1: `src/dnb_toc_ground_truth/crossref.py` (new)

Ported from `crossref_strategy.py`'s `normalize_isbn` and Crossref-query
plumbing (base URL, retry-on-429 loop, never-raises-on-failure discipline),
combined with `evaluation/oa_license.py`'s unfiltered-type query style (that
module already queries `/works?filter=isbn:X` without restricting `type=`,
since it needs non-book-chapter items too for license resolution -- the
same shape this design needs for the book-level DOI).

**Key simplification over the source repo**: Crossref's
`/works?filter=isbn:X` response contains every work type registered under
that ISBN in one call -- not just `book-chapter`. One HTTP request serves
both the book-level DOI (the first item whose `type` is not
`"book-chapter"`, preferring `"book"`/`"monograph"`/`"edited-book"`) and the
chapter list (every `"book-chapter"`-typed item) at once, cached together.

```python
@dataclass(frozen=True)
class CrossrefBookData:
    isbn: str
    doi: Optional[str]
    chapters: tuple[TocEntry, ...]
    fetched_at: str

def normalize_isbn(raw: str) -> Optional[str]: ...  # ported verbatim from crossref_strategy.py

def fetch_crossref_book(
    isbn: str,
    client: httpx.Client,
    contact_email: Optional[str],
    cache_dir: Path,
    force: bool = False,
) -> CrossrefBookData:
    """GET .../works?filter=isbn:{isbn}, select=DOI,title,subtitle,author,page,type.
    Partitions items into the book-level doi (first non-book-chapter item's
    DOI, or None) and chapters (every book-chapter item, projected to
    TocEntry: title = title[0] + subtitle[0] if present (same truncated-
    heading fix as _parse_crossref_item's docstring), authors from
    author[].given/family, printed_page_number = the first page of the
    "page" range string, skip=False -- a Crossref book-chapter record is
    never a part-divider or front/back-matter entry).

    Cached on disk at <cache_dir>/<isbn>.crossref.json (doi + chapters +
    fetched_at), including a doi=None/chapters=() miss, so an
    unregistered ISBN is never re-queried on repeat runs. force=True
    bypasses the cache read (but still overwrites it with the fresh
    result). Any network/HTTP-status/JSON-shape failure is logged and
    treated as CrossrefBookData(doi=None, chapters=()) -- never raises."""
```

`src/dnb_toc_ground_truth/corpus.py` gains:

```python
def crossref_cache_dir() -> Path:
    return CORPUS_DIR / ".crossref-cache"
```

`.gitignore` gains `data/corpus/pilot/.crossref-cache/`, alongside the
existing `.lobid-cache/`/`.locks/`/`.layout-cache/` entries.

`.config.json` / `.config.json.dist` gain a `"contact_email"` key (used for
Crossref's polite-pool `mailto=` parameter); every script below reads it via
the existing `load_config()` helper (`inference.py`), with an optional
`--contact-email` CLI flag overriding it -- same
flag-then-config-then-hardcoded-default resolution order
`_resolve_endpoints` already uses for `use_vision`/`concurrency`. No
hardcoded email constant in source.

## Module 2: `cli/fetch_corpus.py` -- real-time hook

In `_acquire_record()`, immediately after `_append_book()` writes the new
manifest entry: if `_record_key(record)` is a valid ISBN
(`crossref.normalize_isbn` accepts it), call `fetch_crossref_book()`. If it
returns a `doi`, the manifest entry just written gets that `doi` value
before the append (replacing the `null` lobid leaves behind -- lobid rarely
carries one, per `_record_doi`'s existing docstring). The chapter cache is
written as a side effect of `fetch_crossref_book()` itself, regardless of
whether a DOI was found. A Crossref failure here is swallowed the same way
a PDF-download failure already is -- one book's Crossref hiccup must not
abort a multi-hour `--from-dump` run.

`main()` gains `--contact-email` (default from `.config.json`) and a
`--crossref-cache-dir` override, mirroring the existing `--manifest-path`
override pattern.

## Module 3: `cli/backfill_crossref.py` (new)

One-time/rerunnable script for the existing ~1251-book backlog:

```python
uv run python cli/backfill_crossref.py
uv run python cli/backfill_crossref.py --force
uv run python cli/backfill_crossref.py --contact-email you@example.org
```

Iterates `corpus.load_manifest_books()`, filters to books where
`corpus.expected_json_path(corpus.manifest_key(book)).exists()` and
`not book.get("doi")`. For each, calls `fetch_crossref_book()`
(`force=args.force`); if a `doi` came back, writes it into that manifest
entry. Manifest is only rewritten if at least one entry actually changed
(same `manifest_changed` guard pattern `fetch_crossref_gt_corpus.py` uses).
Chapter data is cached by `fetch_crossref_book()` itself regardless of
whether a DOI was found or already present -- an entry that already had a
`doi` some other way (there are 2 today) is out of scope for this script
(the manifest-write step), but if it's also missing its chapter cache, nothing
here backfills that; re-running with `--force` after Module 1 lands is the
one-time way to backfill those two.

Prints a summary: books checked, DOIs newly found, chapter-lists cached
(non-empty vs. empty), books skipped (already cached, no `--force`).

## Module 4: `cli/evaluate_crossref.py` (new)

```python
uv run python cli/evaluate_crossref.py
uv run python cli/evaluate_crossref.py --min-agreement 0.8
```

For every book with a `.expected.json`:

1. Load its ground truth, build `TocEntry` objects from the `"entries"`
   list (same fields `matching.toc_entry_to_gt_dict` serializes), filter to
   `not entry.skip`.
2. Load its cached `.crossref-cache/<isbn>.crossref.json` (skip the book
   entirely -- report separately as "no crossref data" -- if the cache file
   doesn't exist or `chapters` is empty).
3. `matched_pairs, only_in_gt, only_in_crossref = matching.diff_toc_entries(gt_real, crossref_chapters)`
   -- reused completely unmodified. `diff_toc_entries` already aligns on
   title (via `_title_score`/`_title_near_identical`, which strip a leading
   chapter/section number and normalize case/punctuation before comparing)
   plus first-page-number equivalence (`_pages_equivalent`), which is
   exactly this design's bullet-4 comparison spec -- no new matching code
   needed.
4. `agreement_rate = len(matched_pairs) / max(len(gt_real), len(crossref_chapters))`.

Prints a per-book line (`key`, `agreement_rate`, counts of
matched/only-in-gt/only-in-crossref) and an aggregate summary (mean
agreement rate across covered books, count of books with no Crossref
coverage at all). `--min-agreement` is an optional gate: exit code 1 if the
aggregate mean falls below it, for eventual CI use -- off by default (no
threshold enforced) since this is a new, unvalidated signal, not yet a
merge gate.

## Testing

- `tests/test_crossref.py`: `fetch_crossref_book` against a mocked
  `httpx.Client` (a fake transport, same pattern `test_fetch_corpus.py`
  already uses for `httpx.Client`) -- cache hit/miss, a response mixing
  `book`/`book-chapter`/`book-part` types, a 429-then-200 retry, a
  malformed-JSON response, `normalize_isbn` edge cases (ISBN-10 with `X`,
  hyphenated input, garbage).
- `tests/test_backfill_crossref.py`: the manifest-entry filter (has
  expected.json + no doi → included; either condition false → excluded),
  the doi-write, the `manifest_changed` no-op-when-nothing-found case.
- `tests/test_evaluate_crossref.py`: `diff_toc_entries` is already tested
  in `test_matching.py`, so this only needs to test the surrounding glue --
  the `skip` filter, the agreement-rate arithmetic, the "no crossref data"
  skip path, `--min-agreement` exit code.
- `tests/test_fetch_corpus.py` gains a case: `_acquire_record` calls the
  Crossref hook and the appended manifest entry's `doi` reflects what it
  returned.

## Non-goals

- No attempt to reconcile a *disagreement* automatically (no arbitration
  hook, no rewriting `.expected.json` from Crossref data) -- this is a
  read-only cross-check, not a ground-truth-correction pipeline.
- No chapter data pulled for books without a usable ISBN key (the
  lobid-numeric-id-keyed entries) -- Crossref has nothing to look up by in
  that case.
