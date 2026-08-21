# dnb-toc-only Ground Truth Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the tooling that produces `evaluation/corpus/dnb-toc-only/<id>.expected.json` structured ground truth (title/authors/printed_page_number per TOC entry), per `docs/superpowers/specs/2026-08-15-dnb-toc-ground-truth-generation-design.md`: a whole-book agreement-gated bulk-tier generator, a stratified eval-tier sampler, and the docs for the manual eval-tier transcription workflow.

**Architecture:** Two independent extractors already exist and both return `list[TocEntry]` (`find_toc_candidates`, `llm_extract_toc_entries` — `src/chapter_segmentation/segmentation.py`); a new pure-logic module (`evaluation/dnb_toc_matching.py`) aligns and gates their outputs, and a new orchestration script (`evaluation/scripts/generate_dnb_toc_ground_truth.py`) wires extraction, caching, concurrency, and file output around it. A separate script (`evaluation/scripts/select_dnb_toc_eval_sample.py`) picks the held-out eval tier before the bulk generator ever runs.

**Tech Stack:** Python 3.12, `pypdf`, `rapidfuzz`, `httpx`/`openai` (KISSKI), `asyncio`, stdlib `unittest`, `uv run pytest`.

**Correction found during planning, not present in the design spec:** `find_toc_candidates` rejects any printed page number above `len(pages) * _TOC_MAX_PAGE_NUMBER_RATIO` (2.0, `segmentation.py:124`) — a guard against mistaking book-internal noise for a real TOC. A `dnb-toc-only` PDF is only 1-3 pages long but prints page numbers from the *original full book*, so calling `find_toc_candidates` on its raw page list would silently reject nearly every real entry. Task 5 below fixes this the same way `tests/test_segmentation.py`'s own `_FILLER_PAGES` fixture already works around it in tests: pad with harmless filler pages before the call. No change to `segmentation.py` itself.

---

### Task 1: `align_toc_entries` — pair-returning alignment

**Files:**
- Create: `evaluation/dnb_toc_matching.py`
- Test: `tests/test_dnb_toc_matching.py`

- [ ] **Step 1: Write the failing test**

```python
"""Unit tests for evaluation/dnb_toc_matching.py -- the whole-book
agreement gate that decides which dnb-toc-only books' extracted entries
are trustworthy enough for the bulk ground-truth tier (design spec
docs/superpowers/specs/2026-08-15-dnb-toc-ground-truth-generation-design.md
section 4). No PDFs, no network -- pure functions over synthetic
TocEntry lists."""

import unittest

from chapter_segmentation.segmentation import TocEntry
from evaluation.dnb_toc_matching import align_toc_entries


def _entry(title: str, page: int, authors: tuple[str, ...] = ()) -> TocEntry:
    return TocEntry(title=title, printed_page_number=page, source_page_index=0, authors=authors)


class TestAlignTocEntries(unittest.TestCase):
    def test_matches_same_page_and_similar_title(self):
        a = [_entry("Einleitung", 9)]
        b = [_entry("Einleitung", 9)]
        self.assertEqual(align_toc_entries(a, b), [(0, 0)])

    def test_no_match_on_page_mismatch(self):
        a = [_entry("Einleitung", 9)]
        b = [_entry("Einleitung", 11)]
        self.assertEqual(align_toc_entries(a, b), [])

    def test_no_match_when_either_page_unknown(self):
        a = [_entry("Einleitung", -1)]
        b = [_entry("Einleitung", 9)]
        self.assertEqual(align_toc_entries(a, b), [])

    def test_no_match_on_dissimilar_title_same_page(self):
        a = [_entry("Einleitung", 9)]
        b = [_entry("Bibliographie", 9)]
        self.assertEqual(align_toc_entries(a, b), [])

    def test_order_preserving_scan_misses_out_of_order_match(self):
        # Same "TOC order is book order" monotonicity tradeoff
        # src/chapter_segmentation/evidence/fusion.py's _align already
        # makes: a[0] ("Erster Teil", page 9) can only match b at or after
        # b-index 0. b[0] ("Zweiter Teil", page 40) doesn't match, so the
        # scan advances to b[1] ("Erster Teil", page 9), which does --
        # consuming last_j=1. a[1] ("Zweiter Teil", page 40) then has no
        # b-index left to scan (>= 2), even though a real match existed
        # earlier in b. This is expected behavior, not a bug.
        a = [_entry("Erster Teil", 9), _entry("Zweiter Teil", 40)]
        b = [_entry("Zweiter Teil", 40), _entry("Erster Teil", 9)]
        self.assertEqual(align_toc_entries(a, b), [(0, 1)])

    def test_empty_lists(self):
        self.assertEqual(align_toc_entries([], []), [])
        self.assertEqual(align_toc_entries([_entry("X", 1)], []), [])
        self.assertEqual(align_toc_entries([], [_entry("X", 1)]), [])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_dnb_toc_matching.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'evaluation.dnb_toc_matching'`

- [ ] **Step 3: Write minimal implementation**

```python
"""Whole-book agreement gate for dnb-toc-only ground truth -- see design
spec docs/superpowers/specs/2026-08-15-dnb-toc-ground-truth-generation-design.md
section 4. Pure functions over TocEntry lists produced by the two
existing extractors (find_toc_candidates, llm_extract_toc_entries --
src/chapter_segmentation/segmentation.py), which already return the
identical list[TocEntry] shape."""

from rapidfuzz import fuzz

from chapter_segmentation.segmentation import TocEntry

# Same constant src/chapter_segmentation/evidence/fusion.py's _align uses
# for its own title-similarity matching.
_ALIGN_SCORE_THRESHOLD = 70.0


def align_toc_entries(a: list[TocEntry], b: list[TocEntry]) -> list[tuple[int, int]]:
    """Greedy, order-preserving alignment between two independently-
    produced TocEntry lists for the same TOC scan. A pair (i, j) counts as
    a match only when both sides have a KNOWN printed_page_number (neither
    is the -1 "unknown" sentinel) that's numerically equal, AND their
    titles score >= _ALIGN_SCORE_THRESHOLD on rapidfuzz's
    token_sort_ratio -- mirrors evaluation/nuextract_baseline.py's
    match_toc_entries (page-number-first, then title) and
    src/chapter_segmentation/evidence/fusion.py's _align (greedy scan from
    the last matched b-index, "TOC order is book order"), but returns
    index PAIRS rather than a bare count, since the whole-book gate below
    needs to know exactly which entries agreed."""
    pairs: list[tuple[int, int]] = []
    last_j = -1
    for i, entry_a in enumerate(a):
        if entry_a.printed_page_number == -1:
            continue
        best_j = None
        best_score = _ALIGN_SCORE_THRESHOLD
        for j in range(last_j + 1, len(b)):
            entry_b = b[j]
            if entry_b.printed_page_number != entry_a.printed_page_number:
                continue
            score = fuzz.token_sort_ratio(entry_a.title.lower(), entry_b.title.lower())
            if score >= best_score:
                best_score = score
                best_j = j
        if best_j is not None:
            pairs.append((i, best_j))
            last_j = best_j
    return pairs
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_dnb_toc_matching.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add evaluation/dnb_toc_matching.py tests/test_dnb_toc_matching.py
git commit -m "feat: add align_toc_entries for dnb-toc-only GT agreement matching"
```

---

### Task 2: `gate_book` — whole-book threshold gate with union merge

**Files:**
- Modify: `evaluation/dnb_toc_matching.py`
- Modify: `tests/test_dnb_toc_matching.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dnb_toc_matching.py`:

```python
from evaluation.dnb_toc_matching import align_toc_entries, gate_book  # noqa: F811 (extends the existing import line)


class TestGateBook(unittest.TestCase):
    def test_perfect_agreement_passes_with_union_equal_to_either_list(self):
        h = [_entry("Einleitung", 9), _entry("Schluss", 40)]
        l = [_entry("Einleitung", 9), _entry("Schluss", 40)]
        passed, entries = gate_book(h, l)
        self.assertTrue(passed)
        self.assertEqual([e.title for e in entries], ["Einleitung", "Schluss"])

    def test_below_threshold_rejects_whole_book(self):
        h = [_entry("Einleitung", 9), _entry("A", 20), _entry("B", 30), _entry("C", 40)]
        l = [_entry("Einleitung", 9)]  # agreement_rate = 1/4 = 0.25
        passed, entries = gate_book(h, l)
        self.assertFalse(passed)
        self.assertEqual(entries, [])

    def test_above_threshold_unions_singleton_entries_rather_than_dropping_them(self):
        # 9 of 10 heuristic entries agree with the LLM list -- rate 0.90.
        # The heuristic's 10th, LLM-missed entry must survive in the
        # merged result rather than being silently trimmed: the design's
        # core "no incomplete training target" requirement (spec section
        # 4.2) -- once a book clears the trust bar, a line only one
        # extractor caught is more likely a real miss than a hallucination.
        h = [_entry(f"Chapter {i}", i * 10) for i in range(1, 11)]
        l = h[:9]
        passed, entries = gate_book(h, l, threshold=0.90)
        self.assertTrue(passed)
        self.assertEqual(len(entries), 10)
        self.assertIn("Chapter 10", [e.title for e in entries])

    def test_empty_both_lists_rejects(self):
        self.assertEqual(gate_book([], []), (False, []))

    def test_merged_entries_sorted_by_printed_page_number(self):
        h = [_entry("Schluss", 40), _entry("Einleitung", 9)]
        l = [_entry("Schluss", 40), _entry("Einleitung", 9)]
        passed, entries = gate_book(h, l)
        self.assertTrue(passed)
        self.assertEqual([e.printed_page_number for e in entries], [9, 40])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_dnb_toc_matching.py -v`
Expected: FAIL with `ImportError: cannot import name 'gate_book'`

- [ ] **Step 3: Write minimal implementation**

Append to `evaluation/dnb_toc_matching.py`:

```python
def gate_book(
    heuristic: list[TocEntry], llm: list[TocEntry], threshold: float = 0.90,
) -> tuple[bool, list[TocEntry]]:
    """Whole-book agreement gate (design spec section 4.2).
    agreement_rate = matched-pair count / max(len(heuristic), len(llm)).
    Below `threshold`, the book is rejected outright (passed=False,
    entries=[]) rather than trimmed down to just the agreeing entries -- a
    partially-agreeing book is exactly the case this design distrusts
    most, and a caller must not silently write a partial/incomplete
    result for it.

    At or above `threshold`, `entries` is the UNION of matched pairs
    (the heuristic's own TocEntry preferred over its LLM counterpart,
    since its title/author split comes from structured regex capture
    rather than LLM reformatting) plus every singleton entry either
    extractor found alone, ordered by printed_page_number (the -1
    "unknown" sentinel sorts last). This is deliberate: once a book
    clears the trust bar, a line only one extractor caught is far likelier
    a real entry the other missed (OCR noise, an unusual title format)
    than a hallucination -- trimming it out would silently understate the
    page's real content, which is exactly the "incomplete training
    target" failure mode this design exists to avoid."""
    if not heuristic and not llm:
        return False, []
    pairs = align_toc_entries(heuristic, llm)
    agreement_rate = len(pairs) / max(len(heuristic), len(llm))
    if agreement_rate < threshold:
        return False, []
    matched_h = {i for i, _ in pairs}
    matched_l = {j for _, j in pairs}
    merged = [heuristic[i] for i, _ in pairs]
    merged += [entry for i, entry in enumerate(heuristic) if i not in matched_h]
    merged += [entry for j, entry in enumerate(llm) if j not in matched_l]
    merged.sort(key=lambda e: (e.printed_page_number == -1, e.printed_page_number))
    return True, merged
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_dnb_toc_matching.py -v`
Expected: PASS (11 tests)

- [ ] **Step 5: Commit**

```bash
git add evaluation/dnb_toc_matching.py tests/test_dnb_toc_matching.py
git commit -m "feat: add gate_book whole-book agreement threshold with union merge"
```

---

### Task 3: `toc_entry_to_gt_dict` — GT schema serialization

**Files:**
- Modify: `evaluation/dnb_toc_matching.py`
- Modify: `tests/test_dnb_toc_matching.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dnb_toc_matching.py`:

```python
from evaluation.dnb_toc_matching import toc_entry_to_gt_dict  # noqa: F811


class TestTocEntryToGtDict(unittest.TestCase):
    def test_known_page_number_becomes_string(self):
        entry = _entry("Einleitung", 9, authors=("Jane Author",))
        self.assertEqual(
            toc_entry_to_gt_dict(entry),
            {"title": "Einleitung", "authors": ["Jane Author"], "printed_page_number": "9"},
        )

    def test_unknown_page_number_becomes_none(self):
        entry = _entry("Bibliographie", -1)
        self.assertEqual(
            toc_entry_to_gt_dict(entry),
            {"title": "Bibliographie", "authors": [], "printed_page_number": None},
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_dnb_toc_matching.py -v`
Expected: FAIL with `ImportError: cannot import name 'toc_entry_to_gt_dict'`

- [ ] **Step 3: Write minimal implementation**

Append to `evaluation/dnb_toc_matching.py`:

```python
def toc_entry_to_gt_dict(entry: TocEntry) -> dict:
    """Serializes one TocEntry to this corpus's <id>.expected.json entry
    shape (design spec section 2) -- printed_page_number as a string, or
    None for the -1 "unknown" sentinel. Matches
    evaluation/nuextract2_common.py's build_target output shape directly
    (its primary downstream consumer, per the parent program spec's
    section 3), and mirrors how citation_pages is already stored as a
    string elsewhere in this project's ground truth."""
    return {
        "title": entry.title,
        "authors": list(entry.authors),
        "printed_page_number": str(entry.printed_page_number) if entry.printed_page_number != -1 else None,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_dnb_toc_matching.py -v`
Expected: PASS (13 tests)

- [ ] **Step 5: Commit**

```bash
git add evaluation/dnb_toc_matching.py tests/test_dnb_toc_matching.py
git commit -m "feat: add toc_entry_to_gt_dict GT schema serialization"
```

---

### Task 4: `select_dnb_toc_eval_sample.py` — stratified eval-tier sampling

**Files:**
- Create: `evaluation/scripts/select_dnb_toc_eval_sample.py`
- Test: `tests/test_select_dnb_toc_eval_sample.py`

- [ ] **Step 1: Write the failing test**

```python
"""Unit tests for evaluation/scripts/select_dnb_toc_eval_sample.py's pure
stratification logic (design spec
docs/superpowers/specs/2026-08-15-dnb-toc-ground-truth-generation-design.md
section 5). The real file-walking main() is exercised manually against
the real corpus."""

import unittest

from evaluation.scripts.select_dnb_toc_eval_sample import _decade, manifest_key, stratify_sample


class TestManifestKey(unittest.TestCase):
    def test_strips_pdf_extension(self):
        self.assertEqual(manifest_key({"filename": "9783899718188.pdf"}), "9783899718188")


class TestDecade(unittest.TestCase):
    def test_parses_start_date(self):
        self.assertEqual(_decade({"publication": [{"startDate": "2002"}]}), "2000s")

    def test_falls_back_to_unknown_when_absent(self):
        self.assertEqual(_decade({}), "unknown")

    def test_falls_back_to_unknown_when_unparseable(self):
        self.assertEqual(_decade({"publication": [{"startDate": "n.d."}]}), "unknown")


class TestStratifySample(unittest.TestCase):
    def _books(self, n: int, language: str = "de", prefix: str = "book") -> list[dict]:
        return [{"filename": f"{prefix}{i}.pdf", "language": language} for i in range(n)]

    def test_returns_requested_size_when_pool_is_large_enough(self):
        books = self._books(100)
        records = {f"book{i}": {"publication": [{"startDate": "2010"}]} for i in range(100)}
        selected = stratify_sample(books, records, sample_size=20)
        self.assertEqual(len(selected), 20)
        self.assertEqual(len(set(selected)), 20)

    def test_covers_multiple_strata_proportionally(self):
        de_books = self._books(80, "de", prefix="de_book")
        en_books = self._books(20, "en", prefix="en_book")
        books = de_books + en_books
        records = {}
        for b in de_books:
            records[manifest_key(b)] = {"publication": [{"startDate": "2010"}]}
        for b in en_books:
            records[manifest_key(b)] = {"publication": [{"startDate": "1990"}]}
        selected = stratify_sample(books, records, sample_size=50)
        selected_langs = {b["language"] for b in books if manifest_key(b) in selected}
        self.assertEqual(selected_langs, {"de", "en"})

    def test_deterministic_for_fixed_seed(self):
        books = self._books(50)
        records = {f"book{i}": {"publication": [{"startDate": "2010"}]} for i in range(50)}
        first = stratify_sample(books, records, sample_size=10, seed=42)
        second = stratify_sample(books, records, sample_size=10, seed=42)
        self.assertEqual(first, second)

    def test_never_exceeds_available_books(self):
        books = self._books(5)
        records = {f"book{i}": {} for i in range(5)}
        selected = stratify_sample(books, records, sample_size=50)
        self.assertEqual(len(selected), 5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_select_dnb_toc_eval_sample.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
"""Selects a stratified held-out eval-tier sample for dnb-toc-only ground
truth (design spec
docs/superpowers/specs/2026-08-15-dnb-toc-ground-truth-generation-design.md
section 5). Reads each candidate book's .lobid-cache/<id>.lobid.json for
its publication decade and the manifest's language field, and draws a
sample whose decade/language spread mirrors the corpus's own -- so the
held-out eval tier used to score NuExtract fine-tuning, the heuristic
line-parsing harness, and the classifier pilot isn't accidentally
dominated by one era or language.

Not a pytest test, run once (or re-run after the corpus grows further):

    uv run python evaluation/scripts/select_dnb_toc_eval_sample.py --sample-size 75
"""

import argparse
import json
import random
from pathlib import Path

from evaluation.harness import corpus_dir, load_manifest_books

_CORPUS_NAME = "dnb-toc-only"
_DEFAULT_SEED = 20260815


def manifest_key(entry: dict) -> str:
    return Path(entry["filename"]).stem


def _decade(lobid_record: dict) -> str:
    """Best-effort publication decade from a .lobid-cache record's
    "publication" list, e.g. {"startDate": "2002", ...} -> "2000s". Falls
    back to "unknown" when absent or unparseable, so a book with no usable
    date still gets sampled rather than silently excluded from the pool."""
    publication = lobid_record.get("publication") or []
    for event in publication:
        start = event.get("startDate") or event.get("dateStatement")
        if start and start[:4].isdigit():
            return f"{(int(start[:4]) // 10) * 10}s"
    return "unknown"


def stratify_sample(
    books: list[dict], lobid_records: dict[str, dict], sample_size: int, seed: int = _DEFAULT_SEED,
) -> list[str]:
    """books: manifest entries. lobid_records: manifest_key -> parsed
    .lobid-cache JSON (missing entries treated as {}, i.e. "unknown"
    decade). Returns a stratified sample of manifest keys, roughly
    proportional to each (decade, language) stratum's share of `books`,
    capped at sample_size total. Deterministic for a fixed seed/input, so
    re-running with unchanged corpus contents reproduces the same
    sample."""
    strata: dict[tuple[str, str], list[str]] = {}
    for entry in books:
        key = manifest_key(entry)
        decade = _decade(lobid_records.get(key, {}))
        language = entry.get("language") or "unknown"
        strata.setdefault((decade, language), []).append(key)

    total = len(books)
    rng = random.Random(seed)
    selected: list[str] = []
    for stratum_keys in strata.values():
        share = round(sample_size * len(stratum_keys) / total)
        share = min(share, len(stratum_keys))
        selected.extend(rng.sample(stratum_keys, share))

    # Rounding can land a couple of keys under/over sample_size; trim or
    # top up from the remaining pool deterministically rather than drift
    # silently away from the requested size.
    if len(selected) > sample_size:
        selected = rng.sample(selected, sample_size)
    elif len(selected) < sample_size:
        remaining = [k for keys in strata.values() for k in keys if k not in selected]
        selected.extend(rng.sample(remaining, min(sample_size - len(selected), len(remaining))))
    return sorted(selected)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sample-size", type=int, default=75)
    parser.add_argument("--seed", type=int, default=_DEFAULT_SEED)
    args = parser.parse_args()

    cdir = corpus_dir(_CORPUS_NAME)
    books = load_manifest_books(_CORPUS_NAME)
    lobid_records = {}
    for entry in books:
        key = manifest_key(entry)
        lobid_path = cdir / ".lobid-cache" / f"{key}.lobid.json"
        if lobid_path.exists():
            lobid_records[key] = json.loads(lobid_path.read_text(encoding="utf-8"))

    selected = stratify_sample(books, lobid_records, args.sample_size, args.seed)
    output_path = cdir / "eval_tier_ids.json"
    output_path.write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(selected)} eval-tier IDs to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_select_dnb_toc_eval_sample.py -v`
Expected: PASS (9 tests)

- [ ] **Step 5: Commit**

```bash
git add evaluation/scripts/select_dnb_toc_eval_sample.py tests/test_select_dnb_toc_eval_sample.py
git commit -m "feat: add stratified dnb-toc-only eval-tier sample selector"
```

---

### Task 5: `generate_dnb_toc_ground_truth.py` — page-number-guard padding fix

**Files:**
- Create: `evaluation/scripts/generate_dnb_toc_ground_truth.py`
- Test: `tests/test_generate_dnb_toc_ground_truth.py`

- [ ] **Step 1: Write the failing test**

This test proves the real bug found during planning (see plan header) and its fix, using the real `find_toc_candidates` (already tested elsewhere) directly -- no fakes needed here.

```python
"""Unit tests for evaluation/scripts/generate_dnb_toc_ground_truth.py's
pure logic. The real PDF-reading/LLM-calling main() is exercised manually
against the real corpus with a real KISSKI_API_KEY -- see design spec
docs/superpowers/specs/2026-08-15-dnb-toc-ground-truth-generation-design.md."""

import unittest

from chapter_segmentation.segmentation import find_toc_candidates
from evaluation.scripts.generate_dnb_toc_ground_truth import _toc_entries_for_scan

_TOC_PAGE = (
    "Inhaltsverzeichnis\n"
    "Einleitung ..... 9\n"
    "Zur Soziologie des Rechts ..... 17\n"
    "Schlussbetrachtung ..... 89\n"
)


class TestTocEntriesForScan(unittest.TestCase):
    def test_raw_find_toc_candidates_rejects_realistic_page_numbers_on_a_tiny_pdf(self):
        # Demonstrates the bug: on an unpadded 2-page dnb-toc-only-shaped
        # PDF, _TOC_MAX_PAGE_NUMBER_RATIO (2.0) caps plausible page numbers
        # at 2*2=4 -- every real entry above that (9, 17, 89) is rejected.
        pages = [_TOC_PAGE, "digitalisiert durch Deutsche Nationalbibliothek"]
        self.assertEqual(find_toc_candidates(pages), [])

    def test_padded_wrapper_recovers_the_same_entries(self):
        pages = [_TOC_PAGE, "digitalisiert durch Deutsche Nationalbibliothek"]
        entries = _toc_entries_for_scan(pages)
        self.assertEqual(len(entries), 3)
        self.assertEqual(entries[0].title, "Einleitung")
        self.assertEqual(entries[0].printed_page_number, 9)
        self.assertEqual(entries[2].printed_page_number, 89)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_generate_dnb_toc_ground_truth.py -v`
Expected: first test PASSES (it's asserting the *current*, unfixed
behavior of the already-existing `find_toc_candidates`); second test
FAILS with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
"""Generates bulk-tier structured ground truth for dnb-toc-only (design
spec
docs/superpowers/specs/2026-08-15-dnb-toc-ground-truth-generation-design.md).
For every manifest book not held out in eval_tier_ids.json (see
select_dnb_toc_eval_sample.py and evaluation/README.md's "Building
dnb-toc-only ground truth"), runs two independent extractors -- the regex
heuristic (find_toc_candidates) and a KISSKI LLM pass
(llm_extract_toc_entries) -- and writes <id>.expected.json with
"verified": false only when they agree well enough
(evaluation.dnb_toc_matching.gate_book, >=0.90 whole-book agreement).
Books that don't clear the gate are skipped and reported, not partially
written.

Spends real KISSKI API budget (one call per book, not per-model -- see
evaluation/refresh_llm_cache.py's docstring for the shared KISSKI_API_KEY
setup this script reuses). Not a pytest test.

    uv run python evaluation/scripts/generate_dnb_toc_ground_truth.py --limit 50   # smoke test
    uv run python evaluation/scripts/generate_dnb_toc_ground_truth.py               # full corpus
    uv run python evaluation/scripts/generate_dnb_toc_ground_truth.py --spot-check 30
"""

from chapter_segmentation.segmentation import TocEntry, find_toc_candidates

# find_toc_candidates rejects any printed page number above
# len(pages) * _TOC_MAX_PAGE_NUMBER_RATIO (2.0, segmentation.py) -- a
# guard against mistaking book-internal noise for a real TOC elsewhere in
# a full book. A dnb-toc-only PDF *is* the TOC (1-3 pages) but prints page
# numbers from the ORIGINAL BOOK (which can run into the hundreds), so
# calling find_toc_candidates on it unpadded silently rejects nearly every
# real entry. Padding with harmless filler pages before the call -- same
# technique tests/test_segmentation.py's own _FILLER_PAGES fixture already
# uses -- raises the ratio guard's ceiling comfortably above any real
# book's page count without touching segmentation.py. The real content is
# always at the front of the padded list, well within
# find_toc_candidates' default 15% front-scan window regardless of how
# much padding is appended.
_PAGE_NUMBER_GUARD_PADDING = 1000


def _toc_entries_for_scan(pages: list[str]) -> list[TocEntry]:
    """Runs the heuristic regex extractor on a dnb-toc-only book's own
    page texts, working around _TOC_MAX_PAGE_NUMBER_RATIO's tiny-PDF
    false-rejection (see module docstring)."""
    padded = pages + ["Filler page, not part of the digitized TOC scan."] * _PAGE_NUMBER_GUARD_PADDING
    return find_toc_candidates(padded)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_generate_dnb_toc_ground_truth.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add evaluation/scripts/generate_dnb_toc_ground_truth.py tests/test_generate_dnb_toc_ground_truth.py
git commit -m "feat: add dnb-toc-only heuristic extraction with page-number-guard fix"
```

---

### Task 6: LLM extraction cache round-trip and retry helper

**Files:**
- Modify: `evaluation/scripts/generate_dnb_toc_ground_truth.py`
- Modify: `tests/test_generate_dnb_toc_ground_truth.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_generate_dnb_toc_ground_truth.py`:

```python
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

from evaluation.scripts.generate_dnb_toc_ground_truth import (
    _call_with_retry,
    _load_cached_llm_entries,
    _write_cached_llm_entries,
)


class TestLlmCacheRoundTrip(unittest.TestCase):
    def test_round_trips_entries_through_json(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            entries = [
                TocEntry(title="Einleitung", printed_page_number=9, source_page_index=0, authors=("Jane Author",)),
                TocEntry(title="Bibliographie", printed_page_number=-1, source_page_index=1),
            ]
            self.assertIsNone(_load_cached_llm_entries(cache_dir, "book1"))
            _write_cached_llm_entries(cache_dir, "book1", entries)
            loaded = _load_cached_llm_entries(cache_dir, "book1")
            self.assertEqual(loaded, entries)


class TestCallWithRetry(unittest.IsolatedAsyncioTestCase):
    async def test_returns_first_success(self):
        coro_fn = AsyncMock(return_value="ok")
        result = await _call_with_retry(coro_fn, sleep=AsyncMock())
        self.assertEqual(result, "ok")
        coro_fn.assert_awaited_once()

    async def test_retries_then_succeeds(self):
        coro_fn = AsyncMock(side_effect=[RuntimeError("boom"), "ok"])
        result = await _call_with_retry(coro_fn, attempts=3, sleep=AsyncMock())
        self.assertEqual(result, "ok")
        self.assertEqual(coro_fn.await_count, 2)

    async def test_raises_after_exhausting_attempts(self):
        coro_fn = AsyncMock(side_effect=RuntimeError("boom"))
        with self.assertRaises(RuntimeError):
            await _call_with_retry(coro_fn, attempts=2, sleep=AsyncMock())
        self.assertEqual(coro_fn.await_count, 2)
```

Also add `from chapter_segmentation.segmentation import TocEntry` to the
top of the test file if not already present (it already is, from Task 5).

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_generate_dnb_toc_ground_truth.py -v`
Expected: FAIL with `ImportError` (the three new names don't exist yet)

- [ ] **Step 3: Write minimal implementation**

Append to `evaluation/scripts/generate_dnb_toc_ground_truth.py`:

```python
import asyncio
import json
import time
from pathlib import Path
from typing import Optional


def _cache_path(cache_directory: Path, key: str) -> Path:
    return cache_directory / f"{key}.json"


def _load_cached_llm_entries(cache_directory: Path, key: str) -> Optional[list[TocEntry]]:
    path = _cache_path(cache_directory, key)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        TocEntry(
            title=e["title"], printed_page_number=e["printed_page_number"],
            source_page_index=e["source_page_index"], authors=tuple(e["authors"]),
        )
        for e in data["entries"]
    ]


def _write_cached_llm_entries(cache_directory: Path, key: str, entries: list[TocEntry]) -> None:
    cache_directory.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_directory, key)
    data = {
        "generated_at": time.time(),
        "entries": [
            {
                "title": e.title, "printed_page_number": e.printed_page_number,
                "source_page_index": e.source_page_index, "authors": list(e.authors),
            }
            for e in entries
        ],
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


async def _call_with_retry(coro_fn, attempts: int = 3, base_delay: float = 1.0, sleep=asyncio.sleep):
    """Same shape as evaluation/refresh_llm_cache.py's own retry helper
    (3 attempts, exponential backoff from base_delay) -- `sleep` is
    injectable so tests don't actually wait."""
    last_exc: Optional[Exception] = None
    for attempt in range(attempts):
        try:
            return await coro_fn()
        except Exception as exc:  # noqa: BLE001 -- any failure here (network, parse) is retryable
            last_exc = exc
            if attempt < attempts - 1:
                await sleep(base_delay * 2 ** attempt)
    raise last_exc
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_generate_dnb_toc_ground_truth.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add evaluation/scripts/generate_dnb_toc_ground_truth.py tests/test_generate_dnb_toc_ground_truth.py
git commit -m "feat: add dnb-toc-only LLM extraction cache and retry helper"
```

---

### Task 7: `_run_book_pages` — core per-book orchestration

**Files:**
- Modify: `evaluation/scripts/generate_dnb_toc_ground_truth.py`
- Modify: `tests/test_generate_dnb_toc_ground_truth.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_generate_dnb_toc_ground_truth.py`:

```python
import json
import tempfile
from unittest.mock import MagicMock

from evaluation.scripts.generate_dnb_toc_ground_truth import _run_book_pages

_LLM_TOC_RESPONSE = (
    '[{"title": "Einleitung", "authors": [], "printed_page_number": 9}, '
    '{"title": "Zur Soziologie des Rechts", "authors": [], "printed_page_number": 17}, '
    '{"title": "Schlussbetrachtung", "authors": [], "printed_page_number": 89}]'
)


def _fake_llm(response: str):
    llm = MagicMock()
    llm.generate = AsyncMock(return_value=response)
    return llm


class TestRunBookPages(unittest.IsolatedAsyncioTestCase):
    async def test_passing_book_writes_expected_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus_directory = Path(tmp) / "corpus"
            cache_directory = Path(tmp) / "cache"
            corpus_directory.mkdir()
            pages = [_TOC_PAGE, "digitalisiert durch Deutsche Nationalbibliothek"]
            llm = _fake_llm(_LLM_TOC_RESPONSE)
            semaphore = asyncio.Semaphore(1)

            key, passed, reason = await _run_book_pages(
                "9783899718188", pages, llm, semaphore, corpus_directory, cache_directory,
            )

            self.assertEqual(key, "9783899718188")
            self.assertTrue(passed)
            self.assertEqual(reason, "ok")
            gt_path = corpus_directory / "9783899718188.expected.json"
            self.assertTrue(gt_path.exists())
            data = json.loads(gt_path.read_text(encoding="utf-8"))
            self.assertFalse(data["verified"])
            self.assertEqual(len(data["entries"]), 3)
            self.assertEqual(data["entries"][0]["printed_page_number"], "9")

    async def test_needs_ocr_book_is_skipped_without_calling_llm(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus_directory = Path(tmp) / "corpus"
            cache_directory = Path(tmp) / "cache"
            corpus_directory.mkdir()
            llm = _fake_llm(_LLM_TOC_RESPONSE)
            semaphore = asyncio.Semaphore(1)

            key, passed, reason = await _run_book_pages(
                "empty-book", ["", ""], llm, semaphore, corpus_directory, cache_directory,
            )

            self.assertFalse(passed)
            self.assertEqual(reason, "needs_ocr")
            llm.generate.assert_not_called()
            self.assertFalse((corpus_directory / "empty-book.expected.json").exists())

    async def test_below_threshold_book_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus_directory = Path(tmp) / "corpus"
            cache_directory = Path(tmp) / "cache"
            corpus_directory.mkdir()
            pages = [_TOC_PAGE, "digitalisiert durch Deutsche Nationalbibliothek"]
            # LLM disagrees with almost everything the heuristic found.
            llm = _fake_llm('[{"title": "Ganz andere Sache", "authors": [], "printed_page_number": 200}]')
            semaphore = asyncio.Semaphore(1)

            key, passed, reason = await _run_book_pages(
                "disagreeing-book", pages, llm, semaphore, corpus_directory, cache_directory,
            )

            self.assertFalse(passed)
            self.assertEqual(reason, "below_threshold")
            self.assertFalse((corpus_directory / "disagreeing-book.expected.json").exists())

    async def test_cached_llm_entries_are_reused_without_a_new_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus_directory = Path(tmp) / "corpus"
            cache_directory = Path(tmp) / "cache"
            corpus_directory.mkdir()
            pages = [_TOC_PAGE, "digitalisiert durch Deutsche Nationalbibliothek"]
            heuristic_entries = _toc_entries_for_scan(pages)
            _write_cached_llm_entries(cache_directory, "cached-book", heuristic_entries)
            llm = _fake_llm(_LLM_TOC_RESPONSE)
            semaphore = asyncio.Semaphore(1)

            key, passed, reason = await _run_book_pages(
                "cached-book", pages, llm, semaphore, corpus_directory, cache_directory,
            )

            self.assertTrue(passed)
            llm.generate.assert_not_called()
```

Add `from chapter_segmentation.llm import LLMClient` and
`from unittest.mock import AsyncMock, MagicMock` and
`from evaluation.scripts.generate_dnb_toc_ground_truth import (
_toc_entries_for_scan, _write_cached_llm_entries)` to the test file's
imports if not already present from earlier tasks.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_generate_dnb_toc_ground_truth.py -v`
Expected: FAIL with `ImportError: cannot import name '_run_book_pages'`

- [ ] **Step 3: Write minimal implementation**

Append to `evaluation/scripts/generate_dnb_toc_ground_truth.py`:

```python
from chapter_segmentation.llm import LLMClient
from chapter_segmentation.segmentation import llm_extract_toc_entries, pages_need_ocr
from evaluation.dnb_toc_matching import gate_book, toc_entry_to_gt_dict

_GATE_THRESHOLD = 0.90


async def _run_book_pages(
    key: str, pages: list[str], llm_client: LLMClient, semaphore: asyncio.Semaphore,
    corpus_directory: Path, cache_directory: Path,
) -> tuple[str, bool, str]:
    """Core per-book logic, given already-extracted page texts -- kept
    separate from PDF/file reading so it's directly unit-testable with
    synthetic pages and a fake LLMClient, no real PDF needed. Returns
    (key, passed, reason); reason is "ok" on success, else why the book
    was skipped/rejected ("needs_ocr", "no_entries", "below_threshold")."""
    if pages_need_ocr(pages):
        return key, False, "needs_ocr"
    heuristic_entries = _toc_entries_for_scan(pages)
    cached = _load_cached_llm_entries(cache_directory, key)
    async with semaphore:
        if cached is not None:
            llm_entries = cached
        else:
            llm_entries = await _call_with_retry(lambda: llm_extract_toc_entries(pages, llm_client))
            _write_cached_llm_entries(cache_directory, key, llm_entries)
    if not heuristic_entries and not llm_entries:
        return key, False, "no_entries"
    passed, entries = gate_book(heuristic_entries, llm_entries, threshold=_GATE_THRESHOLD)
    if not passed:
        return key, False, "below_threshold"
    gt_path = corpus_directory / f"{key}.expected.json"
    gt_path.write_text(
        json.dumps({"entries": [toc_entry_to_gt_dict(e) for e in entries], "verified": False}, indent=2) + "\n",
        encoding="utf-8",
    )
    return key, True, "ok"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_generate_dnb_toc_ground_truth.py -v`
Expected: PASS (10 tests)

- [ ] **Step 5: Commit**

```bash
git add evaluation/scripts/generate_dnb_toc_ground_truth.py tests/test_generate_dnb_toc_ground_truth.py
git commit -m "feat: add dnb-toc-only per-book gate-and-write orchestration"
```

---

### Task 8: CLI wiring — `main()`, concurrency driver, model selection

**Files:**
- Modify: `evaluation/scripts/generate_dnb_toc_ground_truth.py`

No new test file changes in this task -- `main()`, `_run_all`, and
`_pick_model` are thin I/O/network glue over the already-tested
`_run_book_pages`, following the same "real main() exercised manually"
convention as `select_dnb_toc_eval_sample.py` (Task 4) and this project's
other `evaluation/scripts/*.py` entry points (e.g.
`tests/test_add_toc_ground_truth.py`'s docstring states this convention
explicitly).

- [ ] **Step 1: Add the CLI and driver code**

Append to `evaluation/scripts/generate_dnb_toc_ground_truth.py`:

```python
import argparse
import os

from evaluation.harness import corpus_dir, llm_cache_dir, load_manifest_books
from evaluation.kisski import DEFAULT_KISSKI_BASE_URL, fetch_kisski_models
from evaluation.refresh_llm_cache import _OpenAICompatibleLLMClient
from evaluation.scripts.select_dnb_toc_eval_sample import manifest_key
from chapter_segmentation.segmentation import extract_page_texts_for_analysis

_CORPUS_NAME = "dnb-toc-only"


async def _run_book(
    key: str, pdf_path: Path, llm_client: LLMClient, semaphore: asyncio.Semaphore,
    corpus_directory: Path, cache_directory: Path,
) -> tuple[str, bool, str]:
    """Thin I/O wrapper around _run_book_pages -- reads the real PDF,
    extracts page text the same way production does, and delegates."""
    pages, _ = extract_page_texts_for_analysis(pdf_path.read_bytes())
    return await _run_book_pages(key, pages, llm_client, semaphore, corpus_directory, cache_directory)


def _pick_model(base_url: str, api_key: str) -> str:
    models = fetch_kisski_models(base_url, api_key)
    return min(models, key=lambda m: m.demand).id


async def _run_all(
    keys_and_paths: list[tuple[str, Path]], llm_client: LLMClient, concurrency: int,
    corpus_directory: Path, cache_directory: Path,
) -> list[tuple[str, bool, str]]:
    semaphore = asyncio.Semaphore(concurrency)
    return list(await asyncio.gather(*[
        _run_book(key, path, llm_client, semaphore, corpus_directory, cache_directory)
        for key, path in keys_and_paths
    ]))


def _generate(args: argparse.Namespace) -> int:
    cdir = corpus_dir(_CORPUS_NAME)
    eval_tier_path = cdir / "eval_tier_ids.json"
    eval_tier_ids = set(json.loads(eval_tier_path.read_text(encoding="utf-8"))) if eval_tier_path.exists() else set()

    books = load_manifest_books(_CORPUS_NAME)
    candidates = [
        (manifest_key(b), cdir / b["filename"])
        for b in books
        if manifest_key(b) not in eval_tier_ids and (cdir / b["filename"]).exists()
    ]
    if args.limit is not None:
        candidates = candidates[: args.limit]

    api_key = os.environ["KISSKI_API_KEY"]
    model = _pick_model(DEFAULT_KISSKI_BASE_URL, api_key)
    llm_client = _OpenAICompatibleLLMClient(model, DEFAULT_KISSKI_BASE_URL, api_key)

    results = asyncio.run(_run_all(candidates, llm_client, args.concurrency, cdir, llm_cache_dir(_CORPUS_NAME)))
    passed = [r for r in results if r[1]]
    by_reason: dict[str, int] = {}
    for _, ok, reason in results:
        if not ok:
            by_reason[reason] = by_reason.get(reason, 0) + 1
    print(f"{len(passed)}/{len(results)} books passed the gate and got .expected.json written.")
    for reason, count in sorted(by_reason.items()):
        print(f"  {count} skipped: {reason}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--limit", type=int, default=None, help="Process at most this many books (smoke-test convenience)")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--spot-check", type=int, default=None, metavar="N",
        help="Instead of generating, sample N passing bulk-tier books and walk through a visual Accept/Reject check",
    )
    args = parser.parse_args()
    if args.spot_check is not None:
        return _spot_check(corpus_dir(_CORPUS_NAME), args.spot_check)
    return _generate(args)


if __name__ == "__main__":
    raise SystemExit(main())
```

Note: `_spot_check` is defined in Task 9 -- `main()` references it here
because both belong to the same CLI, but it isn't implemented until the
next task. Move the `if __name__ == "__main__":` block to the very end of
the file after Task 9 adds `_spot_check` (delete the duplicate at the end
of this task's edit once Task 9's code is appended above it).

- [ ] **Step 2: Verify the module still imports cleanly**

Run: `uv run python -c "import evaluation.scripts.generate_dnb_toc_ground_truth"`
Expected: `NameError: name '_spot_check' is not defined` is NOT raised at
import time (Python only evaluates `main()`'s body when called) -- this
command should exit silently with no output. If it raises an
`ImportError`/`SyntaxError` instead, fix that before proceeding.

- [ ] **Step 3: Run the existing test suite to confirm no regression**

Run: `uv run pytest tests/test_generate_dnb_toc_ground_truth.py -v`
Expected: PASS (10 tests, same as Task 7 -- this task added no new tests)

- [ ] **Step 4: Commit**

```bash
git add evaluation/scripts/generate_dnb_toc_ground_truth.py
git commit -m "feat: wire up dnb-toc-only ground-truth generator CLI"
```

---

### Task 9: `--spot-check` mode

**Files:**
- Modify: `evaluation/scripts/generate_dnb_toc_ground_truth.py`

Terminal-driven, interactive (needs a human/Claude looking at real PDFs),
so not unit-tested beyond the smoke check below -- same rationale as
Task 8.

- [ ] **Step 1: Add `_spot_check` and fix the duplicate `__main__` guard**

Insert this above the `if __name__ == "__main__":` block added in Task 8
(and delete Task 8's now-duplicate guard, keeping only one at the true end
of the file):

```python
import random


def _spot_check(cdir: Path, n: int) -> int:
    """Terminal-driven precision check (design spec section 7): sample n
    books that passed the bulk-tier gate, print each one's PDF path and
    generated entries, and prompt for a manual Accept/Reject after
    visually opening the PDF (e.g. via the Read tool's pages param, same
    as the manual eval-tier transcription workflow in evaluation/README.md)
    -- then report measured precision for the >=0.90 gate threshold."""
    passing = sorted(p.name.removesuffix(".expected.json") for p in cdir.glob("*.expected.json"))
    sample = random.sample(passing, min(n, len(passing)))
    accepted = 0
    for key in sample:
        gt = json.loads((cdir / f"{key}.expected.json").read_text(encoding="utf-8"))
        print(f"\n=== {key} ===\nPDF: {cdir / f'{key}.pdf'}")
        print(json.dumps(gt["entries"], indent=2, ensure_ascii=False))
        answer = input("Matches the scan? [y/N] ").strip().lower()
        if answer == "y":
            accepted += 1
    if sample:
        print(f"\nSpot-check precision: {accepted}/{len(sample)} = {accepted / len(sample):.0%}")
    else:
        print("No passing books found to spot-check yet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Verify the module imports and the CLI's --help works**

Run: `uv run python evaluation/scripts/generate_dnb_toc_ground_truth.py --help`
Expected: prints the argparse help text (module docstring first line,
`--limit`, `--concurrency`, `--spot-check` all listed), exits 0.

- [ ] **Step 3: Run the full test suite once more**

Run: `uv run pytest tests/test_generate_dnb_toc_ground_truth.py tests/test_dnb_toc_matching.py tests/test_select_dnb_toc_eval_sample.py -v`
Expected: PASS (all tests from Tasks 1-7)

- [ ] **Step 4: Commit**

```bash
git add evaluation/scripts/generate_dnb_toc_ground_truth.py
git commit -m "feat: add dnb-toc-only bulk-tier spot-check mode"
```

---

### Task 10: Document the eval-tier manual transcription workflow

**Files:**
- Modify: `evaluation/README.md`

- [ ] **Step 1: Add a new subsection**

Insert a new `## Building dnb-toc-only ground truth` section into
`evaluation/README.md`, directly after the existing `## Corpora` section's
`dnb-toc-only` paragraph (search for `A fourth directory,
**\`dnb-toc-only/\`**` to find the right spot):

```markdown
## Building dnb-toc-only ground truth

See
`docs/superpowers/specs/2026-08-15-dnb-toc-ground-truth-generation-design.md`
for the full design. Two tiers, both writing
`evaluation/corpus/dnb-toc-only/<id>.expected.json`
(`{"entries": [{"title", "authors", "printed_page_number"}, ...],
"verified": bool}`):

**Bulk tier** (`"verified": false`, no human review) --
`evaluation/scripts/select_dnb_toc_eval_sample.py` first, then:

```bash
uv run python evaluation/scripts/select_dnb_toc_eval_sample.py --sample-size 75
export KISSKI_API_KEY=$(grep '^KISSKI_API_KEY=' ../zotero-rag/.env | cut -d= -f2-)
uv run python evaluation/scripts/generate_dnb_toc_ground_truth.py
```

Runs two independent extractors per book (the regex heuristic and a
KISSKI LLM pass) and writes `.expected.json` only when they agree on at
least 90% of the book's entries -- see `evaluation/dnb_toc_matching.py`.
Books that don't clear that bar are skipped and reported, not partially
written.

**Eval tier** (`"verified": true`, hand-transcribed, held out of the bulk
tier and never drafted by either extractor) -- for every ID in
`evaluation/corpus/dnb-toc-only/eval_tier_ids.json`:

1. Open the book's `<id>.pdf` (1-3 pages -- the TOC scan itself, no
   chapter-locate search needed, unlike the full-book workflow
   `CLAUDE.md` documents, since the target page *is* the whole PDF).
2. View it directly (`Read` tool, `pages` param).
3. Transcribe every entry the page actually prints, including lines a
   full-book `.expected.json` would mark `skip: true` (bibliography,
   index headers, part dividers) -- this file measures extraction
   fidelity against what's printed, not "which of these are real
   chapters."
4. Save as `<id>.expected.json` with `"verified": true`.

**Spot-checking the bulk tier's real precision:**

```bash
uv run python evaluation/scripts/generate_dnb_toc_ground_truth.py --spot-check 30
```

Samples passing bulk-tier books, prints each one's PDF path and generated
entries, and prompts for a manual Accept/Reject after opening the PDF --
reports a measured precision for the 90% agreement gate. Record the
result in `RESULTS.md`, same convention as every other one-off measurement
in this project.
```

- [ ] **Step 2: Commit**

```bash
git add evaluation/README.md
git commit -m "docs: document dnb-toc-only ground-truth generation workflow"
```

---

## After this plan

This plan builds and unit-tests the tooling only. Running it for real
(the full bulk-tier pass over ~1000 books, the ~50-100-book eval-tier
manual transcription, and the spot-check) spends real KISSKI budget and
substantial human/Claude time, and depends on the corpus actually
reaching its target size -- treat those as separate, deliberate follow-up
steps once this plan's tasks are all merged, not something to run
unsupervised as part of implementation.
