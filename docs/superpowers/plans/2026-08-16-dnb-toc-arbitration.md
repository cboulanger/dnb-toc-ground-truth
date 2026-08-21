# dnb-toc-only Arbitration Tool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a Claude Code session resolve dnb-toc-only books that failed the two-vision-model agreement gate, instead of discarding them, by surfacing each book's already-cached per-model extractions as a diff report and recording the final human/Claude verdict.

**Architecture:** A new pure `diff_toc_entries` helper factors the matched/singleton computation `gate_book` already does out into something reusable. The three cache read/write helpers move from the `generate_dnb_toc_ground_truth.py` script into the shared `dnb_toc_vision.py` module so a second script can use them without importing one script from another. A new `arbitrate_dnb_toc.py` script reports (never decides) which books need a look, using both of the above; a Claude Code session writes the final `.expected.json` directly or records a permanent rejection.

**Tech Stack:** Python, `unittest`, existing `evaluation/dnb_toc_matching.py` / `evaluation/dnb_toc_vision.py` / `evaluation/harness.py` modules.

**Design doc:** `docs/superpowers/specs/2026-08-16-dnb-toc-arbitration-design.md`

---

### Task 1: `diff_toc_entries` in `evaluation/dnb_toc_matching.py`

**Files:**
- Modify: `evaluation/dnb_toc_matching.py:85-125` (the `gate_book` function)
- Test: `tests/test_dnb_toc_matching.py`

`gate_book` already computes, internally, exactly the breakdown an
arbitrator needs to see: which entries matched between the two sides,
and which ones only one side found. This task pulls that computation out
into its own function so a second caller (the arbitration script, Task
3) can get the same breakdown without re-deriving it, and makes
`gate_book` itself call the new function instead of duplicating the
logic.

- [ ] **Step 1: Write the failing tests**

Add this new test class to `tests/test_dnb_toc_matching.py`, after the
existing `TestAlignTocEntries` class and before `class TestGateBook`:

```python
class TestDiffTocEntries(unittest.TestCase):
    def test_full_agreement_has_no_singletons(self):
        a = [_entry("Einleitung", 9), _entry("Schluss", 40)]
        b = [_entry("Einleitung", 9), _entry("Schluss", 40)]
        matched, only_a, only_b = diff_toc_entries(a, b)
        self.assertEqual(len(matched), 2)
        self.assertEqual(only_a, [])
        self.assertEqual(only_b, [])

    def test_partial_agreement_separates_singletons_per_side(self):
        a = [_entry("Einleitung", 9), _entry("Only in A", 20)]
        b = [_entry("Einleitung", 9), _entry("Only in B", 30)]
        matched, only_a, only_b = diff_toc_entries(a, b)
        self.assertEqual(len(matched), 1)
        self.assertEqual([e.title for e in only_a], ["Only in A"])
        self.assertEqual([e.title for e in only_b], ["Only in B"])

    def test_complete_disagreement_puts_everything_in_singletons(self):
        a = [_entry("A", 1)]
        b = [_entry("B", 2)]
        matched, only_a, only_b = diff_toc_entries(a, b)
        self.assertEqual(matched, [])
        self.assertEqual([e.title for e in only_a], ["A"])
        self.assertEqual([e.title for e in only_b], ["B"])

    def test_matched_pairs_hold_entry_objects_not_indices(self):
        a = [_entry("Einleitung", 9, authors=("A Author",))]
        b = [_entry("Einleitung", 9, authors=("B Author",))]
        matched, _, _ = diff_toc_entries(a, b)
        self.assertEqual(len(matched), 1)
        entry_a, entry_b = matched[0]
        self.assertEqual(entry_a.authors, ("A Author",))
        self.assertEqual(entry_b.authors, ("B Author",))
```

Update the module's import line near the top of the file from:

```python
from evaluation.dnb_toc_matching import align_toc_entries, gate_book, toc_entry_to_gt_dict
```

to:

```python
from evaluation.dnb_toc_matching import align_toc_entries, diff_toc_entries, gate_book, toc_entry_to_gt_dict
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `uv run pytest tests/test_dnb_toc_matching.py::TestDiffTocEntries -v`
Expected: FAIL with `ImportError` (`diff_toc_entries` doesn't exist yet).

- [ ] **Step 3: Add `diff_toc_entries` and refactor `gate_book` to use it**

In `evaluation/dnb_toc_matching.py`, insert this new function directly
above `def gate_book(`:

```python
def diff_toc_entries(
    a: list[TocEntry], b: list[TocEntry],
) -> tuple[list[tuple[TocEntry, TocEntry]], list[TocEntry], list[TocEntry]]:
    """Aligns a and b via align_toc_entries and returns (matched_pairs,
    only_in_a, only_in_b) -- matched_pairs holds the actual TocEntry
    objects (not indices) from each side for each matched line,
    only_in_a/only_in_b hold every entry from that side with no match on
    the other. Same underlying alignment gate_book uses to decide
    pass/fail; this exposes the full breakdown for a human (or Claude,
    arbitrating a below-threshold book) to review the actual
    disagreement -- see
    docs/superpowers/specs/2026-08-16-dnb-toc-arbitration-design.md
    section 4.1."""
    pairs = align_toc_entries(a, b)
    matched_a = {i for i, _ in pairs}
    matched_b = {j for _, j in pairs}
    matched_pairs = [(a[i], b[j]) for i, j in pairs]
    only_in_a = [entry for i, entry in enumerate(a) if i not in matched_a]
    only_in_b = [entry for j, entry in enumerate(b) if j not in matched_b]
    return matched_pairs, only_in_a, only_in_b
```

Then replace the body of `gate_book` (keep its docstring unchanged)
from:

```python
    if not a and not b:
        return False, []
    pairs = align_toc_entries(a, b)
    agreement_rate = len(pairs) / max(len(a), len(b))
    if agreement_rate < threshold:
        return False, []
    matched_a = {i for i, _ in pairs}
    matched_b = {j for _, j in pairs}
    merged = [
        replace(a[i], authors=a[i].authors or b[j].authors)
        for i, j in pairs
    ]
    merged += [entry for i, entry in enumerate(a) if i not in matched_a]
    merged += [entry for j, entry in enumerate(b) if j not in matched_b]
    merged.sort(key=lambda e: (e.printed_page_number == -1, e.printed_page_number))
    return True, merged
```

to:

```python
    if not a and not b:
        return False, []
    matched_pairs, only_in_a, only_in_b = diff_toc_entries(a, b)
    agreement_rate = len(matched_pairs) / max(len(a), len(b))
    if agreement_rate < threshold:
        return False, []
    merged = [
        replace(entry_a, authors=entry_a.authors or entry_b.authors)
        for entry_a, entry_b in matched_pairs
    ]
    merged += only_in_a
    merged += only_in_b
    merged.sort(key=lambda e: (e.printed_page_number == -1, e.printed_page_number))
    return True, merged
```

- [ ] **Step 4: Run the full test file to verify everything passes**

Run: `uv run pytest tests/test_dnb_toc_matching.py -v`
Expected: PASS, all tests including the existing `TestGateBook` class
(unchanged behavior) and the new `TestDiffTocEntries` class.

- [ ] **Step 5: Commit**

```bash
git add evaluation/dnb_toc_matching.py tests/test_dnb_toc_matching.py
git commit -m "refactor: extract diff_toc_entries from gate_book's matched/singleton logic"
```

---

### Task 2: Move the LLM cache helpers into `evaluation/dnb_toc_vision.py`

**Files:**
- Modify: `evaluation/dnb_toc_vision.py`
- Modify: `evaluation/scripts/generate_dnb_toc_ground_truth.py`
- Modify: `tests/test_dnb_toc_vision.py`
- Modify: `tests/test_generate_dnb_toc_ground_truth.py`

The cache read/write helpers cache `vision_extract_toc_entries`'s own
results, so they belong next to that function in the shared module, not
inside the one script that happens to call it today. Task 3's new
script needs to read this same cache without importing
`generate_dnb_toc_ground_truth.py` (a script, not a library module).

This is a pure move + rename (drop the leading underscore, since these
become public module functions used by two different scripts) -- no
behavior change.

- [ ] **Step 1: Move the cache helpers into `dnb_toc_vision.py`**

In `evaluation/dnb_toc_vision.py`, change the import block from:

```python
import base64
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from chapter_segmentation._llm_json import parse_json_array
from chapter_segmentation.segmentation import TocEntry, _toc_items_to_entries
```

to:

```python
import base64
import json
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

from pypdf import PdfReader

from chapter_segmentation._llm_json import parse_json_array
from chapter_segmentation.segmentation import TocEntry, _toc_items_to_entries
```

Then insert these three functions directly after the import block,
before the `_VISION_TOC_EXTRACTION_PROMPT = """\` line:

```python
def cache_path(cache_directory: Path, key: str, model: str) -> Path:
    return cache_directory / f"{key}.{model}.json"


def load_cached_llm_entries(cache_directory: Path, key: str, model: str) -> Optional[list[TocEntry]]:
    path = cache_path(cache_directory, key, model)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        TocEntry(
            title=e["title"], printed_page_number=e["printed_page_number"],
            source_page_index=e["source_page_index"], authors=tuple(e["authors"]),
            printed_roman=e["printed_roman"],
        )
        for e in data["entries"]
    ]


def write_cached_llm_entries(cache_directory: Path, key: str, model: str, entries: list[TocEntry]) -> None:
    cache_directory.mkdir(parents=True, exist_ok=True)
    path = cache_path(cache_directory, key, model)
    data = {
        "generated_at": time.time(),
        "entries": [
            {
                "title": e.title, "printed_page_number": e.printed_page_number,
                "source_page_index": e.source_page_index, "authors": list(e.authors),
                "printed_roman": e.printed_roman,
            }
            for e in entries
        ],
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


```

- [ ] **Step 2: Remove the old copies from `generate_dnb_toc_ground_truth.py` and update its imports/call sites**

Change the import block from:

```python
from chapter_segmentation.segmentation import TocEntry
from evaluation.dnb_toc_matching import gate_book, toc_entry_to_gt_dict
from evaluation.dnb_toc_vision import vision_extract_toc_entries
from evaluation.harness import corpus_dir, llm_cache_dir, load_manifest_books
from evaluation.kisski import DEFAULT_KISSKI_BASE_URL, fetch_kisski_models
from evaluation.scripts.select_dnb_toc_eval_sample import manifest_key
```

to:

```python
from chapter_segmentation.segmentation import TocEntry
from evaluation.dnb_toc_matching import gate_book, toc_entry_to_gt_dict
from evaluation.dnb_toc_vision import load_cached_llm_entries, vision_extract_toc_entries, write_cached_llm_entries
from evaluation.harness import corpus_dir, llm_cache_dir, load_manifest_books
from evaluation.kisski import DEFAULT_KISSKI_BASE_URL, fetch_kisski_models
from evaluation.scripts.select_dnb_toc_eval_sample import manifest_key
```

Delete these three function definitions entirely (they now live in
`dnb_toc_vision.py`):

```python
def _cache_path(cache_directory: Path, key: str, model: str) -> Path:
    return cache_directory / f"{key}.{model}.json"


def _load_cached_llm_entries(cache_directory: Path, key: str, model: str) -> Optional[list[TocEntry]]:
    path = _cache_path(cache_directory, key, model)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        TocEntry(
            title=e["title"], printed_page_number=e["printed_page_number"],
            source_page_index=e["source_page_index"], authors=tuple(e["authors"]),
            printed_roman=e["printed_roman"],
        )
        for e in data["entries"]
    ]


def _write_cached_llm_entries(cache_directory: Path, key: str, model: str, entries: list[TocEntry]) -> None:
    cache_directory.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_directory, key, model)
    data = {
        "generated_at": time.time(),
        "entries": [
            {
                "title": e.title, "printed_page_number": e.printed_page_number,
                "source_page_index": e.source_page_index, "authors": list(e.authors),
                "printed_roman": e.printed_roman,
            }
            for e in entries
        ],
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)
```

In `_run_book` (still in `generate_dnb_toc_ground_truth.py`), update the
two call sites: change `cached = _load_cached_llm_entries(cache_directory, key, model)`
to `cached = load_cached_llm_entries(cache_directory, key, model)`, and
change `_write_cached_llm_entries(cache_directory, key, model, entries)`
to `write_cached_llm_entries(cache_directory, key, model, entries)`.

Also remove the now-unused `import time` from the top of the file --
after this change, `time.` no longer appears anywhere else in
`generate_dnb_toc_ground_truth.py` (it was only used inside the
now-deleted `_write_cached_llm_entries`). Keep `import json`: it's still
used elsewhere in the file (the `.expected.json` writer in
`_run_book_entries`, the eval-tier-ids loader, and `_spot_check`). Keep
`from typing import Optional` too -- `_call_with_retry`'s
`last_exc: Optional[Exception]` still uses it.

- [ ] **Step 3: Move the cache round-trip tests to `tests/test_dnb_toc_vision.py`**

In `tests/test_dnb_toc_vision.py`, change the import block from:

```python
from evaluation.dnb_toc_vision import (
    _MAX_VISION_PAGES,
    render_pages_to_images,
    vision_extract_toc_entries,
)
```

to:

```python
from chapter_segmentation.segmentation import TocEntry
from evaluation.dnb_toc_vision import (
    _MAX_VISION_PAGES,
    load_cached_llm_entries,
    render_pages_to_images,
    vision_extract_toc_entries,
    write_cached_llm_entries,
)
```

Add this class right after the `_make_pdf` function and before `class
TestRenderPagesToImages`:

```python
class TestLlmCacheRoundTrip(unittest.TestCase):
    def test_round_trips_entries_through_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            entries = [
                TocEntry(title="Einleitung", printed_page_number=9, source_page_index=0, authors=("Jane Author",)),
                TocEntry(title="Bibliographie", printed_page_number=-1, source_page_index=1),
            ]
            self.assertIsNone(load_cached_llm_entries(cache_dir, "book1", "model-a"))
            write_cached_llm_entries(cache_dir, "book1", "model-a", entries)
            loaded = load_cached_llm_entries(cache_dir, "book1", "model-a")
            self.assertEqual(loaded, entries)

    def test_round_trip_preserves_printed_roman(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            entries = [
                TocEntry(title="Vorwort", printed_page_number=7, source_page_index=0, printed_roman=True),
            ]
            write_cached_llm_entries(cache_dir, "book2", "model-a", entries)
            loaded = load_cached_llm_entries(cache_dir, "book2", "model-a")
            self.assertEqual(loaded, entries)
            self.assertTrue(loaded[0].printed_roman)

    def test_different_models_get_independent_cache_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            entries_a = [TocEntry(title="From model A", printed_page_number=1, source_page_index=0)]
            entries_b = [TocEntry(title="From model B", printed_page_number=1, source_page_index=0)]
            write_cached_llm_entries(cache_dir, "book3", "model-a", entries_a)
            write_cached_llm_entries(cache_dir, "book3", "model-b", entries_b)
            self.assertEqual(load_cached_llm_entries(cache_dir, "book3", "model-a"), entries_a)
            self.assertEqual(load_cached_llm_entries(cache_dir, "book3", "model-b"), entries_b)
```

- [ ] **Step 4: Remove the old cache tests from `tests/test_generate_dnb_toc_ground_truth.py` and update its remaining usages**

Change the import block from:

```python
from chapter_segmentation.segmentation import TocEntry
from evaluation.kisski import KisskiModel
from evaluation.scripts.generate_dnb_toc_ground_truth import (
    _call_with_retry,
    _load_cached_llm_entries,
    _run_book,
    _run_book_entries,
    _select_best_models,
    _write_cached_llm_entries,
)
```

to:

```python
from chapter_segmentation.segmentation import TocEntry
from evaluation.dnb_toc_vision import load_cached_llm_entries, write_cached_llm_entries
from evaluation.kisski import KisskiModel
from evaluation.scripts.generate_dnb_toc_ground_truth import (
    _call_with_retry,
    _run_book,
    _run_book_entries,
    _select_best_models,
)
```

Delete the entire `class TestLlmCacheRoundTrip(unittest.TestCase):`
block (it moved to `tests/test_dnb_toc_vision.py` in Step 3).

In the remaining test methods, replace every call to
`_write_cached_llm_entries(...)` with `write_cached_llm_entries(...)`
(two occurrences, both inside `class TestRunBook`), and every call to
`_load_cached_llm_entries(...)` with `load_cached_llm_entries(...)`
(two occurrences, inside
`test_one_model_failing_preserves_the_others_cache_entry`).

- [ ] **Step 5: Run both test files to verify everything passes**

Run: `uv run pytest tests/test_dnb_toc_vision.py tests/test_generate_dnb_toc_ground_truth.py -v`
Expected: PASS, all tests.

- [ ] **Step 6: Commit**

```bash
git add evaluation/dnb_toc_vision.py evaluation/scripts/generate_dnb_toc_ground_truth.py tests/test_dnb_toc_vision.py tests/test_generate_dnb_toc_ground_truth.py
git commit -m "refactor: move dnb-toc-only LLM cache helpers into dnb_toc_vision.py"
```

---

### Task 3: `evaluation/scripts/arbitrate_dnb_toc.py`

**Files:**
- Create: `evaluation/scripts/arbitrate_dnb_toc.py`
- Test: `tests/test_arbitrate_dnb_toc.py`

The reporting/rejection-recording tool itself. Depends on Task 1's
`diff_toc_entries` and Task 2's `load_cached_llm_entries` living in
`dnb_toc_vision.py`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_arbitrate_dnb_toc.py`:

```python
"""Unit tests for evaluation/scripts/arbitrate_dnb_toc.py's pure logic --
see design spec docs/superpowers/specs/2026-08-16-dnb-toc-arbitration-design.md.
No real corpus paths, no network -- all fixtures live in tempdirs."""

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

from chapter_segmentation.segmentation import TocEntry
from evaluation.dnb_toc_vision import write_cached_llm_entries
from evaluation.scripts.arbitrate_dnb_toc import (
    _cached_models_for_book,
    books_needing_arbitration,
    format_book_report,
    reject_book,
)


def _entry(title: str, page: int) -> TocEntry:
    return TocEntry(title=title, printed_page_number=page, source_page_index=0)


class TestBooksNeedingArbitration(unittest.TestCase):
    def test_includes_a_book_with_cache_and_no_expected_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            cdir = Path(tmp) / "corpus"
            cache_directory = Path(tmp) / "cache"
            cdir.mkdir()
            write_cached_llm_entries(cache_directory, "book1", "model-a", [_entry("X", 1)])

            self.assertEqual(books_needing_arbitration(cdir, cache_directory), ["book1"])

    def test_excludes_a_book_that_already_passed_the_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            cdir = Path(tmp) / "corpus"
            cache_directory = Path(tmp) / "cache"
            cdir.mkdir()
            write_cached_llm_entries(cache_directory, "book2", "model-a", [_entry("X", 1)])
            (cdir / "book2.expected.json").write_text('{"entries": [], "verified": false}', encoding="utf-8")

            self.assertEqual(books_needing_arbitration(cdir, cache_directory), [])

    def test_excludes_an_already_rejected_book(self):
        with tempfile.TemporaryDirectory() as tmp:
            cdir = Path(tmp) / "corpus"
            cache_directory = Path(tmp) / "cache"
            cdir.mkdir()
            write_cached_llm_entries(cache_directory, "book3", "model-a", [_entry("X", 1)])
            (cdir / "arbitration-rejected.json").write_text(
                json.dumps({"rejected": [{"key": "book3", "reason": "unrecoverable", "rejected_at": "2026-08-16"}]}),
                encoding="utf-8",
            )

            self.assertEqual(books_needing_arbitration(cdir, cache_directory), [])


class TestCachedModelsForBook(unittest.TestCase):
    def test_loads_every_cached_model_for_a_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_directory = Path(tmp)
            write_cached_llm_entries(cache_directory, "book1", "model-a", [_entry("X", 1)])
            write_cached_llm_entries(cache_directory, "book1", "model-b", [_entry("Y", 2)])
            write_cached_llm_entries(cache_directory, "book2", "model-a", [_entry("Z", 3)])

            result = _cached_models_for_book(cache_directory, "book1")

            self.assertEqual(set(result), {"model-a", "model-b"})
            self.assertEqual(result["model-a"][0].title, "X")

    def test_handles_a_model_id_that_itself_contains_a_dot(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_directory = Path(tmp)
            write_cached_llm_entries(cache_directory, "book1", "qwen3.6-27b", [_entry("X", 1)])

            result = _cached_models_for_book(cache_directory, "book1")

            self.assertEqual(set(result), {"qwen3.6-27b"})


class TestFormatBookReport(unittest.TestCase):
    def test_two_model_disagreement_lists_unmatched_entries_from_each_side(self):
        report = format_book_report(
            "book1", "Some Title", Path("/tmp/book1.pdf"),
            {
                "model-a": [_entry("Einleitung", 9), _entry("Only in A", 20)],
                "model-b": [_entry("Einleitung", 9), _entry("Only in B", 30)],
            },
        )
        self.assertIn("Only in A", report)
        self.assertIn("Only in B", report)
        self.assertIn("model-a", report)
        self.assertIn("model-b", report)
        self.assertIn("Some Title", report)

    def test_single_surviving_model_lists_its_entries_with_a_note(self):
        report = format_book_report(
            "book2", "Some Title", Path("/tmp/book2.pdf"),
            {"model-a": [_entry("Einleitung", 9)]},
        )
        self.assertIn("Only model-a returned usable output", report)
        self.assertIn("Einleitung", report)


class TestRejectBook(unittest.TestCase):
    def test_creates_the_file_on_first_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            cdir = Path(tmp)
            result = reject_book(cdir, "book1", "unrecoverable", today=lambda: date(2026, 8, 16))
            self.assertEqual(result, 0)
            data = json.loads((cdir / "arbitration-rejected.json").read_text(encoding="utf-8"))
            self.assertEqual(data["rejected"], [{"key": "book1", "reason": "unrecoverable", "rejected_at": "2026-08-16"}])

    def test_appends_to_an_existing_rejection_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            cdir = Path(tmp)
            reject_book(cdir, "book1", "reason one", today=lambda: date(2026, 8, 16))
            reject_book(cdir, "book2", "reason two", today=lambda: date(2026, 8, 16))
            data = json.loads((cdir / "arbitration-rejected.json").read_text(encoding="utf-8"))
            self.assertEqual([entry["key"] for entry in data["rejected"]], ["book1", "book2"])

    def test_errors_on_a_duplicate_key_without_overwriting(self):
        with tempfile.TemporaryDirectory() as tmp:
            cdir = Path(tmp)
            reject_book(cdir, "book1", "original reason", today=lambda: date(2026, 8, 16))
            result = reject_book(cdir, "book1", "different reason", today=lambda: date(2026, 8, 17))
            self.assertEqual(result, 1)
            data = json.loads((cdir / "arbitration-rejected.json").read_text(encoding="utf-8"))
            self.assertEqual(len(data["rejected"]), 1)
            self.assertEqual(data["rejected"][0]["reason"], "original reason")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_arbitrate_dnb_toc.py -v`
Expected: FAIL with `ModuleNotFoundError` (`evaluation.scripts.arbitrate_dnb_toc` doesn't exist yet).

- [ ] **Step 3: Create `evaluation/scripts/arbitrate_dnb_toc.py`**

```python
"""Surfaces dnb-toc-only books whose two vision-model TOC extractions
didn't clear generate_dnb_toc_ground_truth.py's agreement gate, so a
Claude Code session can arbitrate the conflict directly -- see design
spec docs/superpowers/specs/2026-08-16-dnb-toc-arbitration-design.md.
This script only REPORTS and records rejections; it never decides. The
arbitrator reads a book's report, opens the PDF's actual TOC pages via
the Read tool when the text alone doesn't settle it, then either writes
evaluation/corpus/dnb-toc-only/<key>.expected.json directly (same schema
as a passing book, "verified": true) or runs this script's `reject`
subcommand to permanently record the book as unrecoverable.

    uv run python evaluation/scripts/arbitrate_dnb_toc.py
    uv run python evaluation/scripts/arbitrate_dnb_toc.py reject 9783515114868 "both models hallucinate on this scan"
"""

import argparse
import json
from datetime import date
from pathlib import Path

from chapter_segmentation.segmentation import TocEntry
from evaluation.dnb_toc_matching import diff_toc_entries
from evaluation.dnb_toc_vision import load_cached_llm_entries
from evaluation.harness import corpus_dir, llm_cache_dir, load_manifest_books
from evaluation.scripts.select_dnb_toc_eval_sample import manifest_key

_CORPUS_NAME = "dnb-toc-only"


def _rejected_path(cdir: Path) -> Path:
    return cdir / "arbitration-rejected.json"


def _load_rejected_keys(cdir: Path) -> set[str]:
    path = _rejected_path(cdir)
    if not path.exists():
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    return {entry["key"] for entry in data["rejected"]}


def _cached_book_keys(cache_directory: Path) -> list[str]:
    """Every distinct book key with at least one <key>.<model>.json file
    in cache_directory, sorted for stable output."""
    return sorted({p.name.split(".", 1)[0] for p in cache_directory.glob("*.json")})


def _cached_models_for_book(cache_directory: Path, key: str) -> dict[str, list[TocEntry]]:
    """Every model's cached entries for one book key, keyed by model id
    (the cache filename's middle segment, <key>.<model>.json -- sliced
    rather than split on ".", since a model id can itself contain a dot,
    e.g. "qwen3.6-27b")."""
    result: dict[str, list[TocEntry]] = {}
    for path in sorted(cache_directory.glob(f"{key}.*.json")):
        model = path.name[len(key) + 1: -len(".json")]
        entries = load_cached_llm_entries(cache_directory, key, model)
        if entries is not None:
            result[model] = entries
    return result


def books_needing_arbitration(cdir: Path, cache_directory: Path) -> list[str]:
    """Book keys with cached model output, no .expected.json yet, and not
    already permanently rejected."""
    rejected = _load_rejected_keys(cdir)
    needing = []
    for key in _cached_book_keys(cache_directory):
        if key in rejected:
            continue
        if (cdir / f"{key}.expected.json").exists():
            continue
        needing.append(key)
    return needing


def _format_entry(entry: TocEntry) -> str:
    page = entry.printed_page_number if entry.printed_page_number != -1 else "?"
    return f"    p.{page!s:>4}  {entry.title}"


def format_book_report(key: str, title: str, pdf_path: Path, models_to_entries: dict[str, list[TocEntry]]) -> str:
    """Human-readable diff for one book -- the actual disagreement, ready
    for Claude (or a human) to arbitrate. Handles the normal two-model
    case, the single-surviving-model case (the other model's response
    was empty/malformed), and defensively falls back to a plain per-model
    listing for any other count."""
    lines = [f"=== {key} -- {title} ===", f"PDF: {pdf_path}"]
    model_names = sorted(models_to_entries)
    if len(model_names) == 1:
        model = model_names[0]
        entries = models_to_entries[model]
        lines.append(f"Only {model} returned usable output ({len(entries)} entries) -- verify directly against the page images:")
        for entry in entries:
            lines.append(_format_entry(entry))
        return "\n".join(lines)
    if len(model_names) != 2:
        lines.append(f"Expected 1 or 2 cached models, found {len(model_names)}: {model_names} -- review each list directly:")
        for model in model_names:
            lines.append(f"  -- {model} ({len(models_to_entries[model])} entries) --")
            for entry in models_to_entries[model]:
                lines.append(_format_entry(entry))
        return "\n".join(lines)
    model_a, model_b = model_names
    entries_a, entries_b = models_to_entries[model_a], models_to_entries[model_b]
    matched, only_a, only_b = diff_toc_entries(entries_a, entries_b)
    rate = len(matched) / max(len(entries_a), len(entries_b))
    lines.append(f"{model_a}: {len(entries_a)} entries, {model_b}: {len(entries_b)} entries -- {len(matched)} matched, rate={rate:.2f}")
    if only_a:
        lines.append(f"  Only in {model_a}:")
        for entry in only_a:
            lines.append(_format_entry(entry))
    if only_b:
        lines.append(f"  Only in {model_b}:")
        for entry in only_b:
            lines.append(_format_entry(entry))
    return "\n".join(lines)


def _list(cdir: Path, cache_directory: Path) -> int:
    needing = books_needing_arbitration(cdir, cache_directory)
    if not needing:
        print("No books currently need arbitration.")
        return 0
    titles = {manifest_key(book): book.get("title", "") for book in load_manifest_books(_CORPUS_NAME)}
    for key in needing:
        models_to_entries = _cached_models_for_book(cache_directory, key)
        print(format_book_report(key, titles.get(key, ""), cdir / f"{key}.pdf", models_to_entries))
        print()
    return 0


def reject_book(cdir: Path, key: str, reason: str, today=date.today) -> int:
    """Permanently records key as unrecoverable so future arbitration
    passes never resurface it. Errors (returns 1) rather than silently
    overwriting if key is already rejected."""
    path = _rejected_path(cdir)
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"rejected": []}
    if any(entry["key"] == key for entry in data["rejected"]):
        print(f"{key} is already marked rejected -- not overwriting.")
        return 1
    data["rejected"].append({"key": key, "reason": reason, "rejected_at": today().isoformat()})
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("list", help="List books needing arbitration (default)")
    reject_parser = subparsers.add_parser("reject", help="Permanently mark a book as unrecoverable")
    reject_parser.add_argument("key")
    reject_parser.add_argument("reason")
    args = parser.parse_args()

    cdir = corpus_dir(_CORPUS_NAME)
    if args.command == "reject":
        return reject_book(cdir, args.key, args.reason)
    return _list(cdir, llm_cache_dir(_CORPUS_NAME))


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_arbitrate_dnb_toc.py -v`
Expected: PASS, all tests.

- [ ] **Step 5: Commit**

```bash
git add evaluation/scripts/arbitrate_dnb_toc.py tests/test_arbitrate_dnb_toc.py
git commit -m "feat: add arbitrate_dnb_toc.py to surface below-gate books for review"
```

---

### Task 4: Document the arbitration workflow in `evaluation/CLAUDE.md`

**Files:**
- Modify: `evaluation/CLAUDE.md`

- [ ] **Step 1: Insert the new section**

In `evaluation/CLAUDE.md`, find this exact text (the end of "Step 5: TOC
ground truth", right before "## Known failure modes"):

```markdown
and confirming the exact page range by eye every time.

## Known failure modes (found the hard way while building this evaluation set)
```

Replace it with:

```markdown
and confirming the exact page range by eye every time.

## Arbitrating below-gate dnb-toc-only books

`evaluation/scripts/generate_dnb_toc_ground_truth.py`'s two-vision-model
gate discards a book outright when the two models disagree too much
(below 0.90 agreement) or one of them fails outright -- but it never
deletes either model's cached raw extraction
(`evaluation/corpus/dnb-toc-only/llm-cache/<key>.<model>.json`). Rather
than re-running the whole book from scratch or leaving it discarded,
walk through the following after a generation run leaves books below
the gate (design spec
docs/superpowers/specs/2026-08-16-dnb-toc-arbitration-design.md):

1. List every book still needing a decision:

   ```bash
   uv run python evaluation/scripts/arbitrate_dnb_toc.py
   ```

   This prints, per book: its title and PDF path, both models' entry
   counts and agreement rate, and every entry each side found that the
   other didn't (or, if only one model produced usable output, that
   model's full list with a note to verify it directly).

2. For each book, read the printed diff. The disagreement patterns
   found in practice so far (`evaluation/RESULTS.md` § "dnb-toc-only
   ground truth: two-vision-model gate") usually make the right call
   obvious from the text alone: one side dropping real content, one
   side including front/back matter or a part-divider that should have
   been skipped, a two-line title wrongly split into two entries, or a
   deeply nested TOC segmented at different granularities.

3. When the text alone doesn't settle it, open the book's actual TOC
   page images directly: use the `Read` tool on the PDF with a `pages`
   parameter (1-based viewer pages, same convention as Step 3 above).

4. Write the final `evaluation/corpus/dnb-toc-only/<key>.expected.json`
   yourself -- same schema as a passing book
   (`{"entries": [...], "verified": true}`, each entry via
   `evaluation.dnb_toc_matching.toc_entry_to_gt_dict`), but with
   `"verified": true` rather than `false`: unlike the bulk-tier gate's
   own output, this went through direct scrutiny (including the images,
   when needed), the same standard `_spot_check`'s docstring in
   `generate_dnb_toc_ground_truth.py` already treats as
   "independently human-verified" -- so it's also correctly excluded
   from that function's own sampling pool going forward.

5. If a book is genuinely unrecoverable (both models hallucinate, the
   scan itself is too degraded to read even directly), record that
   instead of leaving it to resurface every run:

   ```bash
   uv run python evaluation/scripts/arbitrate_dnb_toc.py reject <key> "<short reason>"
   ```

   This writes to the committed
   `evaluation/corpus/dnb-toc-only/arbitration-rejected.json` -- refuses
   (rather than silently overwriting) if `<key>` is already present, so
   re-running this step is safe.

## Known failure modes (found the hard way while building this evaluation set)
```

- [ ] **Step 2: Commit**

```bash
git add evaluation/CLAUDE.md
git commit -m "docs: document the below-gate arbitration workflow"
```
