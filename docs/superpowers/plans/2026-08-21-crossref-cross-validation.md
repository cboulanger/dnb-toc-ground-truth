# Crossref Cross-Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port chapter-segmentation's Crossref chapter-lookup mechanism into this repo, backfill missing book DOIs, cache each book's Crossref-registered chapter list, and measure agreement between that data and this corpus's own ground truth.

**Architecture:** One new module (`src/dnb_toc_ground_truth/crossref.py`) does a single Crossref API call per ISBN and derives both the book-level DOI and its chapter list from the same response, caching both together. `cli/fetch_corpus.py` calls it in real time for every newly acquired book; a new `cli/backfill_crossref.py` does the same for the existing manifest backlog; a new `cli/evaluate_crossref.py` reuses `matching.diff_toc_entries` (already built to diff two independent `TocEntry` lists) unmodified to compare cached Crossref chapters against each book's `.expected.json`.

**Tech Stack:** Python 3.12, httpx (sync `Client`), `unittest`, `TocEntry` (existing dataclass in `src/dnb_toc_ground_truth/toc_entry.py`).

**Design doc:** `docs/superpowers/specs/2026-08-21-crossref-cross-validation-design.md`

---

## Task 1: `corpus.py` — Crossref cache directory helper

**Files:**
- Modify: `src/dnb_toc_ground_truth/corpus.py`
- Test: `tests/test_corpus.py`

- [ ] **Step 1: Write the failing test**

Open `tests/test_corpus.py` and check its existing style first (it follows the same `patch.object(corpus, "CORPUS_DIR", ...)` pattern used everywhere else in this repo's tests). Add:

```python
def test_crossref_cache_dir(self):
    with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
        self.assertEqual(corpus.crossref_cache_dir(), Path(tmp) / ".crossref-cache")
```

Add `import tempfile` and `from unittest.mock import patch` at the top of the file if not already present (check first — `test_fetch_corpus.py` already imports both this way).

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_corpus.py -k test_crossref_cache_dir -v`
Expected: FAIL with `AttributeError: module 'dnb_toc_ground_truth.corpus' has no attribute 'crossref_cache_dir'`

- [ ] **Step 3: Write minimal implementation**

In `src/dnb_toc_ground_truth/corpus.py`, add after `lobid_cache_dir()`:

```python
def crossref_cache_dir() -> Path:
    return CORPUS_DIR / ".crossref-cache"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_corpus.py -k test_crossref_cache_dir -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/dnb_toc_ground_truth/corpus.py tests/test_corpus.py
git commit -m "feat: add corpus.crossref_cache_dir() helper"
```

---

## Task 2: Repo scaffolding — gitignore and config template

**Files:**
- Modify: `.gitignore`
- Modify: `.config.json.dist`

No tests — these are plain config files.

- [ ] **Step 1: Add the cache directory to `.gitignore`**

Current `.gitignore` ends with:
```
data/corpus/pilot/.layout-cache/
```

Add a new line directly after it:
```
data/corpus/pilot/.crossref-cache/
```

- [ ] **Step 2: Add `contact_email` to `.config.json.dist`**

Current `.config.json.dist`:
```json
{
  "endpoints_file": ".endpoints",
  "use_vision": ["mistralai/Pixtral-12B-2409", "Qwen/Qwen3-Omni-30B-A3B-Instruct"],
  "use_text": [],
  "concurrency": 4,
  "limit": null,
  "gate_threshold": 0.90
}
```

Replace with (adds `contact_email` as the first key, documenting its purpose since every other key here is self-explanatory from its name alone but this one is new and Crossref-specific):

```json
{
  "contact_email": "you@example.org",
  "endpoints_file": ".endpoints",
  "use_vision": ["mistralai/Pixtral-12B-2409", "Qwen/Qwen3-Omni-30B-A3B-Instruct"],
  "use_text": [],
  "concurrency": 4,
  "limit": null,
  "gate_threshold": 0.90
}
```

Also update your own local (gitignored) `.config.json` with a real contact email, since `fetch_corpus.py`/`backfill_crossref.py`/`evaluate_crossref.py` will read it from there once Task 5 lands — check whether `.config.json` already exists in the repo root and add the `"contact_email"` key to it the same way if so.

- [ ] **Step 3: Commit**

```bash
git add .gitignore .config.json.dist
git commit -m "chore: add crossref cache dir to gitignore, contact_email to config template"
```

---

## Task 3: `crossref.py` — ISBN normalization and cache data type

**Files:**
- Create: `src/dnb_toc_ground_truth/crossref.py`
- Test: `tests/test_crossref.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_crossref.py`:

```python
"""Unit tests for src/dnb_toc_ground_truth/crossref.py -- Crossref
book-DOI and chapter-list lookup by ISBN, ported from chapter-
segmentation's src/chapter_segmentation/evidence/crossref_strategy.py --
see design spec
docs/superpowers/specs/2026-08-21-crossref-cross-validation-design.md.
No live network -- httpx.Client is mocked throughout."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import httpx

from dnb_toc_ground_truth.crossref import normalize_isbn


class TestNormalizeIsbn(unittest.TestCase):
    def test_prefers_isbn13(self):
        self.assertEqual(normalize_isbn("978-3-89971-818-8"), "9783899718188")

    def test_falls_back_to_isbn10(self):
        self.assertEqual(normalize_isbn("3-89971-818-6"), "3899718186")

    def test_isbn10_with_trailing_x_uppercased(self):
        self.assertEqual(normalize_isbn("380305027x"), "380305027X")

    def test_picks_first_isbn13_from_semicolon_separated_list(self):
        self.assertEqual(normalize_isbn("9783899718188; 3899718186"), "9783899718188")

    def test_none_for_empty_string(self):
        self.assertIsNone(normalize_isbn(""))

    def test_none_for_garbage(self):
        self.assertIsNone(normalize_isbn("not-an-isbn"))

    def test_none_for_wrong_length(self):
        self.assertIsNone(normalize_isbn("12345"))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_crossref.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dnb_toc_ground_truth.crossref'`

- [ ] **Step 3: Write minimal implementation**

Create `src/dnb_toc_ground_truth/crossref.py`:

```python
"""Crossref book-DOI and chapter-list lookup by ISBN -- ported from
chapter-segmentation's
src/chapter_segmentation/evidence/crossref_strategy.py (async
httpx.AsyncClient, ChapterCandidate output) to this repo's sync
httpx.Client convention (matching cli/fetch_corpus.py's own style) and
TocEntry output, so it composes directly with matching.py's diff logic.
See design spec
docs/superpowers/specs/2026-08-21-crossref-cross-validation-design.md.

One GET .../works?filter=isbn:{isbn} serves both this design's needs at
once -- Crossref returns every work type registered under an ISBN in one
response, not just book-chapter, so the book-level DOI (first item typed
as an actual book, not one of its chapters) and the chapter list (every
book-chapter-typed item) both come from a single call."""

import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx

from dnb_toc_ground_truth.toc_entry import TocEntry

_CROSSREF_BASE_URL = "https://api.crossref.org/works"
_MAX_RETRIES = 3
_DEFAULT_RETRY_DELAY_SECONDS = 1.0

# Preferred book-level Crossref work types, checked in the order Crossref
# returns items (not reordered by preference among these) -- a narrower
# whitelist than "anything not book-chapter", since a stray "book-part" or
# "book-set" item registered under the same ISBN is not itself the book's
# own DOI.
_BOOK_TYPES = ("book", "monograph", "edited-book", "reference-book")


def normalize_isbn(raw: str) -> Optional[str]:
    """Extracts the first ISBN-13 (13 digits after stripping separators)
    from a manifest ISBN field, falling back to the first ISBN-10 (10
    digits, last may be 'X'), or None if nothing usable is present."""
    if not raw:
        return None
    candidates = re.split(r"[;\s]+", raw.strip())
    stripped = [re.sub(r"[-\s]", "", c) for c in candidates if c.strip()]
    for c in stripped:
        if len(c) == 13 and c.isdigit():
            return c
    for c in stripped:
        if len(c) == 10 and c[:-1].isdigit() and (c[-1].isdigit() or c[-1].upper() == "X"):
            return c[:-1] + c[-1].upper()
    return None


@dataclass(frozen=True)
class CrossrefBookData:
    isbn: str
    doi: Optional[str]
    chapters: tuple[TocEntry, ...]
    fetched_at: str
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_crossref.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add src/dnb_toc_ground_truth/crossref.py tests/test_crossref.py
git commit -m "feat: add crossref.normalize_isbn and CrossrefBookData"
```

---

## Task 4: `crossref.py` — fetch, parse, and cache

**Files:**
- Modify: `src/dnb_toc_ground_truth/crossref.py`
- Modify: `tests/test_crossref.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_crossref.py` (below the existing imports, add `from dnb_toc_ground_truth.crossref import fetch_crossref_book` to the import line):

```python
from dnb_toc_ground_truth.crossref import CrossrefBookData, fetch_crossref_book, normalize_isbn


def _json_response(payload: dict, status_code: int = 200) -> Mock:
    response = Mock()
    response.status_code = status_code
    response.json.return_value = payload
    response.headers = {}
    if status_code == 200:
        response.raise_for_status = Mock()
    else:
        response.raise_for_status = Mock(side_effect=httpx.HTTPStatusError("error", request=Mock(), response=response))
    return response


_MIXED_TYPE_RESPONSE = {
    "message": {
        "items": [
            {"type": "book", "DOI": "10.1515/book-doi", "title": ["Some Book"]},
            {
                "type": "book-chapter", "DOI": "10.1515/ch1",
                "title": ["Re:Law."], "subtitle": ["Recht überdenken und neu gestalten"],
                "author": [{"given": "Jane", "family": "Author"}],
                "page": "21-49",
            },
            {
                "type": "book-chapter", "DOI": "10.1515/ch2",
                "title": ["A Second Chapter"], "author": [], "page": "50-70",
            },
            {"type": "book-chapter", "DOI": "10.1515/untitled", "title": []},
        ]
    }
}


class TestFetchCrossrefBook(unittest.TestCase):
    def test_parses_book_doi_and_chapters_from_one_response(self):
        client = Mock()
        client.get.return_value = _json_response(_MIXED_TYPE_RESPONSE)
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            data = fetch_crossref_book("9783899718188", client, "me@example.org", cache_dir)

        self.assertEqual(data.isbn, "9783899718188")
        self.assertEqual(data.doi, "10.1515/book-doi")
        self.assertEqual(len(data.chapters), 2)
        self.assertEqual(data.chapters[0].title, "Re:Law. Recht überdenken und neu gestalten")
        self.assertEqual(data.chapters[0].authors, ("Jane Author",))
        self.assertEqual(data.chapters[0].printed_page_number, "21")
        self.assertFalse(data.chapters[0].skip)
        self.assertEqual(data.chapters[1].title, "A Second Chapter")
        self.assertEqual(data.chapters[1].printed_page_number, "50")

    def test_untitled_chapter_item_is_dropped(self):
        client = Mock()
        client.get.return_value = _json_response(_MIXED_TYPE_RESPONSE)
        with tempfile.TemporaryDirectory() as tmp:
            data = fetch_crossref_book("9783899718188", client, None, Path(tmp))
        self.assertEqual({c.title for c in data.chapters}, {"Re:Law. Recht überdenken und neu gestalten", "A Second Chapter"})

    def test_no_book_typed_item_yields_none_doi(self):
        client = Mock()
        response_only_chapters = {"message": {"items": _MIXED_TYPE_RESPONSE["message"]["items"][1:]}}
        client.get.return_value = _json_response(response_only_chapters)
        with tempfile.TemporaryDirectory() as tmp:
            data = fetch_crossref_book("9783899718188", client, None, Path(tmp))
        self.assertIsNone(data.doi)
        self.assertEqual(len(data.chapters), 2)

    def test_writes_and_reads_cache(self):
        client = Mock()
        client.get.return_value = _json_response(_MIXED_TYPE_RESPONSE)
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            fetch_crossref_book("9783899718188", client, None, cache_dir)
            cache_path = cache_dir / "9783899718188.crossref.json"
            self.assertTrue(cache_path.exists())
            cached_payload = json.loads(cache_path.read_text(encoding="utf-8"))
            self.assertEqual(cached_payload["doi"], "10.1515/book-doi")
            self.assertEqual(len(cached_payload["chapters"]), 2)

            client.get.reset_mock()
            second = fetch_crossref_book("9783899718188", client, None, cache_dir)
            client.get.assert_not_called()
            self.assertEqual(second.doi, "10.1515/book-doi")
            self.assertEqual(len(second.chapters), 2)

    def test_force_bypasses_cache(self):
        client = Mock()
        client.get.return_value = _json_response(_MIXED_TYPE_RESPONSE)
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            fetch_crossref_book("9783899718188", client, None, cache_dir)
            client.get.reset_mock()
            fetch_crossref_book("9783899718188", client, None, cache_dir, force=True)
            client.get.assert_called_once()

    def test_network_error_returns_empty_and_does_not_cache(self):
        client = Mock()
        client.get.side_effect = httpx.HTTPError("boom")
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            data = fetch_crossref_book("9783899718188", client, None, cache_dir)
        self.assertIsNone(data.doi)
        self.assertEqual(data.chapters, ())
        self.assertFalse((cache_dir / "9783899718188.crossref.json").exists())

    def test_malformed_response_returns_empty_and_does_not_cache(self):
        client = Mock()
        client.get.return_value = _json_response({"unexpected": "shape"})
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            data = fetch_crossref_book("9783899718188", client, None, cache_dir)
        self.assertIsNone(data.doi)
        self.assertEqual(data.chapters, ())
        self.assertFalse((cache_dir / "9783899718188.crossref.json").exists())

    def test_429_retried_then_succeeds(self):
        client = Mock()
        too_many = _json_response({}, status_code=429)
        too_many.headers = {"Retry-After": "0"}
        ok = _json_response(_MIXED_TYPE_RESPONSE)
        client.get.side_effect = [too_many, ok]
        with tempfile.TemporaryDirectory() as tmp:
            data = fetch_crossref_book("9783899718188", client, None, Path(tmp))
        self.assertEqual(data.doi, "10.1515/book-doi")
        self.assertEqual(client.get.call_count, 2)

    def test_exhausted_429_retries_returns_empty_and_does_not_cache(self):
        client = Mock()
        too_many = _json_response({}, status_code=429)
        too_many.headers = {"Retry-After": "0"}
        client.get.return_value = too_many
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            data = fetch_crossref_book("9783899718188", client, None, cache_dir)
        self.assertIsNone(data.doi)
        self.assertEqual(data.chapters, ())
        self.assertFalse((cache_dir / "9783899718188.crossref.json").exists())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_crossref.py -v`
Expected: FAIL with `ImportError: cannot import name 'fetch_crossref_book'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/dnb_toc_ground_truth/crossref.py`:

```python
def _cache_path(cache_dir: Path, isbn: str) -> Path:
    return cache_dir / f"{isbn}.crossref.json"


def _load_cache(cache_dir: Path, isbn: str) -> Optional[CrossrefBookData]:
    path = _cache_path(cache_dir, isbn)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        chapters = tuple(
            TocEntry(
                title=c["title"], authors=tuple(c["authors"]),
                printed_page_number=c["printed_page_number"], source_page_index=-1, skip=False,
            )
            for c in data["chapters"]
        )
        return CrossrefBookData(isbn=isbn, doi=data.get("doi"), chapters=chapters, fetched_at=data["fetched_at"])
    except Exception as exc:
        print(f"  [warn] corrupted crossref cache for {isbn}, refetching: {exc}")
        return None


def _save_cache(cache_dir: Path, data: CrossrefBookData) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "isbn": data.isbn,
        "doi": data.doi,
        "fetched_at": data.fetched_at,
        "chapters": [
            {"title": c.title, "authors": list(c.authors), "printed_page_number": c.printed_page_number}
            for c in data.chapters
        ],
    }
    _cache_path(cache_dir, data.isbn).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
    )


def _first_page(page_range: Optional[str]) -> Optional[str]:
    if not page_range:
        return None
    first = page_range.split("-")[0].strip()
    return first or None


def _parse_chapter_item(item: dict) -> Optional[TocEntry]:
    if item.get("type") != "book-chapter":
        return None
    titles = item.get("title") or []
    if not titles:
        return None
    # Crossref splits a chapter's real printed heading into separate
    # title/subtitle fields -- see crossref_strategy.py's
    # _parse_crossref_item for the full rationale this ports.
    subtitles = item.get("subtitle") or []
    title = f"{titles[0]} {subtitles[0]}" if subtitles else titles[0]
    authors = tuple(
        f"{a.get('given', '')} {a.get('family', '')}".strip()
        for a in item.get("author", []) if a.get("family")
    )
    return TocEntry(
        title=title, authors=authors, printed_page_number=_first_page(item.get("page")),
        source_page_index=-1, skip=False,
    )


def _book_doi(items: list[dict]) -> Optional[str]:
    for item in items:
        if item.get("type") in _BOOK_TYPES:
            return item.get("DOI")
    return None


def _query_crossref(isbn: str, client: httpx.Client, contact_email: Optional[str]) -> Optional[list[dict]]:
    """Returns the raw items list on a confirmed successful response
    (including a confirmed-empty one), or None on any network/HTTP-
    status/JSON-shape failure -- the None/empty-list distinction is what
    lets fetch_crossref_book below decide whether the result is safe to
    cache."""
    params: dict[str, str | int] = {
        "filter": f"isbn:{isbn}",
        "select": "DOI,title,subtitle,author,page,type",
        "rows": 100,
    }
    if contact_email:
        params["mailto"] = contact_email

    response = None
    for _attempt in range(_MAX_RETRIES):
        try:
            response = client.get(_CROSSREF_BASE_URL, params=params, timeout=10.0)
        except httpx.HTTPError as exc:
            print(f"  [warn] network error fetching Crossref data for {isbn}: {exc}")
            return None
        if response.status_code != 429:
            break
        retry_after = response.headers.get("Retry-After")
        delay = float(retry_after) if retry_after and retry_after.isdigit() else _DEFAULT_RETRY_DELAY_SECONDS
        time.sleep(delay)
    else:
        print(f"  [warn] exhausted retries (429) fetching Crossref data for {isbn}")
        return None

    try:
        response.raise_for_status()
        return response.json()["message"]["items"]
    except Exception as exc:
        print(f"  [warn] bad Crossref response for {isbn}: {exc}")
        return None


def fetch_crossref_book(
    isbn: str,
    client: httpx.Client,
    contact_email: Optional[str],
    cache_dir: Path,
    force: bool = False,
) -> CrossrefBookData:
    """The book-level DOI plus its Crossref-registered chapter list,
    fetched from one GET .../works?filter=isbn:{isbn} and cached on disk
    at <cache_dir>/<isbn>.crossref.json. Cached even when doi is None and
    chapters is empty (a confirmed miss, never re-queried on repeat
    runs) -- but NOT cached when the request itself failed (network
    error, exhausted 429 retries, malformed response), so a transient
    failure gets retried on the next run instead of being mistaken for a
    confirmed miss forever."""
    if not force:
        cached = _load_cache(cache_dir, isbn)
        if cached is not None:
            return cached

    items = _query_crossref(isbn, client, contact_email)
    if items is None:
        return CrossrefBookData(isbn=isbn, doi=None, chapters=(), fetched_at="")

    doi = _book_doi(items)
    chapters = tuple(c for item in items if (c := _parse_chapter_item(item)) is not None)
    data = CrossrefBookData(
        isbn=isbn, doi=doi, chapters=chapters, fetched_at=datetime.now(timezone.utc).isoformat(),
    )
    _save_cache(cache_dir, data)
    return data
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_crossref.py -v`
Expected: PASS (16 tests total)

- [ ] **Step 5: Commit**

```bash
git add src/dnb_toc_ground_truth/crossref.py tests/test_crossref.py
git commit -m "feat: fetch, parse, and cache Crossref book DOI + chapter data"
```

---

## Task 5: `fetch_corpus.py` — real-time Crossref hook

**Files:**
- Modify: `cli/fetch_corpus.py`
- Modify: `tests/test_fetch_corpus.py`

- [ ] **Step 1: Write the failing test**

In `tests/test_fetch_corpus.py`, add `crossref` to the existing `from dnb_toc_ground_truth import corpus` line:

```python
from dnb_toc_ground_truth import corpus, crossref
```

Add this test class after `TestAcquireRecord`:

```python
class TestAcquireRecordCrossref(unittest.TestCase):
    """_acquire_record's Crossref hook -- see TestAcquireRecord's own
    docstring for the tempdir-isolation pattern this reuses."""

    def _setup(self, tmp):
        manifest_path = corpus.manifest_path()
        _ensure_manifest_shell(manifest_path)
        return manifest_path

    def test_writes_doi_found_via_crossref(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            manifest_path = self._setup(tmp)
            pdf_response = Mock()
            pdf_response.raise_for_status = Mock()
            pdf_response.content = b"%PDF-1.4 fake toc bytes"
            crossref_response = _json_response(
                {"message": {"items": [{"type": "book", "DOI": "10.1515/book-doi", "title": ["X"]}]}}
            )
            client = Mock()
            client.get.side_effect = [pdf_response, crossref_response]
            seen_keys = set()

            result = _acquire_record(
                _SAMPLE_RECORD, manifest_path, client, 0, seen_keys, contact_email="me@example.org",
            )

            self.assertIsNone(result)
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(data["books"][0]["doi"], "10.1515/book-doi")
            cache_path = corpus.crossref_cache_dir() / "9783899718188.crossref.json"
            self.assertTrue(cache_path.exists())

    def test_crossref_failure_leaves_doi_null_and_still_acquires(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            manifest_path = self._setup(tmp)
            pdf_response = Mock()
            pdf_response.raise_for_status = Mock()
            pdf_response.content = b"%PDF-1.4 fake toc bytes"
            client = Mock()
            client.get.side_effect = [pdf_response, httpx.HTTPError("boom")]
            seen_keys = set()

            result = _acquire_record(_SAMPLE_RECORD, manifest_path, client, 0, seen_keys)

            self.assertIsNone(result)
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertIsNone(data["books"][0]["doi"])

    def test_non_isbn_key_skips_crossref_lookup_entirely(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            manifest_path = self._setup(tmp)
            record = {k: v for k, v in _SAMPLE_RECORD.items() if k != "isbn"}
            pdf_response = Mock()
            pdf_response.raise_for_status = Mock()
            pdf_response.content = b"%PDF-1.4 fake toc bytes"
            client = Mock()
            client.get.return_value = pdf_response
            seen_keys = set()

            result = _acquire_record(record, manifest_path, client, 0, seen_keys)

            self.assertIsNone(result)
            client.get.assert_called_once()  # only the PDF download, no Crossref lookup
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_fetch_corpus.py -k TestAcquireRecordCrossref -v`
Expected: FAIL — `test_writes_doi_found_via_crossref` and `test_crossref_failure_leaves_doi_null_and_still_acquires` fail because `data["books"][0]["doi"]` is `None`/doesn't reflect the lookup yet; `test_non_isbn_key_skips_crossref_lookup_entirely` currently passes already (no behavior change needed for that case) but must keep passing after Step 3.

- [ ] **Step 3: Write minimal implementation**

In `cli/fetch_corpus.py`, add the import (alongside the existing `from dnb_toc_ground_truth import corpus`):

```python
from dnb_toc_ground_truth import corpus, crossref
```

Modify `_acquire_record`'s signature and body — current code:

```python
def _acquire_record(
    record: dict,
    manifest_path: Path,
    client: httpx.Client,
    rate_limit_seconds: float,
    seen_keys: set[str],
) -> Optional[str]:
```

...

```python
    corpus.pdf_dir().mkdir(parents=True, exist_ok=True)
    (corpus.pdf_dir() / filename).write_bytes(response.content)
    corpus.lobid_cache_dir().mkdir(parents=True, exist_ok=True)
    (corpus.lobid_cache_dir() / f"{key}.lobid.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    _append_book(manifest_path, manifest_entry_from_record(record, filename))
    seen_keys.add(key)
    print(f"[fetch] {filename} <- {toc_url}")
    time.sleep(rate_limit_seconds)
    return None
```

Replace with:

```python
def _acquire_record(
    record: dict,
    manifest_path: Path,
    client: httpx.Client,
    rate_limit_seconds: float,
    seen_keys: set[str],
    contact_email: Optional[str] = None,
    crossref_cache_dir: Optional[Path] = None,
) -> Optional[str]:
```

...

```python
    corpus.pdf_dir().mkdir(parents=True, exist_ok=True)
    (corpus.pdf_dir() / filename).write_bytes(response.content)
    corpus.lobid_cache_dir().mkdir(parents=True, exist_ok=True)
    (corpus.lobid_cache_dir() / f"{key}.lobid.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    entry = manifest_entry_from_record(record, filename)
    isbn = crossref.normalize_isbn(key)
    if isbn:
        crossref_data = crossref.fetch_crossref_book(
            isbn, client, contact_email, crossref_cache_dir or corpus.crossref_cache_dir(),
        )
        if crossref_data.doi:
            entry["doi"] = crossref_data.doi
    _append_book(manifest_path, entry)
    seen_keys.add(key)
    print(f"[fetch] {filename} <- {toc_url}")
    time.sleep(rate_limit_seconds)
    return None
```

Now thread `contact_email`/`crossref_cache_dir` through every caller. `_run_isbns_file` currently:

```python
def _run_isbns_file(args: argparse.Namespace, manifest_path: Path, client: httpx.Client) -> None:
    seen_keys = _load_existing_keys(manifest_path)
    acquired = 0
    for isbn in _read_isbns_file(Path(args.isbns_file)):
        if args.limit is not None and acquired >= args.limit:
            break
        try:
            record = _search_by_isbn(isbn, client)
        except httpx.HTTPError as exc:
            print(f"[skip] {isbn}: lookup failed: {exc}")
            continue
        if record is None:
            print(f"[skip] {isbn}: no lobid-resources record found")
            continue
        reason = _acquire_record(record, manifest_path, client, args.rate_limit_seconds, seen_keys)
        if reason is None:
            acquired += 1
        else:
            print(f"[skip] {isbn}: {reason}")
    print(f"Acquired {acquired} new book(s).")
```

Replace with:

```python
def _run_isbns_file(
    args: argparse.Namespace, manifest_path: Path, client: httpx.Client,
    contact_email: Optional[str], crossref_cache_dir: Path,
) -> None:
    seen_keys = _load_existing_keys(manifest_path)
    acquired = 0
    for isbn in _read_isbns_file(Path(args.isbns_file)):
        if args.limit is not None and acquired >= args.limit:
            break
        try:
            record = _search_by_isbn(isbn, client)
        except httpx.HTTPError as exc:
            print(f"[skip] {isbn}: lookup failed: {exc}")
            continue
        if record is None:
            print(f"[skip] {isbn}: no lobid-resources record found")
            continue
        reason = _acquire_record(
            record, manifest_path, client, args.rate_limit_seconds, seen_keys,
            contact_email, crossref_cache_dir,
        )
        if reason is None:
            acquired += 1
        else:
            print(f"[skip] {isbn}: {reason}")
    print(f"Acquired {acquired} new book(s).")
```

`_scan_and_acquire` currently:

```python
def _scan_and_acquire(
    records: Iterator[dict],
    manifest_path: Path,
    client: httpx.Client,
    rate_limit_seconds: float,
    limit: Optional[int],
    seen_keys: set[str],
    acquired_so_far: int,
) -> tuple[int, int]:
    ...
    scanned = 0
    acquired = 0
    for record in records:
        scanned += 1
        if scanned % 100_000 == 0:
            print(f"[scan] {scanned:,} records scanned this attempt, {acquired_so_far + acquired} acquired so far")
        if limit is not None and acquired_so_far + acquired >= limit:
            break
        reason = _acquire_record(record, manifest_path, client, rate_limit_seconds, seen_keys)
```

Replace the signature and the `_acquire_record` call:

```python
def _scan_and_acquire(
    records: Iterator[dict],
    manifest_path: Path,
    client: httpx.Client,
    rate_limit_seconds: float,
    limit: Optional[int],
    seen_keys: set[str],
    acquired_so_far: int,
    contact_email: Optional[str] = None,
    crossref_cache_dir: Optional[Path] = None,
) -> tuple[int, int]:
    ...
    scanned = 0
    acquired = 0
    for record in records:
        scanned += 1
        if scanned % 100_000 == 0:
            print(f"[scan] {scanned:,} records scanned this attempt, {acquired_so_far + acquired} acquired so far")
        if limit is not None and acquired_so_far + acquired >= limit:
            break
        reason = _acquire_record(
            record, manifest_path, client, rate_limit_seconds, seen_keys,
            contact_email, crossref_cache_dir,
        )
```

(Leave the rest of `_scan_and_acquire`'s body unchanged.)

`_run_from_dump` currently:

```python
def _run_from_dump(args: argparse.Namespace, manifest_path: Path, client: httpx.Client) -> None:
    seen_keys = _load_existing_keys(manifest_path)
    acquired = 0
    attempt = 0
    while True:
        try:
            scanned, newly = _scan_and_acquire(
                _iter_dump_records(args.dump_url, client), manifest_path, client,
                args.rate_limit_seconds, args.limit, seen_keys, acquired,
            )
```

Replace with:

```python
def _run_from_dump(
    args: argparse.Namespace, manifest_path: Path, client: httpx.Client,
    contact_email: Optional[str], crossref_cache_dir: Path,
) -> None:
    seen_keys = _load_existing_keys(manifest_path)
    acquired = 0
    attempt = 0
    while True:
        try:
            scanned, newly = _scan_and_acquire(
                _iter_dump_records(args.dump_url, client), manifest_path, client,
                args.rate_limit_seconds, args.limit, seen_keys, acquired,
                contact_email, crossref_cache_dir,
            )
```

(Leave the rest of `_run_from_dump`'s body, including its retry/backoff loop, unchanged.)

Finally, `main()` currently:

```python
    parser.add_argument(
        "--manifest-path", type=Path, default=None,
        help=(
            "Override where manifest.json entries are read from and appended to "
            "(default: data/corpus/pilot/manifest.json). PDFs and "
            ".lobid-cache/ always go to the real corpus directory regardless -- "
            "only the tracked manifest write is redirectable, e.g. to a scratch "
            "copy that gets merged into the real manifest.json once a long run "
            "finishes, without touching the committed file mid-run."
        ),
    )
    args = parser.parse_args()

    corpus.corpus_dir().mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest_path or corpus.manifest_path()
    _ensure_manifest_shell(manifest_path)

    with httpx.Client(
        follow_redirects=True,
        timeout=httpx.Timeout(connect=30.0, read=120.0, write=30.0, pool=30.0),
    ) as client:
        if args.isbns_file:
            _run_isbns_file(args, manifest_path, client)
        else:
            _run_from_dump(args, manifest_path, client)
    return 0
```

Replace with:

```python
    parser.add_argument(
        "--manifest-path", type=Path, default=None,
        help=(
            "Override where manifest.json entries are read from and appended to "
            "(default: data/corpus/pilot/manifest.json). PDFs and "
            ".lobid-cache/ always go to the real corpus directory regardless -- "
            "only the tracked manifest write is redirectable, e.g. to a scratch "
            "copy that gets merged into the real manifest.json once a long run "
            "finishes, without touching the committed file mid-run."
        ),
    )
    parser.add_argument(
        "--contact-email", default=None,
        help="Crossref polite-pool contact email (default: config file's \"contact_email\")",
    )
    parser.add_argument(
        "--crossref-cache-dir", type=Path, default=None,
        help="Override where Crossref DOI/chapter data is cached (default: data/corpus/pilot/.crossref-cache/)",
    )
    parser.add_argument(
        "--config-file", type=Path, default=Path(inference.DEFAULT_CONFIG_FILENAME),
        help=f"Path to the config file (default: {inference.DEFAULT_CONFIG_FILENAME})",
    )
    args = parser.parse_args()

    corpus.corpus_dir().mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest_path or corpus.manifest_path()
    _ensure_manifest_shell(manifest_path)
    config = inference.load_config(args.config_file)
    contact_email = args.contact_email or config.get("contact_email")
    crossref_cache_dir = args.crossref_cache_dir or corpus.crossref_cache_dir()

    with httpx.Client(
        follow_redirects=True,
        timeout=httpx.Timeout(connect=30.0, read=120.0, write=30.0, pool=30.0),
    ) as client:
        if args.isbns_file:
            _run_isbns_file(args, manifest_path, client, contact_email, crossref_cache_dir)
        else:
            _run_from_dump(args, manifest_path, client, contact_email, crossref_cache_dir)
    return 0
```

Add the `inference` import alongside the `corpus, crossref` import line at the top of the file:

```python
from dnb_toc_ground_truth import corpus, crossref, inference
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_fetch_corpus.py -v`
Expected: PASS (all tests, including the pre-existing ones — `TestScanAndAcquire`'s two tests still pass unmodified since the two new parameters default to `None`)

- [ ] **Step 5: Run the full test suite to check nothing else broke**

Run: `uv run pytest -v`
Expected: PASS, all tests

- [ ] **Step 6: Commit**

```bash
git add cli/fetch_corpus.py tests/test_fetch_corpus.py
git commit -m "feat: look up Crossref DOI/chapters in real time when fetch_corpus.py acquires a book"
```

---

## Task 6: `cli/backfill_crossref.py`

**Files:**
- Create: `cli/backfill_crossref.py`
- Test: `tests/test_backfill_crossref.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_backfill_crossref.py`:

```python
"""Unit tests for cli/backfill_crossref.py -- backfills Crossref DOI +
chapter data for existing manifest entries that already have
.expected.json but no doi yet. See design spec
docs/superpowers/specs/2026-08-21-crossref-cross-validation-design.md."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from dnb_toc_ground_truth import corpus

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cli"))

from backfill_crossref import _needs_backfill, backfill


def _json_response(payload: dict) -> Mock:
    response = Mock()
    response.status_code = 200
    response.json.return_value = payload
    response.raise_for_status = Mock()
    response.headers = {}
    return response


class TestNeedsBackfill(unittest.TestCase):
    def test_true_when_expected_json_exists_and_no_doi(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            corpus.ground_truth_dir().mkdir(parents=True)
            corpus.expected_json_path("9783899718188").write_text("{}", encoding="utf-8")
            self.assertTrue(_needs_backfill({"filename": "9783899718188.pdf", "doi": None}))

    def test_false_when_doi_already_present(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            corpus.ground_truth_dir().mkdir(parents=True)
            corpus.expected_json_path("9783899718188").write_text("{}", encoding="utf-8")
            self.assertFalse(_needs_backfill({"filename": "9783899718188.pdf", "doi": "10.1/x"}))

    def test_false_when_no_expected_json(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            corpus.ground_truth_dir().mkdir(parents=True)
            self.assertFalse(_needs_backfill({"filename": "9783899718188.pdf", "doi": None}))


class TestBackfill(unittest.TestCase):
    def test_writes_doi_for_eligible_book_and_caches_chapters(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            corpus.ground_truth_dir().mkdir(parents=True)
            corpus.expected_json_path("9783899718188").write_text('{"entries": []}', encoding="utf-8")
            manifest_path = corpus.manifest_path()
            manifest_path.write_text(
                json.dumps({"toc_only": True, "books": [
                    {"filename": "9783899718188.pdf", "doi": None},
                ]}),
                encoding="utf-8",
            )
            client = Mock()
            client.get.return_value = _json_response({
                "message": {"items": [
                    {"type": "book", "DOI": "10.1515/found", "title": ["X"]},
                    {"type": "book-chapter", "DOI": "10.1515/ch1", "title": ["A Chapter"], "author": [], "page": "1-10"},
                ]}
            })

            checked, found, cached = backfill(manifest_path, client, None, corpus.crossref_cache_dir(), force=False)

            self.assertEqual(checked, 1)
            self.assertEqual(found, 1)
            self.assertEqual(cached, 1)
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(data["books"][0]["doi"], "10.1515/found")

    def test_skips_book_without_expected_json(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            corpus.ground_truth_dir().mkdir(parents=True)
            manifest_path = corpus.manifest_path()
            manifest_path.write_text(
                json.dumps({"toc_only": True, "books": [{"filename": "9783899718188.pdf", "doi": None}]}),
                encoding="utf-8",
            )
            client = Mock()

            checked, found, cached = backfill(manifest_path, client, None, corpus.crossref_cache_dir(), force=False)

            self.assertEqual(checked, 0)
            client.get.assert_not_called()

    def test_skips_book_that_already_has_doi(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            corpus.ground_truth_dir().mkdir(parents=True)
            corpus.expected_json_path("9783899718188").write_text('{"entries": []}', encoding="utf-8")
            manifest_path = corpus.manifest_path()
            manifest_path.write_text(
                json.dumps({"toc_only": True, "books": [
                    {"filename": "9783899718188.pdf", "doi": "10.1/already"},
                ]}),
                encoding="utf-8",
            )
            client = Mock()

            checked, found, cached = backfill(manifest_path, client, None, corpus.crossref_cache_dir(), force=False)

            self.assertEqual(checked, 0)
            client.get.assert_not_called()

    def test_manifest_untouched_when_no_doi_found(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            corpus.ground_truth_dir().mkdir(parents=True)
            corpus.expected_json_path("9783899718188").write_text('{"entries": []}', encoding="utf-8")
            manifest_path = corpus.manifest_path()
            original = json.dumps({"toc_only": True, "books": [
                {"filename": "9783899718188.pdf", "doi": None},
            ]})
            manifest_path.write_text(original, encoding="utf-8")
            client = Mock()
            client.get.return_value = _json_response({"message": {"items": []}})

            checked, found, cached = backfill(manifest_path, client, None, corpus.crossref_cache_dir(), force=False)

            self.assertEqual(found, 0)
            self.assertEqual(cached, 0)
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertIsNone(data["books"][0]["doi"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_backfill_crossref.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backfill_crossref'`

- [ ] **Step 3: Write minimal implementation**

Create `cli/backfill_crossref.py`:

```python
#!/usr/bin/env python3
"""Backfills Crossref book DOI and chapter data for existing manifest.json
entries that already have a .expected.json ground-truth file but no doi
yet -- see design spec
docs/superpowers/specs/2026-08-21-crossref-cross-validation-design.md.

For every such book, looks up its ISBN on Crossref (dnb_toc_ground_truth.
crossref.fetch_crossref_book): if Crossref has a DOI for the book, writes
it into manifest.json, regardless of whether Crossref also has usable
chapter data for it. Chapter data (if any) is cached to
.crossref-cache/<isbn>.crossref.json either way, as a side effect of
fetch_crossref_book itself. Already-cached ISBNs are skipped on repeat
runs unless --force is passed.

Usage:
    uv run python cli/backfill_crossref.py
    uv run python cli/backfill_crossref.py --force
    uv run python cli/backfill_crossref.py --contact-email you@example.org
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import httpx

from dnb_toc_ground_truth import corpus, crossref, inference


def _needs_backfill(book: dict) -> bool:
    if book.get("doi"):
        return False
    key = corpus.manifest_key(book)
    return corpus.expected_json_path(key).exists()


def backfill(
    manifest_path: Path,
    client: httpx.Client,
    contact_email: Optional[str],
    cache_dir: Path,
    force: bool,
) -> tuple[int, int, int]:
    """Returns (checked, dois_found, chapter_lists_cached) -- checked
    counts only books that pass _needs_backfill; dois_found counts those
    where Crossref returned a doi; chapter_lists_cached counts those
    where Crossref returned at least one chapter."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    books = manifest["books"]
    checked = 0
    dois_found = 0
    chapter_lists_cached = 0
    manifest_changed = False

    for book in books:
        if not _needs_backfill(book):
            continue
        key = corpus.manifest_key(book)
        isbn = crossref.normalize_isbn(key)
        if isbn is None:
            print(f"[skip] {key}: not a valid ISBN")
            continue
        checked += 1
        data = crossref.fetch_crossref_book(isbn, client, contact_email, cache_dir, force=force)
        if data.doi:
            book["doi"] = data.doi
            dois_found += 1
            manifest_changed = True
        if data.chapters:
            chapter_lists_cached += 1
        print(f"[{key}] doi={data.doi or 'none'} chapters={len(data.chapters)}")

    if manifest_changed:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return checked, dois_found, chapter_lists_cached


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--force", action="store_true",
        help="Re-query Crossref even for an already-cached ISBN",
    )
    parser.add_argument(
        "--contact-email", default=None,
        help="Crossref polite-pool contact email (default: config file's \"contact_email\")",
    )
    parser.add_argument(
        "--config-file", type=Path, default=Path(inference.DEFAULT_CONFIG_FILENAME),
        help=f"Path to the config file (default: {inference.DEFAULT_CONFIG_FILENAME})",
    )
    args = parser.parse_args()

    config = inference.load_config(args.config_file)
    contact_email = args.contact_email or config.get("contact_email")

    with httpx.Client(follow_redirects=True) as client:
        checked, found, cached = backfill(
            corpus.manifest_path(), client, contact_email, corpus.crossref_cache_dir(), args.force,
        )
    print(f"\n{checked} book(s) checked, {found} DOI(s) found, {cached} chapter list(s) cached.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_backfill_crossref.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add cli/backfill_crossref.py tests/test_backfill_crossref.py
git commit -m "feat: add cli/backfill_crossref.py to backfill DOIs for ground-truthed books"
```

---

## Task 7: `cli/evaluate_crossref.py`

**Files:**
- Create: `cli/evaluate_crossref.py`
- Test: `tests/test_evaluate_crossref.py`

- [ ] **Step 1: Write the failing tests**

First check `matching.py`'s `diff_toc_entries` signature and `TocEntry`'s exact field names again (`title`, `printed_page_number`, `source_page_index`, `authors`, `skip`) — both already read during design; no surprises expected, but confirm before writing code that calls them.

Create `tests/test_evaluate_crossref.py`:

```python
"""Unit tests for cli/evaluate_crossref.py -- measures agreement between
this corpus's own ground truth and cached Crossref chapter data, reusing
matching.diff_toc_entries unmodified. See design spec
docs/superpowers/specs/2026-08-21-crossref-cross-validation-design.md."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dnb_toc_ground_truth import corpus
from dnb_toc_ground_truth.crossref import CrossrefBookData
from dnb_toc_ground_truth.toc_entry import TocEntry

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cli"))

from evaluate_crossref import BookAgreement, evaluate_book, evaluate_corpus, _load_gt_entries


def _write_expected_json(key: str, entries: list[dict]) -> None:
    corpus.ground_truth_dir().mkdir(parents=True, exist_ok=True)
    corpus.expected_json_path(key).write_text(
        json.dumps({"entries": entries, "verified": True, "source": "agent_arbitration"}), encoding="utf-8",
    )


class TestLoadGtEntries(unittest.TestCase):
    def test_builds_toc_entries_from_expected_json(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            _write_expected_json("9783899718188", [
                {"title": "Preface", "authors": [], "printed_page_number": "vii", "skip": True},
                {"title": "1. Introduction", "authors": ["Jane Author"], "printed_page_number": "1", "skip": False},
            ])
            entries = _load_gt_entries("9783899718188")
            self.assertEqual(len(entries), 2)
            self.assertTrue(entries[0].skip)
            self.assertFalse(entries[1].skip)
            self.assertEqual(entries[1].authors, ("Jane Author",))


class TestEvaluateBook(unittest.TestCase):
    def test_filters_skip_entries_before_comparing(self):
        gt_entries = (
            TocEntry(title="Part I", printed_page_number="9", source_page_index=0, skip=True),
            TocEntry(title="1. Introduction", printed_page_number="11", source_page_index=0, skip=False),
        )
        crossref_data = CrossrefBookData(
            isbn="9783899718188", doi="10.1/x", fetched_at="",
            chapters=(TocEntry(title="Introduction", printed_page_number="11", source_page_index=-1, skip=False),),
        )
        result = evaluate_book("9783899718188", gt_entries, crossref_data)
        self.assertEqual(result.matched, 1)
        self.assertEqual(result.only_in_gt, 0)
        self.assertEqual(result.only_in_crossref, 0)
        self.assertEqual(result.agreement_rate, 1.0)

    def test_reports_disagreement(self):
        gt_entries = (
            TocEntry(title="1. Introduction", printed_page_number="11", source_page_index=0, skip=False),
            TocEntry(title="2. Methods", printed_page_number="30", source_page_index=0, skip=False),
        )
        crossref_data = CrossrefBookData(
            isbn="9783899718188", doi="10.1/x", fetched_at="",
            chapters=(TocEntry(title="Introduction", printed_page_number="11", source_page_index=-1, skip=False),),
        )
        result = evaluate_book("9783899718188", gt_entries, crossref_data)
        self.assertEqual(result.matched, 1)
        self.assertEqual(result.only_in_gt, 1)
        self.assertEqual(result.only_in_crossref, 0)
        self.assertEqual(result.agreement_rate, 0.5)


class TestEvaluateCorpus(unittest.TestCase):
    def test_skips_books_with_no_cached_crossref_data(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            manifest_path = corpus.manifest_path()
            manifest_path.write_text(
                json.dumps({"toc_only": True, "books": [{"filename": "9783899718188.pdf", "doi": None}]}),
                encoding="utf-8",
            )
            _write_expected_json("9783899718188", [
                {"title": "1. Introduction", "authors": [], "printed_page_number": "1", "skip": False},
            ])
            results, no_coverage = evaluate_corpus()
            self.assertEqual(results, [])
            self.assertEqual(no_coverage, ["9783899718188"])

    def test_evaluates_book_with_cached_crossref_data(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            manifest_path = corpus.manifest_path()
            manifest_path.write_text(
                json.dumps({"toc_only": True, "books": [{"filename": "9783899718188.pdf", "doi": "10.1/x"}]}),
                encoding="utf-8",
            )
            _write_expected_json("9783899718188", [
                {"title": "1. Introduction", "authors": [], "printed_page_number": "1", "skip": False},
            ])
            corpus.crossref_cache_dir().mkdir(parents=True, exist_ok=True)
            (corpus.crossref_cache_dir() / "9783899718188.crossref.json").write_text(
                json.dumps({
                    "isbn": "9783899718188", "doi": "10.1/x", "fetched_at": "",
                    "chapters": [{"title": "Introduction", "authors": [], "printed_page_number": "1"}],
                }),
                encoding="utf-8",
            )
            results, no_coverage = evaluate_corpus()
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].key, "9783899718188")
            self.assertEqual(results[0].agreement_rate, 1.0)
            self.assertEqual(no_coverage, [])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_evaluate_crossref.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'evaluate_crossref'`

- [ ] **Step 3: Write minimal implementation**

Create `cli/evaluate_crossref.py`:

```python
#!/usr/bin/env python3
"""Measures agreement between this corpus's own ground truth and each
book's cached Crossref chapter data -- see design spec
docs/superpowers/specs/2026-08-21-crossref-cross-validation-design.md.

For every book with both a .expected.json and a cached
.crossref-cache/<isbn>.crossref.json carrying at least one chapter,
compares the ground truth's real chapters ("skip": false) against the
Crossref chapter list via matching.diff_toc_entries -- reused completely
unmodified, since it already aligns on title (chapter-number-prefix and
capitalization normalized) and first-page-number equivalence, exactly
this script's comparison spec. Books with no cached Crossref data at all
are reported separately, not silently dropped.

Usage:
    uv run python cli/evaluate_crossref.py
    uv run python cli/evaluate_crossref.py --min-agreement 0.8
"""

import argparse
import json
from dataclasses import dataclass

from dnb_toc_ground_truth import corpus, matching
from dnb_toc_ground_truth.crossref import CrossrefBookData
from dnb_toc_ground_truth.toc_entry import TocEntry


@dataclass(frozen=True)
class BookAgreement:
    key: str
    matched: int
    only_in_gt: int
    only_in_crossref: int
    agreement_rate: float


def _load_gt_entries(key: str) -> tuple[TocEntry, ...]:
    data = json.loads(corpus.expected_json_path(key).read_text(encoding="utf-8"))
    return tuple(
        TocEntry(
            title=e["title"], authors=tuple(e.get("authors", [])),
            printed_page_number=e["printed_page_number"], source_page_index=-1, skip=e.get("skip", False),
        )
        for e in data["entries"]
    )


def _load_crossref_data(key: str) -> CrossrefBookData | None:
    path = corpus.crossref_cache_dir() / f"{key}.crossref.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    chapters = tuple(
        TocEntry(
            title=c["title"], authors=tuple(c["authors"]),
            printed_page_number=c["printed_page_number"], source_page_index=-1, skip=False,
        )
        for c in data["chapters"]
    )
    return CrossrefBookData(isbn=key, doi=data.get("doi"), chapters=chapters, fetched_at=data["fetched_at"])


def evaluate_book(key: str, gt_entries: tuple[TocEntry, ...], crossref_data: CrossrefBookData) -> BookAgreement:
    gt_real = [e for e in gt_entries if not e.skip]
    matched_pairs, only_in_gt, only_in_crossref = matching.diff_toc_entries(gt_real, list(crossref_data.chapters))
    denominator = max(len(gt_real), len(crossref_data.chapters))
    agreement_rate = len(matched_pairs) / denominator if denominator else 0.0
    return BookAgreement(
        key=key, matched=len(matched_pairs), only_in_gt=len(only_in_gt),
        only_in_crossref=len(only_in_crossref), agreement_rate=agreement_rate,
    )


def evaluate_corpus() -> tuple[list[BookAgreement], list[str]]:
    """Returns (results, keys_with_no_crossref_coverage) for every
    manifest book that has a .expected.json."""
    results = []
    no_coverage = []
    for book in corpus.load_manifest_books():
        key = corpus.manifest_key(book)
        if not corpus.expected_json_path(key).exists():
            continue
        crossref_data = _load_crossref_data(key)
        if crossref_data is None or not crossref_data.chapters:
            no_coverage.append(key)
            continue
        gt_entries = _load_gt_entries(key)
        results.append(evaluate_book(key, gt_entries, crossref_data))
    return results, no_coverage


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--min-agreement", type=float, default=None,
        help="Exit 1 if the aggregate mean agreement rate falls below this (0-1). Unset: no gate enforced.",
    )
    args = parser.parse_args()

    results, no_coverage = evaluate_corpus()
    for result in results:
        print(
            f"[{result.key}] agreement={result.agreement_rate:.0%} "
            f"matched={result.matched} only_in_gt={result.only_in_gt} only_in_crossref={result.only_in_crossref}"
        )

    if results:
        mean_agreement = sum(r.agreement_rate for r in results) / len(results)
        print(f"\n{len(results)} book(s) compared, mean agreement {mean_agreement:.0%}")
    else:
        mean_agreement = None
        print("\nNo books had both ground truth and cached Crossref chapter data.")
    print(f"{len(no_coverage)} book(s) with ground truth but no Crossref coverage.")

    if args.min_agreement is not None and (mean_agreement is None or mean_agreement < args.min_agreement):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_evaluate_crossref.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add cli/evaluate_crossref.py tests/test_evaluate_crossref.py
git commit -m "feat: add cli/evaluate_crossref.py to measure ground-truth/Crossref agreement"
```

---

## Task 8: Documentation and full-suite verification

**Files:**
- Modify: `cli/README.md`

- [ ] **Step 1: Regenerate the `fetch_corpus.py` entry**

Run:
```bash
uv run python cli/fetch_corpus.py --help
```

Replace the fenced code block under the existing `## \`fetch_corpus.py\`` heading in `cli/README.md` with this fresh `--help` output (the usage line and options will have grown to include `--contact-email`, `--crossref-cache-dir`, `--config-file`).

- [ ] **Step 2: Add a `backfill_crossref.py` entry**

Run:
```bash
uv run python cli/backfill_crossref.py --help
```

Insert a new section, alphabetically between `## \`arbitrate.py\`` and `## \`fetch_corpus.py\``:

```markdown
## `backfill_crossref.py`

Backfills Crossref book DOI and cached chapter data for existing
manifest entries that already have ground truth but no DOI yet.

```
<paste the --help output here>
```
```

- [ ] **Step 3: Add an `evaluate_crossref.py` entry**

Run:
```bash
uv run python cli/evaluate_crossref.py --help
```

Insert a new section, alphabetically between `## \`arbitrate.py\`` and `## \`backfill_crossref.py\`` (i.e. `arbitrate.py` → `backfill_crossref.py` → `evaluate_crossref.py` → `fetch_corpus.py` → `generate_ground_truth.py` → `select_eval_sample.py`):

```markdown
## `evaluate_crossref.py`

Measures agreement between this corpus's ground truth and each book's
cached Crossref chapter data, reusing `matching.diff_toc_entries`.

```
<paste the --help output here>
```
```

- [ ] **Step 4: Run the complete test suite**

Run: `uv run pytest -v`
Expected: PASS, every test in the repo (existing + all new ones added in Tasks 1–7)

- [ ] **Step 5: Commit**

```bash
git add cli/README.md
git commit -m "docs: document backfill_crossref.py and evaluate_crossref.py in cli/README.md"
```

---

## Final check

- [ ] Confirm the branch is `feature/crossref-cross-validation` (not main, not a worktree): `git branch --show-current`
- [ ] Confirm the working tree is clean: `git status --short`
- [ ] Confirm the full suite passes one more time: `uv run pytest -v`

At this point the branch is ready for `superpowers:finishing-a-development-branch` (merge / PR / cleanup decision) — do not merge or push without the user's explicit go-ahead.
