# dnb-toc-ground-truth Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract the dnb-toc-only ground-truth generation pipeline out of `chapter-segmentation` into a new standalone repository `dnb-toc-ground-truth`, generalizing its inference-endpoint system (dropping KISSKI-specific code, supporting N>=2 candidate models gated "any two agree"), then remove the migrated code/data from `chapter-segmentation` once a real smoke test against live endpoints succeeds.

**Architecture:** A new repo at `/Users/cboulanger/Code/dnb-toc-ground-truth`, built inside an isolated git worktree of `chapter-segmentation` (branch `dnb-toc-ground-truth-wip`) so source files are read from a stable snapshot. Core TOC data types (`TocEntry` etc.) are vendored (copied, not imported) so the new repo has zero dependency on the `chapter_segmentation` package. `chapter-segmentation`'s layout-classifier/NuExtract pilots keep working against the relocated corpus via a sibling-checkout path override. Cleanup of the old repo happens last, gated on a real two-endpoint smoke test the user runs.

**Tech Stack:** Python >=3.12, `uv` + `hatchling`, `openai` (AsyncOpenAI), `httpx`, `pypdf`, `rapidfuzz`, `pytest`/`pytest-asyncio`.

**Design spec:** `docs/superpowers/specs/2026-08-21-dnb-toc-ground-truth-extraction-design.md` — read it before starting; every task below implements a specific section of it.

---

## Path conventions used throughout this plan

- `OLD` = `/Users/cboulanger/Code/chapter-segmentation` (the real, live checkout — touched only in Phase 9)
- `WT` = `/Users/cboulanger/Code/chapter-segmentation-dnb-migration` (git worktree, source of truth for every file being ported, and where Phase 9's cleanup happens)
- `NEW` = `/Users/cboulanger/Code/dnb-toc-ground-truth` (the new standalone repo)

Every task that reads an existing file reads it from `WT`, never `OLD` — this keeps the live checkout untouched until Phase 9.

---

## Phase 0: Workspace setup

### Task 1: Create the worktree and the new repo's skeleton

**Files:**
- Create: worktree at `WT`
- Create: `NEW/` (git-initialized), `NEW/pyproject.toml`, `NEW/.gitignore`, `NEW/README.md` (stub, filled in Task 20), `NEW/src/dnb_toc_ground_truth/__init__.py`, `NEW/tests/__init__.py`, `NEW/cli/`, `NEW/docs/superpowers/{specs,plans}/`, `NEW/data/corpus/pilot/{pdf,llm-cache,ground-truth}/`

- [ ] **Step 1: Create the worktree**

```bash
cd /Users/cboulanger/Code/chapter-segmentation
git worktree add /Users/cboulanger/Code/chapter-segmentation-dnb-migration dnb-toc-ground-truth-wip
```

- [ ] **Step 2: Create the new repo's directory tree**

```bash
mkdir -p /Users/cboulanger/Code/dnb-toc-ground-truth/{cli,src/dnb_toc_ground_truth,tests,docs/superpowers/specs,docs/superpowers/plans}
mkdir -p /Users/cboulanger/Code/dnb-toc-ground-truth/data/corpus/pilot/{pdf,llm-cache,ground-truth}
cd /Users/cboulanger/Code/dnb-toc-ground-truth
git init
touch src/dnb_toc_ground_truth/__init__.py tests/__init__.py
```

- [ ] **Step 3: Write `NEW/.gitignore`**

```
.venv/
__pycache__/
*.pyc
.pytest_cache/
.DS_Store
.endpoints
.config
data/corpus/pilot/pdf/
data/corpus/pilot/.lobid-cache/
data/corpus/pilot/.locks/
data/corpus/pilot/.layout-cache/
```

- [ ] **Step 4: Write `NEW/pyproject.toml`**

```toml
[project]
name = "dnb-toc-ground-truth"
version = "0.1.0"
description = "Pilot: structured ground truth from DNB's open table-of-contents scans, via gated independent LLM reads"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "pypdf>=5.1.0",
    "rapidfuzz>=3.10.0",
    "openai>=1.0.0",
    "httpx>=0.27.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
python_files = "test_*.py"
python_classes = "Test*"
python_functions = "test_*"
asyncio_mode = "auto"

[tool.hatch.build.targets.wheel]
packages = ["src/dnb_toc_ground_truth"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[dependency-groups]
dev = [
    "pytest>=8.3.0",
    "pytest-asyncio>=0.24.0",
]
```

- [ ] **Step 5: Write a stub `NEW/README.md` (real content lands in Task 20)**

```markdown
# dnb-toc-ground-truth

Placeholder -- see Task 20 of docs/superpowers/plans/2026-08-21-dnb-toc-ground-truth-extraction.md.
```

- [ ] **Step 6: Install and verify the empty project builds**

```bash
cd /Users/cboulanger/Code/dnb-toc-ground-truth
uv sync
uv run pytest
```

Expected: `uv sync` succeeds; `pytest` reports "no tests ran" (empty `tests/`).

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "chore: initialize dnb-toc-ground-truth repo skeleton"
```

---

## Phase 1: Vendored core data types

### Task 2: Vendor `TocEntry` and its parsing helpers

**Files:**
- Create: `NEW/src/dnb_toc_ground_truth/toc_entry.py`
- Create: `NEW/tests/test_toc_entry.py`

This is the one deliberate fork from `chapter_segmentation.segmentation`/`_llm_json` (see design spec "Moves to dnb-toc-ground-truth"). Read `WT/src/chapter_segmentation/segmentation.py` (the `TocEntry` class at line ~238, `_parse_toc_page_number` at line ~91, `_normalize_printed_page_number` at line ~107, `_toc_items_to_entries` at line ~640) and `WT/src/chapter_segmentation/_llm_json.py` (`parse_json_array`) to copy from — the code below is that exact logic, copied verbatim into one new module with no other dependencies.

- [ ] **Step 1: Write `NEW/src/dnb_toc_ground_truth/toc_entry.py`**

```python
"""TOC-entry data type and parsing helpers -- vendored from
chapter-segmentation's src/chapter_segmentation/segmentation.py
(TocEntry, _parse_toc_page_number, _normalize_printed_page_number,
_toc_items_to_entries) and src/chapter_segmentation/_llm_json.py
(parse_json_array), copied rather than imported so this repo has zero
dependency on the chapter_segmentation package -- see design spec
docs/superpowers/specs/2026-08-21-dnb-toc-ground-truth-extraction-design.md.
Kept byte-for-byte identical to its origin except for this module's own
docstring and any DNB-toc-only-specific mentions that no longer need a
disclaimer, since this whole repo IS dnb-toc-only-specific now."""

import json
import math
import re
from dataclasses import dataclass

_STRICT_ROMAN_RE = re.compile(r"^m{0,3}(cm|cd|d?c{0,3})(xc|xl|l?x{0,3})(ix|iv|v?i{0,3})$")
_ROMAN_VALUES = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}
_ROMAN_PAGE_MAX_VALUE = 50


def _parse_toc_page_number(raw: str) -> int | None:
    """The integer value of a TOC line's captured page field, or None if
    the field is not a plausible page number (an invalid or implausibly
    large roman numeral)."""
    if raw.isdigit():
        return int(raw)
    lowered = raw.lower()
    if not _STRICT_ROMAN_RE.match(lowered) or not lowered:
        return None
    total = 0
    for ch, nxt in zip(lowered, lowered[1:] + " "):
        value = _ROMAN_VALUES[ch]
        total += -value if nxt != " " and _ROMAN_VALUES.get(nxt, 0) > value else value
    return total if total <= _ROMAN_PAGE_MAX_VALUE else None


def _normalize_printed_page_number(value: object) -> str | None:
    """Coerces any of TocEntry.printed_page_number's legal input shapes
    into the canonical str | None form."""
    if isinstance(value, str):
        text = value.strip()
        return None if not text or text.lower() == "null" else text
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        value = int(value)
        return None if value == -1 else str(value)
    return None


@dataclass(frozen=True)
class TocEntry:
    title: str
    printed_page_number: str | None
    source_page_index: int  # which page (0-based) the TOC entry itself was found on
    authors: tuple[str, ...] = ()
    printed_roman: bool = False
    title_variants: tuple[str, ...] = ()
    skip: bool = False  # True when this entry is not itself a real chapter --
    # a part/section divider, or front/back matter (preface, bibliography,
    # index, ...). Set by vision.py/ocr.py's extraction: both deliberately
    # extract EVERY printed TOC line verbatim rather than omitting
    # non-chapter lines outright, so a two-model disagreement over what to
    # extract can never cause an editorial-judgment mismatch to fail the
    # whole-book agreement gate (matching.py) -- only a genuine reading
    # mismatch can.

    def __post_init__(self) -> None:
        object.__setattr__(self, "printed_page_number", _normalize_printed_page_number(self.printed_page_number))


def _toc_items_to_entries(items: list) -> list[TocEntry]:
    """Converts a parsed JSON array of {"title", "authors",
    "printed_page_number"} dicts -- the shape both vision.py's and
    ocr.py's prompts ask an LLM to return -- into TocEntry objects,
    tolerating the malformed-response shapes a real model occasionally
    produces. An optional "skip" key becomes TocEntry.skip (default
    False when absent)."""
    entries: list[TocEntry] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        if len(title) < 3:
            continue
        raw_authors = item.get("authors")
        authors = tuple(str(a).strip() for a in raw_authors if str(a).strip()) if isinstance(raw_authors, list) else ()
        printed_page_number = _normalize_printed_page_number(item.get("printed_page_number"))
        printed_roman = (
            printed_page_number is not None
            and not printed_page_number.isdigit()
            and _parse_toc_page_number(printed_page_number) is not None
        )
        entries.append(TocEntry(
            title=title, printed_page_number=printed_page_number, source_page_index=-1,
            authors=authors, printed_roman=printed_roman, skip=bool(item.get("skip", False)),
        ))
    return entries


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(line for line in lines if not line.startswith("```")).strip()
    return text


def parse_json_array(text: str) -> list:
    """Extract and parse the first JSON array ([...]) found in *text*."""
    text = _strip_code_fence(text)
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON array found in LLM response: {text!r}")
    return json.loads(text[start: end + 1])
```

- [ ] **Step 2: Write `NEW/tests/test_toc_entry.py`**

```python
"""Unit tests for toc_entry.py -- vendored from
chapter-segmentation's tests/test_segmentation.py's TocEntry/
_parse_toc_page_number/_toc_items_to_entries coverage and
tests/test_llm_json.py's parse_json_array coverage, trimmed to what this
repo actually exercises (page-number parsing, item-to-entry conversion,
JSON-array extraction)."""

import unittest

from dnb_toc_ground_truth.toc_entry import TocEntry, _parse_toc_page_number, _toc_items_to_entries, parse_json_array


class TestParseTocPageNumber(unittest.TestCase):
    def test_parses_digit_string(self):
        self.assertEqual(_parse_toc_page_number("42"), 42)

    def test_parses_lowercase_roman(self):
        self.assertEqual(_parse_toc_page_number("vii"), 7)

    def test_rejects_implausibly_large_roman(self):
        self.assertIsNone(_parse_toc_page_number("mmmmm"))

    def test_rejects_non_roman_word(self):
        self.assertIsNone(_parse_toc_page_number("civil"))


class TestTocEntryPageNormalization(unittest.TestCase):
    def test_legacy_negative_one_sentinel_becomes_none(self):
        entry = TocEntry(title="X", printed_page_number=-1, source_page_index=0)
        self.assertIsNone(entry.printed_page_number)

    def test_string_is_stripped(self):
        entry = TocEntry(title="X", printed_page_number=" 12 ", source_page_index=0)
        self.assertEqual(entry.printed_page_number, "12")


class TestTocItemsToEntries(unittest.TestCase):
    def test_converts_well_formed_item(self):
        entries = _toc_items_to_entries([
            {"title": "Introduction", "authors": ["Jane Doe"], "printed_page_number": "12", "skip": False},
        ])
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].title, "Introduction")
        self.assertEqual(entries[0].authors, ("Jane Doe",))
        self.assertFalse(entries[0].skip)

    def test_skips_items_with_too_short_title(self):
        entries = _toc_items_to_entries([{"title": "AB", "printed_page_number": "1"}])
        self.assertEqual(entries, [])

    def test_tolerates_string_authors_field(self):
        entries = _toc_items_to_entries([{"title": "Chapter One", "authors": "Jane Doe", "printed_page_number": "1"}])
        self.assertEqual(entries[0].authors, ())

    def test_defaults_skip_to_false_when_absent(self):
        entries = _toc_items_to_entries([{"title": "Chapter One", "printed_page_number": "1"}])
        self.assertFalse(entries[0].skip)


class TestParseJsonArray(unittest.TestCase):
    def test_extracts_bare_array(self):
        self.assertEqual(parse_json_array('[{"a": 1}]'), [{"a": 1}])

    def test_strips_markdown_code_fence(self):
        self.assertEqual(parse_json_array('```json\n[{"a": 1}]\n```'), [{"a": 1}])

    def test_raises_on_no_array_found(self):
        with self.assertRaises(ValueError):
            parse_json_array("no array here")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Run the tests**

```bash
cd /Users/cboulanger/Code/dnb-toc-ground-truth
uv run pytest tests/test_toc_entry.py -v
```

Expected: all tests PASS.

- [ ] **Step 4: Commit**

```bash
git add src/dnb_toc_ground_truth/toc_entry.py tests/test_toc_entry.py
git commit -m "feat: vendor TocEntry and TOC-parsing helpers"
```

### Task 3: Vendor `pdfalto_runner.py`

**Files:**
- Create: `NEW/src/dnb_toc_ground_truth/pdfalto_runner.py` (verbatim copy of `WT/evaluation/scripts/pdfalto_runner.py`, no changes needed — it has no imports beyond stdlib)
- Create: `NEW/tests/test_pdfalto_runner.py`

- [ ] **Step 1: Copy the file verbatim**

```bash
cp /Users/cboulanger/Code/chapter-segmentation-dnb-migration/evaluation/scripts/pdfalto_runner.py \
   /Users/cboulanger/Code/dnb-toc-ground-truth/src/dnb_toc_ground_truth/pdfalto_runner.py
```

- [ ] **Step 2: Write `NEW/tests/test_pdfalto_runner.py`**

```python
"""Unit tests for pdfalto_runner.py's pure resolve_pdfalto_binary logic --
no real pdfalto binary needed (ensure_alto_xml shells out and is exercised
manually, matching chapter-segmentation's own convention for this
module)."""

import os
import unittest
from unittest.mock import patch

from dnb_toc_ground_truth.pdfalto_runner import resolve_pdfalto_binary


class TestResolvePdfaltoBinary(unittest.TestCase):
    def test_explicit_cli_arg_wins(self):
        with patch.dict(os.environ, {"PDFALTO_BIN": "/env/pdfalto"}, clear=False):
            self.assertEqual(resolve_pdfalto_binary("/explicit/pdfalto"), "/explicit/pdfalto")

    def test_falls_back_to_env_var(self):
        with patch.dict(os.environ, {"PDFALTO_BIN": "/env/pdfalto"}, clear=False):
            self.assertEqual(resolve_pdfalto_binary(None), "/env/pdfalto")

    def test_falls_back_to_bare_command(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(resolve_pdfalto_binary(None), "pdfalto")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Run and commit**

```bash
cd /Users/cboulanger/Code/dnb-toc-ground-truth
uv run pytest tests/test_pdfalto_runner.py -v
git add src/dnb_toc_ground_truth/pdfalto_runner.py tests/test_pdfalto_runner.py
git commit -m "feat: vendor pdfalto_runner.py"
```

---

## Phase 2: Extraction and matching modules

### Task 4: Port `vision.py`

**Files:**
- Create: `NEW/src/dnb_toc_ground_truth/vision.py` (copy of `WT/evaluation/dnb_toc_vision.py`)
- Create: `NEW/tests/test_vision.py` (copy of `WT/tests/test_dnb_toc_vision.py`)

- [ ] **Step 1: Copy and fix imports**

```bash
cp /Users/cboulanger/Code/chapter-segmentation-dnb-migration/evaluation/dnb_toc_vision.py \
   /Users/cboulanger/Code/dnb-toc-ground-truth/src/dnb_toc_ground_truth/vision.py
```

In `NEW/src/dnb_toc_ground_truth/vision.py`, change:

```python
from chapter_segmentation._llm_json import parse_json_array
from chapter_segmentation.segmentation import TocEntry, _toc_items_to_entries
```

to:

```python
from dnb_toc_ground_truth.toc_entry import TocEntry, _toc_items_to_entries, parse_json_array
```

No other changes needed — the rest of the module (prompt text, caching, `render_pages_to_images`, `vision_extract_toc_entries`) has no other `chapter_segmentation`/`evaluation` dependency. Update the module docstring's `generate_dnb_toc_ground_truth.py, arbitrate_dnb_toc.py` mention to `cli/generate_ground_truth.py, cli/arbitrate.py` for accuracy.

- [ ] **Step 2: Copy the test file and fix imports**

```bash
cp /Users/cboulanger/Code/chapter-segmentation-dnb-migration/tests/test_dnb_toc_vision.py \
   /Users/cboulanger/Code/dnb-toc-ground-truth/tests/test_vision.py
```

In `NEW/tests/test_vision.py`, change:

```python
from chapter_segmentation.segmentation import TocEntry
from evaluation.dnb_toc_vision import (
    _MAX_VISION_PAGES,
    cache_path,
    load_cached_kind,
    load_cached_llm_entries,
    render_pages_to_images,
    versioned_cache_dir,
    vision_extract_toc_entries,
```

to:

```python
from dnb_toc_ground_truth.toc_entry import TocEntry
from dnb_toc_ground_truth.vision import (
    _MAX_VISION_PAGES,
    cache_path,
    load_cached_kind,
    load_cached_llm_entries,
    render_pages_to_images,
    versioned_cache_dir,
    vision_extract_toc_entries,
```

(Read the rest of the original file for any further `evaluation.dnb_toc_vision`/`chapter_segmentation` references beyond this header block — e.g. `write_cached_llm_entries` if imported separately — and apply the same `dnb_toc_ground_truth.vision`/`dnb_toc_ground_truth.toc_entry` substitution throughout. Test bodies (fixtures, assertions) are unchanged.)

- [ ] **Step 3: Run the tests**

```bash
cd /Users/cboulanger/Code/dnb-toc-ground-truth
uv run pytest tests/test_vision.py -v
```

Expected: all PASS except any test marked as needing the real `pdftoppm` binary, which should still pass if `poppler` is installed locally (same dependency `chapter-segmentation` already documents) — if `pdftoppm` isn't on `PATH` in this environment, that one test fails with a clear "binary not found" error; note it and continue (it will be re-verified in Phase 9's full-suite run once poppler is confirmed available).

- [ ] **Step 4: Commit**

```bash
git add src/dnb_toc_ground_truth/vision.py tests/test_vision.py
git commit -m "feat: port vision.py TOC extraction"
```

### Task 5: Port `ocr.py`

**Files:**
- Create: `NEW/src/dnb_toc_ground_truth/ocr.py` (copy of `WT/evaluation/dnb_toc_ocr.py`)
- Create: `NEW/tests/test_ocr.py` (copy of `WT/tests/test_dnb_toc_ocr.py`)

This module depends on `pdfalto_runner` (Task 3, done) and on `OpenAICompatibleLLMClient` (not yet ported — lands in Task 8's `inference.py`). Do Task 8 before this one if working strictly in order; if working out of order, stub the import and revisit.

- [ ] **Step 1: Copy and fix imports**

```bash
cp /Users/cboulanger/Code/chapter-segmentation-dnb-migration/evaluation/dnb_toc_ocr.py \
   /Users/cboulanger/Code/dnb-toc-ground-truth/src/dnb_toc_ground_truth/ocr.py
```

Change:

```python
from chapter_segmentation._llm_json import parse_json_array
from chapter_segmentation.segmentation import TocEntry, _toc_items_to_entries
from evaluation.inference_endpoints import OpenAICompatibleLLMClient
from evaluation.scripts import pdfalto_runner
```

to:

```python
from dnb_toc_ground_truth.toc_entry import TocEntry, _toc_items_to_entries, parse_json_array
from dnb_toc_ground_truth.inference import OpenAICompatibleLLMClient
from dnb_toc_ground_truth import pdfalto_runner
```

No other changes needed (OCR row-reconstruction, tessdata-best resolution, and the text-extraction prompt are unchanged).

- [ ] **Step 2: Copy the test file and fix imports**

```bash
cp /Users/cboulanger/Code/chapter-segmentation-dnb-migration/tests/test_dnb_toc_ocr.py \
   /Users/cboulanger/Code/dnb-toc-ground-truth/tests/test_ocr.py
```

Change:

```python
from evaluation.dnb_toc_ocr import (
    _MAX_TEXT_PAGES, _resolve_tessdata_best_env, _rows_from_alto_xml, ocr_pages_to_rows, text_extract_toc_entries,
)
```

to:

```python
from dnb_toc_ground_truth.ocr import (
    _MAX_TEXT_PAGES, _resolve_tessdata_best_env, _rows_from_alto_xml, ocr_pages_to_rows, text_extract_toc_entries,
)
```

Read the rest of the file for any `evaluation.scripts.pdfalto_runner`/`chapter_segmentation` mock-patch targets (e.g. `@patch("evaluation.dnb_toc_ocr.pdfalto_runner...")`) and update the patch-target string to `dnb_toc_ground_truth.ocr.pdfalto_runner...` accordingly.

- [ ] **Step 3: Run and commit**

```bash
cd /Users/cboulanger/Code/dnb-toc-ground-truth
uv run pytest tests/test_ocr.py -v
git add src/dnb_toc_ground_truth/ocr.py tests/test_ocr.py
git commit -m "feat: port ocr.py TOC extraction"
```

### Task 6: Port `matching.py`

**Files:**
- Create: `NEW/src/dnb_toc_ground_truth/matching.py` (copy of `WT/evaluation/dnb_toc_matching.py`)
- Create: `NEW/tests/test_matching.py` (copy of `WT/tests/test_dnb_toc_matching.py`)

- [ ] **Step 1: Copy and fix imports**

```bash
cp /Users/cboulanger/Code/chapter-segmentation-dnb-migration/evaluation/dnb_toc_matching.py \
   /Users/cboulanger/Code/dnb-toc-ground-truth/src/dnb_toc_ground_truth/matching.py
```

Change:

```python
from chapter_segmentation.segmentation import TocEntry, _parse_toc_page_number
```

to:

```python
from dnb_toc_ground_truth.toc_entry import TocEntry, _parse_toc_page_number
```

No other changes to this file in this task — `align_toc_entries`, `diff_toc_entries`, `gate_book`, `toc_entry_to_gt_dict`, and every title-normalization helper are unchanged. The N-way extension is a separate function added in Task 7, not a modification of these.

- [ ] **Step 2: Copy the test file and fix imports**

```bash
cp /Users/cboulanger/Code/chapter-segmentation-dnb-migration/tests/test_dnb_toc_matching.py \
   /Users/cboulanger/Code/dnb-toc-ground-truth/tests/test_matching.py
```

Change:

```python
from chapter_segmentation.segmentation import TocEntry
from evaluation.dnb_toc_matching import (
    _title_near_identical,
    align_toc_entries,
    diff_toc_entries,
    gate_book,
    toc_entry_to_gt_dict,
)
```

to:

```python
from dnb_toc_ground_truth.toc_entry import TocEntry
from dnb_toc_ground_truth.matching import (
    _title_near_identical,
    align_toc_entries,
    diff_toc_entries,
    gate_book,
    toc_entry_to_gt_dict,
)
```

- [ ] **Step 3: Run and commit**

```bash
cd /Users/cboulanger/Code/dnb-toc-ground-truth
uv run pytest tests/test_matching.py -v
git add src/dnb_toc_ground_truth/matching.py tests/test_matching.py
git commit -m "feat: port matching.py pairwise agreement gate"
```

### Task 7: Add N-way "best pair wins" gating to `matching.py`

**Files:**
- Modify: `NEW/src/dnb_toc_ground_truth/matching.py`
- Modify: `NEW/tests/test_matching.py`

Implements design spec's "N-way gating" section: extends the pairwise gate to N>=2 entry lists without changing `gate_book`/`align_toc_entries`/`diff_toc_entries` at all.

- [ ] **Step 1: Write the failing tests** — append to `NEW/tests/test_matching.py`:

```python
from dnb_toc_ground_truth.matching import gate_books


def _entry(title: str, page) -> TocEntry:
    return TocEntry(title=title, printed_page_number=page, source_page_index=0)


class TestGateBooks(unittest.TestCase):
    def test_raises_on_fewer_than_two_lists(self):
        with self.assertRaises(ValueError):
            gate_books([[_entry("A", "1")]])

    def test_passes_when_two_of_three_agree(self):
        a = [_entry("Introduction", "1"), _entry("Chapter One", "5")]
        b = [_entry("Introduction", "1"), _entry("Chapter One", "5")]
        c = [_entry("Totally different reading", "99")]
        passed, merged, winning_pair = gate_books([a, b, c], threshold=0.90)
        self.assertTrue(passed)
        self.assertEqual(winning_pair, (0, 1))
        self.assertEqual({e.title for e in merged}, {"Introduction", "Chapter One"})

    def test_fails_when_no_pair_agrees(self):
        a = [_entry("Reading A", "1")]
        b = [_entry("Reading B", "2")]
        c = [_entry("Reading C", "3")]
        passed, merged, winning_pair = gate_books([a, b, c], threshold=0.90)
        self.assertFalse(passed)
        self.assertEqual(merged, [])
        self.assertIsNone(winning_pair)

    def test_prefers_highest_agreement_pair_when_multiple_pass(self):
        # a/b agree fully (2/2); a/c agree on only one of two (page mismatch
        # on the second entry) -- both clear a low 0.5 threshold, but a/b's
        # higher rate must win.
        a = [_entry("Introduction", "1"), _entry("Chapter One", "5")]
        b = [_entry("Introduction", "1"), _entry("Chapter One", "5")]
        c = [_entry("Introduction", "1"), _entry("Chapter One", "999")]
        passed, merged, winning_pair = gate_books([a, b, c], threshold=0.5)
        self.assertTrue(passed)
        self.assertEqual(winning_pair, (0, 1))
```

- [ ] **Step 2: Run to verify the new tests fail**

```bash
cd /Users/cboulanger/Code/dnb-toc-ground-truth
uv run pytest tests/test_matching.py -k GateBooks -v
```

Expected: FAIL with `ImportError: cannot import name 'gate_books'`.

- [ ] **Step 3: Implement `gate_books`** — append to `NEW/src/dnb_toc_ground_truth/matching.py`:

```python
from typing import Optional


def gate_books(
    entry_lists: list[list[TocEntry]], threshold: float = 0.90,
) -> tuple[bool, list[TocEntry], Optional[tuple[int, int]]]:
    """Extends gate_book to N >= 2 independently-produced TocEntry lists
    -- "best pair wins" (design spec
    docs/superpowers/specs/2026-08-21-dnb-toc-ground-truth-extraction-design.md
    "N-way gating"): every pair among entry_lists is checked with the SAME
    pairwise alignment + agreement-rate + near-identical-title gate
    gate_book already implements, unchanged. The book passes if at least
    one pair clears `threshold`; among passing pairs, the one with the
    HIGHEST agreement rate is used, and its merged output (gate_book's own
    merge -- union of matched pairs plus each side's singletons) becomes
    the result. Returns (passed, merged_entries, winning_pair_indices) --
    winning_pair_indices names which two positions in entry_lists produced
    the result (for logging which two models actually agreed), None if
    every pair failed.

    Deliberately does NOT attempt a multi-way consensus merge across more
    than one passing pair -- see the design spec section this implements
    for why: reusing gate_book's proven, heavily-tested two-list merge
    logic verbatim was chosen over a new N-way merge algorithm."""
    if len(entry_lists) < 2:
        raise ValueError(f"gate_books needs at least 2 entry lists to gate, got {len(entry_lists)}")
    best_rate = -1.0
    best_pair: Optional[tuple[int, int]] = None
    for i in range(len(entry_lists)):
        for j in range(i + 1, len(entry_lists)):
            a, b = entry_lists[i], entry_lists[j]
            if not a and not b:
                continue
            matched_pairs, _, _ = diff_toc_entries(a, b)
            rate = len(matched_pairs) / max(len(a), len(b))
            if rate < threshold or rate <= best_rate:
                continue
            if any(not _title_near_identical(entry_a, entry_b) for entry_a, entry_b in matched_pairs):
                continue
            best_rate = rate
            best_pair = (i, j)
    if best_pair is None:
        return False, [], None
    i, j = best_pair
    passed, merged = gate_book(entry_lists[i], entry_lists[j], threshold=threshold)
    assert passed  # guaranteed by the loop above having already checked this exact pair
    return True, merged, best_pair
```

- [ ] **Step 4: Run to verify all tests pass**

```bash
uv run pytest tests/test_matching.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dnb_toc_ground_truth/matching.py tests/test_matching.py
git commit -m "feat: add N-way best-pair-wins gating (gate_books)"
```

---

## Phase 3: Generic inference-endpoint system

### Task 8: Build `inference.py`

**Files:**
- Create: `NEW/src/dnb_toc_ground_truth/inference.py`
- Create: `NEW/tests/test_inference.py`

Implements the design spec's "Endpoint and config system" section in full: `.endpoints` file parsing (JSON-array and legacy plain-text), `.config` file loading, and N-model resolution by exact model-id match with the "Running"-status tie-break. No KISSKI code carries over at all.

- [ ] **Step 1: Write the failing tests** — `NEW/tests/test_inference.py`:

```python
"""Unit tests for inference.py -- endpoints-file parsing (both formats)
and model-id resolution. No real network calls; AsyncOpenAI client
construction is exercised for real (it doesn't connect until a call is
made) so tests can assert on endpoint.client.base_url."""

import json
import tempfile
import unittest
from pathlib import Path

from dnb_toc_ground_truth.inference import (
    ModelEndpoint, load_config, load_endpoint_entries, resolve_model_endpoints,
)


def _write(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


class TestLoadEndpointEntriesJson(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_parses_model_field_directly(self):
        path = _write(self.tmp_path, ".endpoints", json.dumps([
            {"url": "https://example.invalid/session-a", "key": "secret-a", "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct"},
        ]))
        entries = load_endpoint_entries(path)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].model, "Qwen/Qwen3-Omni-30B-A3B-Instruct")
        self.assertEqual(entries[0].base_url, "https://example.invalid/session-a/v1")

    def test_parses_model_from_framework_args_when_model_key_absent(self):
        path = _write(self.tmp_path, ".endpoints", json.dumps([
            {
                "url": "https://example.invalid/session-b/v1", "key": "secret-b",
                "framework_args": "--model=mistralai/Mistral-Small-3.2-24B-Instruct-2506 --tensor-parallel-size=2",
            },
        ]))
        entries = load_endpoint_entries(path)
        self.assertEqual(entries[0].model, "mistralai/Mistral-Small-3.2-24B-Instruct-2506")

    def test_skips_entry_missing_required_fields(self):
        path = _write(self.tmp_path, ".endpoints", json.dumps([
            {"url": "https://example.invalid/x", "key": "secret"},  # no model, no framework_args
        ]))
        with self.assertRaises(ValueError):
            load_endpoint_entries(path)

    def test_carries_status_field(self):
        path = _write(self.tmp_path, ".endpoints", json.dumps([
            {"url": "https://example.invalid/x", "key": "secret", "model": "model-a", "status": "Running"},
        ]))
        entries = load_endpoint_entries(path)
        self.assertEqual(entries[0].status, "Running")


class TestLoadEndpointEntriesPlainText(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_parses_pasted_session_table(self):
        path = _write(self.tmp_path, ".endpoints", (
            "framework\tvLLM\n"
            "framework_args\t--model=Qwen/Qwen3-Omni-30B-A3B-Instruct --tensor-parallel-size=2\n"
            "key\tsecret-a\n"
            "url\thttps://llm.mpcdf.mpg.de/abc123\n"
        ))
        entries = load_endpoint_entries(path)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].model, "Qwen/Qwen3-Omni-30B-A3B-Instruct")
        self.assertEqual(entries[0].base_url, "https://llm.mpcdf.mpg.de/abc123/v1")
        self.assertEqual(entries[0].status, "")

    def test_parses_multiple_blocks_separated_by_blank_line(self):
        path = _write(self.tmp_path, ".endpoints", (
            "framework_args\t--model=model-a\nkey\tkey-a\nurl\thttps://x.invalid/a/v1\n"
            "\n"
            "framework_args\t--model=model-b\nkey\tkey-b\nurl\thttps://x.invalid/b/v1\n"
        ))
        entries = load_endpoint_entries(path)
        self.assertEqual([e.model for e in entries], ["model-a", "model-b"])

    def test_raises_on_missing_file(self):
        with self.assertRaises(ValueError):
            load_endpoint_entries(self.tmp_path / "nonexistent")


class TestResolveModelEndpoints(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _entries(self, rows):
        path = _write(self.tmp_path, ".endpoints", json.dumps(rows))
        return load_endpoint_entries(path)

    def test_resolves_exact_model_match(self):
        entries = self._entries([{"url": "https://x.invalid/a", "key": "k", "model": "model-a"}])
        resolved = resolve_model_endpoints(["model-a"], "vision", entries)
        self.assertEqual(len(resolved), 1)
        self.assertIsInstance(resolved[0], ModelEndpoint)
        self.assertEqual(resolved[0].model_id, "model-a")
        self.assertEqual(resolved[0].kind, "vision")

    def test_raises_when_model_not_found(self):
        entries = self._entries([{"url": "https://x.invalid/a", "key": "k", "model": "model-a"}])
        with self.assertRaises(ValueError):
            resolve_model_endpoints(["model-missing"], "vision", entries)

    def test_disambiguates_duplicate_model_by_running_status(self):
        entries = self._entries([
            {"url": "https://x.invalid/a", "key": "k1", "model": "model-a", "status": "Stopped"},
            {"url": "https://x.invalid/b", "key": "k2", "model": "model-a", "status": "Running"},
        ])
        resolved = resolve_model_endpoints(["model-a"], "vision", entries)
        self.assertEqual(str(resolved[0].client.base_url), "https://x.invalid/b/v1")

    def test_raises_on_ambiguous_duplicate_with_no_running_tiebreak(self):
        entries = self._entries([
            {"url": "https://x.invalid/a", "key": "k1", "model": "model-a"},
            {"url": "https://x.invalid/b", "key": "k2", "model": "model-a"},
        ])
        with self.assertRaises(ValueError):
            resolve_model_endpoints(["model-a"], "vision", entries)

    def test_resolves_same_model_id_twice_for_two_independent_reads(self):
        entries = self._entries([{"url": "https://x.invalid/a", "key": "k", "model": "model-a"}])
        resolved = resolve_model_endpoints(["model-a", "model-a"], "vision", entries)
        self.assertEqual(len(resolved), 2)


class TestLoadConfig(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_returns_empty_dict_when_missing(self):
        self.assertEqual(load_config(self.tmp_path / "nonexistent"), {})

    def test_parses_json_config(self):
        path = _write(self.tmp_path, ".config", json.dumps({"use_vision": ["model-a", "model-b"], "concurrency": 2}))
        self.assertEqual(load_config(path), {"use_vision": ["model-a", "model-b"], "concurrency": 2})


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify failure**

```bash
cd /Users/cboulanger/Code/dnb-toc-ground-truth
uv run pytest tests/test_inference.py -v
```

Expected: FAIL (`ModuleNotFoundError: No module named 'dnb_toc_ground_truth.inference'`).

- [ ] **Step 3: Write `NEW/src/dnb_toc_ground_truth/inference.py`**

```python
"""Generic OpenAI-compatible inference-endpoint resolution --
forked from chapter-segmentation's evaluation/inference_endpoints.py with
all KISSKI-specific auto-discovery removed. Every model must be named
explicitly via --use-vision/--use-text and resolved against an
--endpoints-file; see design spec
docs/superpowers/specs/2026-08-21-dnb-toc-ground-truth-extraction-design.md
"Endpoint and config system"."""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from openai import AsyncOpenAI

DEFAULT_TIMEOUT = 90.0
DEFAULT_ENDPOINTS_FILENAME = ".endpoints"
DEFAULT_CONFIG_FILENAME = ".config"

_MODEL_ARG_RE = re.compile(r"--model[= ](\S+)")


@dataclass(frozen=True)
class ModelEndpoint:
    """One ready-to-call (client, model_id) pair, plus which extraction
    path it was requested for ("vision" or "text"). `label` is the
    resolved model id, used only for log/print output."""

    label: str
    model_id: str
    kind: str
    client: AsyncOpenAI


class OpenAICompatibleLLMClient:
    """Minimal LLMClient wrapping an already-built AsyncOpenAI client --
    callers construct the client themselves (via resolve_model_endpoints
    below), so this class has no provider-specific knowledge at all."""

    def __init__(self, model: str, client: AsyncOpenAI):
        self._client = client
        self._model = model

    async def generate(
        self, prompt: str, *, max_tokens: int, temperature: float,
        is_valid: Optional[Callable[[str], bool]] = None,
    ) -> str:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""


@dataclass(frozen=True)
class _EndpointEntry:
    """One parsed row from an --endpoints-file, before resolution against
    a requested model id. `status` is the JSON format's raw status string
    ("Running", "Stopped", ...) used only to break a multi-match tie --
    always "" for the plain-text format, which has no equivalent field."""

    base_url: str
    api_key: str
    model: str
    status: str = ""


def _normalize_base_url(url: str) -> str:
    return url if url.rstrip("/").endswith("/v1") else url.rstrip("/") + "/v1"


def _parse_session_block(block: str) -> dict[str, str]:
    """Parses one pasted dashboard session table (tab-separated
    `field<TAB>value` lines, exactly as copied from a provider's UI) into
    a dict."""
    fields: dict[str, str] = {}
    for line in block.splitlines():
        if "\t" not in line:
            continue
        key, _, value = line.partition("\t")
        key = key.strip()
        if key:
            fields[key] = value.strip()
    return fields


def _model_from_fields(fields: dict[str, str]) -> str:
    model = fields.get("model", "").strip()
    if model:
        return model
    match = _MODEL_ARG_RE.search(fields.get("framework_args", ""))
    return match.group(1) if match else ""


def _parse_plain_text_endpoints(text: str) -> list[_EndpointEntry]:
    """Legacy pasted-session-table format (backward-compatible
    alternative to the JSON array format): one or more blocks separated
    by a blank line, each with `url`/`key`/`framework_args` (or `model`)
    fields. A block missing url/key/model is skipped."""
    entries = []
    for block in (b for b in re.split(r"\n\s*\n", text.strip()) if b.strip()):
        fields = _parse_session_block(block)
        url = fields.get("url", "").strip()
        api_key = fields.get("key", "").strip()
        model = _model_from_fields(fields)
        if not (url and api_key and model):
            continue
        entries.append(_EndpointEntry(base_url=_normalize_base_url(url), api_key=api_key, model=model))
    return entries


def _parse_json_endpoints(data: list[dict]) -> list[_EndpointEntry]:
    """Officially-supported endpoints-file format: a JSON array of
    objects as pasted from a provider dashboard. Consumes only `url`,
    `key`, and the model id (from `model` if present, else parsed out of
    `framework_args`'s `--model=...` token), plus `status` for
    tie-breaking -- every other field (framework, gpus, job_id, ...) is
    ignored. An entry missing url/key/model is skipped."""
    entries = []
    for row in data:
        url = str(row.get("url", "")).strip()
        api_key = str(row.get("key", "")).strip()
        model = str(row.get("model", "")).strip()
        if not model:
            match = _MODEL_ARG_RE.search(str(row.get("framework_args", "")))
            model = match.group(1) if match else ""
        if not (url and api_key and model):
            continue
        entries.append(_EndpointEntry(
            base_url=_normalize_base_url(url), api_key=api_key, model=model,
            status=str(row.get("status", "")),
        ))
    return entries


def load_endpoint_entries(path: Path) -> list[_EndpointEntry]:
    """Parses an --endpoints-file, auto-detecting the JSON-array format
    vs. the plain-text pasted-session-table format by trying JSON first.
    Raises ValueError naming the path if it doesn't exist or contains no
    usable entry -- meant to be diagnosed directly by whoever set up the
    file, not a bare empty-list surprise downstream."""
    if not path.exists():
        raise ValueError(f"--endpoints-file {path} does not exist")
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = None
    entries = _parse_json_endpoints(data) if isinstance(data, list) else _parse_plain_text_endpoints(text)
    if not entries:
        raise ValueError(f"--endpoints-file {path} has no usable endpoint entries")
    return entries


def resolve_model_endpoints(
    model_ids: list[str], kind: str, entries: list[_EndpointEntry], *, timeout: float = DEFAULT_TIMEOUT,
) -> list[ModelEndpoint]:
    """Resolves each of `model_ids` (in order; duplicates allowed -- the
    same model id may be requested twice for two independent reads
    against endpoints that happen to share a model id) against `entries`
    by exact model-id match. More than one entry matching the same id is
    resolved by preferring the one whose `status` is "Running"; if that
    still leaves more than one candidate (or none of the matches report
    status at all -- the plain-text format never does), raises ValueError
    naming the ambiguous id so the caller can fix the endpoints file.
    Raises ValueError naming the id if no entry matches at all."""
    resolved = []
    for model_id in model_ids:
        matches = [e for e in entries if e.model == model_id]
        if not matches:
            raise ValueError(f"no endpoint found for model {model_id!r} in the endpoints file")
        if len(matches) > 1:
            running = [e for e in matches if e.status == "Running"]
            if len(running) == 1:
                matches = running
            else:
                raise ValueError(
                    f"model {model_id!r} matches {len(matches)} endpoint entries and exactly one \"Running\" "
                    f"entry could not be identified to disambiguate -- fix the endpoints file"
                )
        entry = matches[0]
        client = AsyncOpenAI(base_url=entry.base_url, api_key=entry.api_key, timeout=timeout)
        resolved.append(ModelEndpoint(label=entry.model, model_id=entry.model, kind=kind, client=client))
    return resolved


def load_config(path: Path) -> dict:
    """Parses a --config-file (JSON) into a dict of CLI-flag defaults.
    Returns {} if the file doesn't exist -- config is optional, every
    value has its own script-level default."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
```

- [ ] **Step 4: Run to verify all tests pass**

```bash
uv run pytest tests/test_inference.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dnb_toc_ground_truth/inference.py tests/test_inference.py
git commit -m "feat: generic multi-model inference-endpoint resolution (no KISSKI)"
```

---

## Phase 4: Corpus helpers

### Task 9: Build `corpus.py`

**Files:**
- Create: `NEW/src/dnb_toc_ground_truth/corpus.py`
- Create: `NEW/tests/test_corpus.py`

Slim, single-corpus equivalent of `chapter-segmentation`'s `evaluation/harness.py`. This repo has exactly one corpus (`data/corpus/pilot/`), split into `pdf/` and `ground-truth/` subdirectories per the design spec's repo layout (a structural change from the old repo's flat `dnb-toc-only/<key>.pdf` + `<key>.expected.json` layout) — every path-construction helper other scripts need lives here so that split is made in exactly one place.

- [ ] **Step 1: Write the failing test** — `NEW/tests/test_corpus.py`:

```python
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from dnb_toc_ground_truth import corpus


class TestCorpusPaths(unittest.TestCase):
    def test_pdf_path_lives_under_pdf_subdir(self):
        self.assertEqual(corpus.pdf_path("9783899718188").name, "9783899718188.pdf")
        self.assertEqual(corpus.pdf_path("9783899718188").parent, corpus.pdf_dir())

    def test_expected_json_path_lives_under_ground_truth_subdir(self):
        path = corpus.expected_json_path("9783899718188")
        self.assertEqual(path.name, "9783899718188.expected.json")
        self.assertEqual(path.parent, corpus.ground_truth_dir())


class TestManifestKey(unittest.TestCase):
    def test_strips_pdf_extension(self):
        self.assertEqual(corpus.manifest_key({"filename": "9783899718188.pdf"}), "9783899718188")


class TestLoadManifestBooks(unittest.TestCase):
    def test_reads_books_list_from_manifest_json(self):
        with patch.object(corpus, "manifest_path", return_value=Path("/tmp/does-not-matter")):
            with patch.object(Path, "read_text", return_value=json.dumps({"books": [{"filename": "a.pdf"}]})):
                self.assertEqual(corpus.load_manifest_books(), [{"filename": "a.pdf"}])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify failure**

```bash
cd /Users/cboulanger/Code/dnb-toc-ground-truth
uv run pytest tests/test_corpus.py -v
```

Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Write `NEW/src/dnb_toc_ground_truth/corpus.py`**

```python
"""Corpus-loading helpers for dnb-toc-ground-truth -- slim, single-corpus
equivalent of chapter-segmentation's evaluation/harness.py. This repo has
exactly one corpus (data/corpus/pilot/), so there's no multi-corpus
list_corpora() indirection to carry over. PDFs live under pdf/ and
ground-truth JSON lives under ground-truth/ -- see design spec
docs/superpowers/specs/2026-08-21-dnb-toc-ground-truth-extraction-design.md
"Repo layout"."""

import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CORPUS_DIR = _REPO_ROOT / "data" / "corpus" / "pilot"


def corpus_dir() -> Path:
    return CORPUS_DIR


def pdf_dir() -> Path:
    return CORPUS_DIR / "pdf"


def ground_truth_dir() -> Path:
    return CORPUS_DIR / "ground-truth"


def llm_cache_dir() -> Path:
    return CORPUS_DIR / "llm-cache"


def lobid_cache_dir() -> Path:
    return CORPUS_DIR / ".lobid-cache"


def locks_dir() -> Path:
    return CORPUS_DIR / ".locks"


def manifest_path() -> Path:
    return CORPUS_DIR / "manifest.json"


def eval_tier_ids_path() -> Path:
    return CORPUS_DIR / "eval_tier_ids.json"


def arbitration_rejected_path() -> Path:
    return CORPUS_DIR / "arbitration-rejected.json"


def pdf_path(key: str) -> Path:
    return pdf_dir() / f"{key}.pdf"


def expected_json_path(key: str) -> Path:
    return ground_truth_dir() / f"{key}.expected.json"


def manifest_key(entry: dict) -> str:
    return Path(entry["filename"]).stem


def load_manifest_books() -> list[dict]:
    return json.loads(manifest_path().read_text(encoding="utf-8"))["books"]
```

- [ ] **Step 4: Run to verify all tests pass**

```bash
uv run pytest tests/test_corpus.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/dnb_toc_ground_truth/corpus.py tests/test_corpus.py
git commit -m "feat: add single-corpus path helpers (pdf/ vs ground-truth/ split)"
```

---

## Phase 5: CLI scripts

### Task 10: Port `cli/fetch_corpus.py`

**Files:**
- Create: `NEW/cli/fetch_corpus.py` (copy of `WT/evaluation/scripts/fetch_dnb_toc_corpus.py`)
- Create: `NEW/tests/test_fetch_corpus.py` (copy of `WT/tests/test_fetch_dnb_toc_corpus.py`)

- [ ] **Step 1: Copy and adapt**

```bash
cp /Users/cboulanger/Code/chapter-segmentation-dnb-migration/evaluation/scripts/fetch_dnb_toc_corpus.py \
   /Users/cboulanger/Code/dnb-toc-ground-truth/cli/fetch_corpus.py
```

Remove the `sys.path.insert` hack (not needed — `dnb_toc_ground_truth` is an installed package, same reasoning `generate_dnb_toc_ground_truth.py` never needed it):

```python
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from evaluation.harness import corpus_dir
```

becomes:

```python
from dnb_toc_ground_truth import corpus
```

Remove the now-unused `import sys`. This script has no multi-corpus concept in the new repo (it only ever wrote `dnb-toc-only`), so every `corpus_dir(_CORPUS_NAME)` call becomes `corpus.corpus_dir()`, and `_CORPUS_NAME = "dnb-toc-only"` is deleted. The PDF write target changes from the corpus root to the new `pdf/` subdirectory — in `_acquire_record`, change:

```python
(cdir / filename).write_bytes(response.content)
lobid_cache_dir = cdir / _LOBID_CACHE_DIRNAME
lobid_cache_dir.mkdir(parents=True, exist_ok=True)
(lobid_cache_dir / f"{key}.lobid.json").write_text(
```

to:

```python
corpus.pdf_dir().mkdir(parents=True, exist_ok=True)
(corpus.pdf_dir() / filename).write_bytes(response.content)
corpus.lobid_cache_dir().mkdir(parents=True, exist_ok=True)
(corpus.lobid_cache_dir() / f"{key}.lobid.json").write_text(
```

Delete the now-unused `_LOBID_CACHE_DIRNAME` constant. In `main()`, change:

```python
cdir = corpus_dir(_CORPUS_NAME)
cdir.mkdir(parents=True, exist_ok=True)
manifest_path = args.manifest_path or (cdir / "manifest.json")
```

to:

```python
corpus.corpus_dir().mkdir(parents=True, exist_ok=True)
manifest_path = args.manifest_path or corpus.manifest_path()
```

Update the module docstring's `evaluation/corpus/dnb-toc-only/` mentions to `data/corpus/pilot/`.

- [ ] **Step 2: Copy the test file and fix imports**

```bash
cp /Users/cboulanger/Code/chapter-segmentation-dnb-migration/tests/test_fetch_dnb_toc_corpus.py \
   /Users/cboulanger/Code/dnb-toc-ground-truth/tests/test_fetch_corpus.py
```

Change the import block:

```python
from evaluation.scripts.fetch_dnb_toc_corpus import (
    _ChunkStreamReader,
    _acquire_record,
    _append_book,
    _ensure_manifest_shell,
    _iter_dump_records_from_chunks,
```

to (module path only; the rest of that `import (...)` list is unchanged):

```python
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "cli"))
from fetch_corpus import (
    _ChunkStreamReader,
    _acquire_record,
    _append_book,
    _ensure_manifest_shell,
    _iter_dump_records_from_chunks,
```

(`cli/` scripts are not part of the installed `dnb_toc_ground_truth` package — they're entry points, not library modules — so tests importing them directly need `cli/` on `sys.path`. Use a clean form instead of the inline `__import__` above:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cli"))

from fetch_corpus import (
    _ChunkStreamReader,
    _acquire_record,
    _append_book,
    _ensure_manifest_shell,
    _iter_dump_records_from_chunks,
    ...
)
```

at the top of the test file, replacing the `sys`/`Path` imports already present there if any duplicate.) Any test that asserts on where `_acquire_record` writes the PDF needs updating to check `corpus.pdf_dir() / filename` instead of `cdir / filename` — read the full original test file's fixture setup (it builds a temp `cdir` and passes it directly into `_acquire_record`) and adjust to pass a temp `pdf_dir`/`lobid_cache_dir` pair consistent with the new `_acquire_record` signature from Step 1.

- [ ] **Step 3: Run and commit**

```bash
cd /Users/cboulanger/Code/dnb-toc-ground-truth
uv run pytest tests/test_fetch_corpus.py -v
git add cli/fetch_corpus.py tests/test_fetch_corpus.py
git commit -m "feat: port cli/fetch_corpus.py"
```

### Task 11: Port `cli/select_eval_sample.py`

**Files:**
- Create: `NEW/cli/select_eval_sample.py` (copy of `WT/evaluation/scripts/select_dnb_toc_eval_sample.py`)
- Create: `NEW/tests/test_select_eval_sample.py` (copy of `WT/tests/test_select_dnb_toc_eval_sample.py`)

- [ ] **Step 1: Copy and adapt**

```bash
cp /Users/cboulanger/Code/chapter-segmentation-dnb-migration/evaluation/scripts/select_dnb_toc_eval_sample.py \
   /Users/cboulanger/Code/dnb-toc-ground-truth/cli/select_eval_sample.py
```

Change:

```python
from evaluation.harness import corpus_dir, load_manifest_books

_CORPUS_NAME = "dnb-toc-only"
_DEFAULT_SEED = 20260815


def manifest_key(entry: dict) -> str:
    return Path(entry["filename"]).stem
```

to:

```python
from dnb_toc_ground_truth import corpus

_DEFAULT_SEED = 20260815
```

(drop the local `manifest_key` — use `corpus.manifest_key` everywhere it was called). In `main()`, change:

```python
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
```

to:

```python
books = corpus.load_manifest_books()
lobid_records = {}
for entry in books:
    key = corpus.manifest_key(entry)
    lobid_path = corpus.lobid_cache_dir() / f"{key}.lobid.json"
    if lobid_path.exists():
        lobid_records[key] = json.loads(lobid_path.read_text(encoding="utf-8"))

selected = stratify_sample(books, lobid_records, args.sample_size, args.seed)
output_path = corpus.eval_tier_ids_path()
```

And every remaining `manifest_key(...)` call inside `stratify_sample`/`_decade` callers becomes `corpus.manifest_key(...)`.

- [ ] **Step 2: Copy the test file and fix imports**

```bash
cp /Users/cboulanger/Code/chapter-segmentation-dnb-migration/tests/test_select_dnb_toc_eval_sample.py \
   /Users/cboulanger/Code/dnb-toc-ground-truth/tests/test_select_eval_sample.py
```

Change:

```python
from evaluation.scripts.select_dnb_toc_eval_sample import _decade, manifest_key, stratify_sample
```

to:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cli"))

from dnb_toc_ground_truth.corpus import manifest_key
from select_eval_sample import _decade, stratify_sample
```

- [ ] **Step 3: Run and commit**

```bash
cd /Users/cboulanger/Code/dnb-toc-ground-truth
uv run pytest tests/test_select_eval_sample.py -v
git add cli/select_eval_sample.py tests/test_select_eval_sample.py
git commit -m "feat: port cli/select_eval_sample.py"
```

### Task 12: Port `cli/arbitrate.py`

**Files:**
- Create: `NEW/cli/arbitrate.py` (copy of `WT/evaluation/scripts/arbitrate_dnb_toc.py`)
- Create: `NEW/tests/test_arbitrate.py` (copy of `WT/tests/test_arbitrate_dnb_toc.py`)

- [ ] **Step 1: Copy and adapt**

```bash
cp /Users/cboulanger/Code/chapter-segmentation-dnb-migration/evaluation/scripts/arbitrate_dnb_toc.py \
   /Users/cboulanger/Code/dnb-toc-ground-truth/cli/arbitrate.py
```

Change:

```python
from chapter_segmentation.segmentation import TocEntry
from evaluation.dnb_toc_matching import diff_toc_entries
from evaluation.dnb_toc_vision import load_cached_kind, load_cached_llm_entries, versioned_cache_dir
from evaluation.harness import corpus_dir, llm_cache_dir, load_manifest_books
from evaluation.scripts.select_dnb_toc_eval_sample import manifest_key

_CORPUS_NAME = "dnb-toc-only"
```

to:

```python
from dnb_toc_ground_truth import corpus
from dnb_toc_ground_truth.matching import diff_toc_entries
from dnb_toc_ground_truth.toc_entry import TocEntry
from dnb_toc_ground_truth.vision import load_cached_kind, load_cached_llm_entries, versioned_cache_dir
```

Replace every `corpus_dir(_CORPUS_NAME)` with `corpus.corpus_dir()`, `llm_cache_dir(_CORPUS_NAME)` with `corpus.llm_cache_dir()`, `load_manifest_books(_CORPUS_NAME)` with `corpus.load_manifest_books()`, and every bare `manifest_key(...)` call with `corpus.manifest_key(...)`. `_rejected_path` already builds its path as `cdir / "arbitration-rejected.json"` — replace that with `corpus.arbitration_rejected_path()` directly and drop the `cdir` parameter from `_rejected_path`/`_load_rejected_keys`/`reject_book` (they can call `corpus.arbitration_rejected_path()`/`corpus.corpus_dir()` internally instead of taking a `cdir` argument — simplifies every call site in `main()` too, since there's only ever one corpus). The one behavior-relevant path change: `_list`'s per-book `.expected.json` existence check and `reject_book`'s eventual arbitration write target move from `cdir / f"{key}.expected.json"` to `corpus.expected_json_path(key)`, and the printed PDF path in `format_book_report`'s caller (`cdir / f"{key}.pdf"`) becomes `corpus.pdf_path(key)`.

- [ ] **Step 2: Copy the test file and fix imports**

```bash
cp /Users/cboulanger/Code/chapter-segmentation-dnb-migration/tests/test_arbitrate_dnb_toc.py \
   /Users/cboulanger/Code/dnb-toc-ground-truth/tests/test_arbitrate.py
```

Change:

```python
from chapter_segmentation.segmentation import TocEntry
from evaluation.dnb_toc_vision import write_cached_llm_entries
from evaluation.scripts.arbitrate_dnb_toc import (
    _cached_kinds_for_book,
    _cached_models_for_book,
    books_needing_arbitration,
    format_book_report,
    reject_book,
)
```

to:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cli"))

from dnb_toc_ground_truth.toc_entry import TocEntry
from dnb_toc_ground_truth.vision import write_cached_llm_entries
from arbitrate import (
    _cached_kinds_for_book,
    _cached_models_for_book,
    books_needing_arbitration,
    format_book_report,
    reject_book,
)
```

Update any call to `reject_book(cdir, key, reason, ...)`/`books_needing_arbitration(cdir, cache_directory)` whose signature changed in Step 1 (dropped `cdir` params) to match the new no-`cdir` calling convention, using `tempfile`-based monkeypatching of `dnb_toc_ground_truth.corpus.CORPUS_DIR` (or equivalent — read how the original test isolates its temp corpus directory, likely via `unittest.mock.patch` on `corpus_dir`/`llm_cache_dir`, and patch `dnb_toc_ground_truth.corpus.CORPUS_DIR` the same way) rather than passing paths as arguments.

- [ ] **Step 3: Run and commit**

```bash
cd /Users/cboulanger/Code/dnb-toc-ground-truth
uv run pytest tests/test_arbitrate.py -v
git add cli/arbitrate.py tests/test_arbitrate.py
git commit -m "feat: port cli/arbitrate.py"
```

### Task 13: Rewrite `cli/generate_ground_truth.py`

**Files:**
- Create: `NEW/cli/generate_ground_truth.py`
- Create: `NEW/tests/test_generate_ground_truth.py`

This is the heaviest task: the per-book gating/locking/retry orchestration carries over almost unchanged, but endpoint resolution is completely rebuilt around `inference.py`'s generic N-model resolution instead of KISSKI auto-discovery, and gating now goes through `matching.gate_books` for N>=2 lists instead of `matching.gate_book` for exactly 2.

- [ ] **Step 1: Write `NEW/cli/generate_ground_truth.py`**

```python
"""Generates structured ground truth for the dnb-toc-only pilot corpus.
For every manifest book not held out in eval_tier_ids.json (see
select_eval_sample.py), not already carrying a ground-truth JSON file
(bulk-gated or arbitrated), and not permanently rejected
(arbitration-rejected.json), sends the book's TOC pages to every model
named via --use-vision/--use-text (resolved against --endpoints-file),
and writes a ground-truth file only when at least two of the resulting
reads agree well enough (dnb_toc_ground_truth.matching.gate_books,
>=0.90 agreement between the best-agreeing pair) -- see design spec
docs/superpowers/specs/2026-08-21-dnb-toc-ground-truth-extraction-design.md.
Books that don't clear the gate are skipped and reported, not partially
written -- run cli/arbitrate.py on them next.

Safe to run two invocations concurrently against the same checkout --
each book is claimed via a per-key lock file under .locks/ before either
process touches its cache or spends an API call on it.

    uv run python cli/generate_ground_truth.py --use-vision modelA,modelB --limit 50
    uv run python cli/generate_ground_truth.py --use-vision modelA --use-text modelC
    uv run python cli/generate_ground_truth.py --spot-check 30

--spot-check N does not generate anything -- instead it samples N books
that already passed the bulk-tier gate (i.e. "verified": false;
already-human-verified eval-tier entries are excluded) and walks through
a manual, terminal-driven visual Accept/Reject check against the real
PDF for each, then prints the measured accept rate as an estimate of the
gate's real precision.
"""

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Optional

from openai import RateLimitError

from dnb_toc_ground_truth import corpus
from dnb_toc_ground_truth.inference import (
    DEFAULT_CONFIG_FILENAME, DEFAULT_ENDPOINTS_FILENAME, ModelEndpoint,
    load_config, load_endpoint_entries, resolve_model_endpoints,
)
from dnb_toc_ground_truth.matching import gate_books, toc_entry_to_gt_dict
from dnb_toc_ground_truth.ocr import text_extract_toc_entries
from dnb_toc_ground_truth.toc_entry import TocEntry
from dnb_toc_ground_truth.vision import (
    load_cached_kind, load_cached_llm_entries, vision_extract_toc_entries, write_cached_llm_entries,
)

_RATE_LIMIT_WINDOW_ORDER = ("day", "hour", "minute")
_INLINE_RETRY_WINDOWS = frozenset({"hour", "minute"})


def _binding_rate_limit_window(headers) -> Optional[str]:
    """Which of a provider's optional `x-ratelimit-remaining-<window>`
    response headers is actually at 0 -- i.e. which window is the real
    reason this request was rejected. Returns the LONGEST zeroed window
    (day > hour > minute) when more than one is reported at 0, since
    that's the one whose reset actually gates recovery. None if no
    `remaining-*` header reports exactly "0" (a 429 for some other
    reason, or a provider that doesn't send these headers at all --
    every caller must treat that the same as an unclassifiable rate
    limit and fall back to blind backoff)."""
    if not headers:
        return None
    zeroed = {
        key.lower().rsplit("-", 1)[-1]
        for key, value in headers.items()
        if key.lower().startswith("x-ratelimit-remaining-") and value.strip() == "0"
    }
    for window in _RATE_LIMIT_WINDOW_ORDER:
        if window in zeroed:
            return window
    return None


def _retry_after_seconds(headers) -> Optional[float]:
    if not headers:
        return None
    value = headers.get("retry-after")
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


async def _call_with_retry(
    coro_fn, attempts: int = 6, base_delay: float = 2.0, rate_limit_delay: float = 20.0, sleep=asyncio.sleep,
):
    """Exponential backoff for a non-429 failure. A 429 instead schedules
    its retry from the response's own rate-limit headers when present
    (`retry-after` for the exact delay, `x-ratelimit-remaining-<window>`
    to identify which window is actually binding), falling back to a
    blind `rate_limit_delay * attempt_number` linear backoff when those
    headers are absent (not every OpenAI-compatible provider sends
    them). A 429 whose binding window is "day" gives up immediately
    instead of sleeping -- a daily quota, once exhausted, typically does
    not reset within one script invocation's realistic lifetime, so
    blind or even header-precise inline retrying for it just burns wall
    time. Re-invoking the script once the daily quota actually resets
    already skips every book with a cached/decided result, so nothing
    extra is lost by not waiting inline. `sleep` is injectable so tests
    don't actually wait."""
    last_exc: Optional[Exception] = None
    for attempt in range(attempts):
        try:
            return await coro_fn()
        except Exception as exc:  # noqa: BLE001 -- any failure here (network, parse) is retryable
            last_exc = exc
            if attempt >= attempts - 1:
                break
            if isinstance(exc, RateLimitError):
                response = getattr(exc, "response", None)
                headers = response.headers if response is not None else None
                window = _binding_rate_limit_window(headers)
                if window is not None and window not in _INLINE_RETRY_WINDOWS:
                    break
                retry_after = _retry_after_seconds(headers)
                delay = retry_after if retry_after is not None else rate_limit_delay * (attempt + 1)
            else:
                delay = base_delay * 2 ** attempt
            await sleep(delay)
    raise last_exc


_GATE_THRESHOLD_DEFAULT = 0.90


def _run_book_entries(
    key: str, entries_by_endpoint: list[list[TocEntry]], threshold: float,
) -> tuple[str, bool, str]:
    """Core per-book gating logic, given every endpoint's already-
    extracted TocEntry list -- kept separate from PDF/network I/O so
    it's directly unit-testable with synthetic entries. Returns (key,
    passed, reason); reason is "ok" on success, else why the book was
    skipped ("no_entries", "below_threshold")."""
    if all(not entries for entries in entries_by_endpoint):
        return key, False, "no_entries"
    passed, entries, _winning_pair = gate_books(entries_by_endpoint, threshold=threshold)
    if not passed:
        return key, False, "below_threshold"
    gt_path = corpus.expected_json_path(key)
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    gt_path.write_text(
        json.dumps(
            {"entries": [toc_entry_to_gt_dict(e) for e in entries], "verified": False, "source": "bulk_gate"},
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    return key, True, "ok"


_LOCK_STALE_AFTER_SECONDS = 1800.0


def _lock_path(key: str) -> Path:
    return corpus.locks_dir() / f"{key}.lock"


def _acquire_lock(key: str, *, stale_after: float = _LOCK_STALE_AFTER_SECONDS) -> bool:
    """Claims `key` for this process via an atomic exclusive file create
    -- safe against two separate generate_ground_truth.py invocations
    sharing the same checkout racing on the same book. A lock older than
    `stale_after` is assumed to belong to a crashed/killed process and
    is reclaimed by deleting it and retrying the exclusive create once."""
    lock_path = _lock_path(key)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        lock_path.touch(exist_ok=False)
        return True
    except FileExistsError:
        pass
    try:
        age = time.time() - lock_path.stat().st_mtime
    except FileNotFoundError:
        age = None
    if age is not None and age < stale_after:
        return False
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass
    try:
        lock_path.touch(exist_ok=False)
        return True
    except FileExistsError:
        return False


def _release_lock(key: str) -> None:
    try:
        _lock_path(key).unlink()
    except FileNotFoundError:
        pass


async def _run_book(
    key: str, pdf_path: Path, endpoints: list[ModelEndpoint], semaphore: asyncio.Semaphore,
    cache_directory: Path, threshold: float, *, sleep=asyncio.sleep,
) -> tuple[str, bool, str]:
    """Thin I/O wrapper around _run_book_entries -- calls
    vision_extract_toc_entries or text_extract_toc_entries (per each
    endpoint's own `.kind`) through the cache, then _call_with_retry on a
    miss, and delegates every endpoint's resulting entry list to
    _run_book_entries. Catches any exception and reports it as a
    failed-but-tuple-shaped result instead of letting it propagate -- one
    book's failure must never abort the rest of a long, unattended,
    budget-spending batch run."""
    if not _acquire_lock(key):
        return key, False, "locked_by_another_process"
    try:
        entries_by_endpoint: list[list[TocEntry]] = []
        for endpoint in endpoints:
            cached = load_cached_llm_entries(cache_directory, key, endpoint.model_id)
            if cached is not None and load_cached_kind(cache_directory, key, endpoint.model_id) == endpoint.kind:
                entries = cached
            else:
                async def _call(ep=endpoint):
                    async with semaphore:
                        if ep.kind == "text":
                            return await text_extract_toc_entries(pdf_path, ep.model_id, ep.client)
                        return await vision_extract_toc_entries(pdf_path, ep.model_id, ep.client)
                entries = await _call_with_retry(_call, sleep=sleep)
                if entries:
                    write_cached_llm_entries(cache_directory, key, endpoint.model_id, entries, kind=endpoint.kind)
            entries_by_endpoint.append(entries)
        return _run_book_entries(key, entries_by_endpoint, threshold)
    except Exception as exc:  # noqa: BLE001 -- must never let one book crash the whole batch
        print(f"[error] {key}: {exc}{_rate_limit_headers_suffix(exc)}")
        return key, False, f"error: {type(exc).__name__}"
    finally:
        _release_lock(key)


def _rate_limit_headers_suffix(exc: Exception) -> str:
    response = getattr(exc, "response", None)
    if not isinstance(exc, RateLimitError) or response is None:
        return ""
    relevant = {k: v for k, v in response.headers.items() if "ratelimit" in k.lower() or k.lower() == "retry-after"}
    if not relevant:
        return ""
    return " [" + ", ".join(f"{k}={v}" for k, v in sorted(relevant.items())) + "]"


async def _run_all(
    keys_and_paths: list[tuple[str, Path]], endpoints: list[ModelEndpoint], concurrency: int,
    cache_directory: Path, threshold: float,
) -> list[tuple[str, bool, str]]:
    semaphore = asyncio.Semaphore(concurrency)
    return list(await asyncio.gather(*[
        _run_book(key, path, endpoints, semaphore, cache_directory, threshold)
        for key, path in keys_and_paths
    ]))


def _is_stale_bulk_gate_entry(gt_path: Path) -> bool:
    """True if gt_path is a BULK-tier ("source": "bulk_gate") file
    missing the "skip" key on at least one entry -- the current
    extraction standard (verbatim per-line extraction plus a "skip"
    flag) always sets it, so its absence marks a file written to an
    older standard that should be regenerated. A "claude_arbitration"
    file is never touched here regardless of this check."""
    try:
        data = json.loads(gt_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    if data.get("source") != "bulk_gate":
        return False
    entries = data.get("entries", [])
    return bool(entries) and not all("skip" in e for e in entries)


def _still_needs_a_decision(book: dict, eval_tier_ids: set[str], rejected_ids: set[str]) -> bool:
    key = corpus.manifest_key(book)
    if key in eval_tier_ids or key in rejected_ids:
        return False
    gt_path = corpus.expected_json_path(key)
    if not gt_path.exists():
        return True
    return _is_stale_bulk_gate_entry(gt_path)


def _resolve_endpoints(args: argparse.Namespace, config: dict) -> list[ModelEndpoint]:
    """Resolves every requested vision/text model against
    --endpoints-file, config-file defaults filled in for any CLI flag
    left unset. Requires at least one --use-vision model; --use-text is
    optional. Raises SystemExit naming what's wrong on a user error (no
    vision model given at all)."""
    endpoints_file = Path(args.endpoints_file or config.get("endpoints_file", DEFAULT_ENDPOINTS_FILENAME))
    use_vision = args.use_vision or config.get("use_vision") or []
    use_text = args.use_text or config.get("use_text") or []
    if not use_vision:
        raise SystemExit("--use-vision is required (directly or via the config file's \"use_vision\" key)")
    entries = load_endpoint_entries(endpoints_file)
    vision_endpoints = resolve_model_endpoints(use_vision, "vision", entries)
    text_endpoints = resolve_model_endpoints(use_text, "text", entries) if use_text else []
    return vision_endpoints + text_endpoints


def _generate(args: argparse.Namespace, config: dict) -> int:
    eval_tier_path = corpus.eval_tier_ids_path()
    eval_tier_ids = set(json.loads(eval_tier_path.read_text(encoding="utf-8"))) if eval_tier_path.exists() else set()
    rejected_path = corpus.arbitration_rejected_path()
    rejected_ids = (
        {entry["key"] for entry in json.loads(rejected_path.read_text(encoding="utf-8"))["rejected"]}
        if rejected_path.exists() else set()
    )

    books = corpus.load_manifest_books()
    eligible = [b for b in books if _still_needs_a_decision(b, eval_tier_ids, rejected_ids)]
    limit = args.limit if args.limit is not None else config.get("limit")
    if limit is not None:
        eligible = eligible[:limit]
    candidates = [
        (corpus.manifest_key(b), corpus.pdf_path(corpus.manifest_key(b)))
        for b in eligible if corpus.pdf_path(corpus.manifest_key(b)).exists()
    ]
    missing_pdf_count = len(eligible) - len(candidates)

    endpoints = _resolve_endpoints(args, config)
    concurrency = args.concurrency if args.concurrency is not None else config.get("concurrency", 4)
    threshold = args.gate_threshold if args.gate_threshold is not None else config.get("gate_threshold", _GATE_THRESHOLD_DEFAULT)

    results = asyncio.run(_run_all(candidates, endpoints, concurrency, corpus.llm_cache_dir(), threshold))
    passed = [r for r in results if r[1]]
    by_reason: dict[str, int] = {}
    for _, ok, reason in results:
        if not ok:
            by_reason[reason] = by_reason.get(reason, 0) + 1
    print("Endpoints used: " + ", ".join(f"{e.kind}={e.label}" for e in endpoints))
    print(f"{len(passed)}/{len(results)} books passed the gate and got ground-truth JSON written.")
    for reason, count in sorted(by_reason.items()):
        print(f"  {count} skipped: {reason}")
    if missing_pdf_count:
        print(f"  {missing_pdf_count} skipped: missing_pdf (not downloaded locally)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--limit", type=int, default=None, help="Process at most this many books (smoke-test convenience)")
    parser.add_argument("--concurrency", type=int, default=None, help="How many books to process concurrently (default: 4, or config file's \"concurrency\")")
    parser.add_argument(
        "--spot-check", type=int, default=None, metavar="N",
        help="Instead of generating, sample N passing bulk-tier books and walk through a visual Accept/Reject check",
    )
    parser.add_argument(
        "--use-vision", type=lambda s: [m.strip() for m in s.split(",") if m.strip()], default=None, metavar="MODEL[,MODEL...]",
        help="Model id(s) to resolve against --endpoints-file for the VISION side -- required (directly or via config), at least one",
    )
    parser.add_argument(
        "--use-text", type=lambda s: [m.strip() for m in s.split(",") if m.strip()], default=None, metavar="MODEL[,MODEL...]",
        help="Model id(s) to resolve against --endpoints-file for the TEXT (OCR'd) side -- optional",
    )
    parser.add_argument("--endpoints-file", type=Path, default=None, help=f"Path to the endpoints file (default: {DEFAULT_ENDPOINTS_FILENAME}, or config file's \"endpoints_file\")")
    parser.add_argument("--config-file", type=Path, default=Path(DEFAULT_CONFIG_FILENAME), help=f"Path to the config file (default: {DEFAULT_CONFIG_FILENAME})")
    parser.add_argument("--gate-threshold", type=float, default=None, help="Whole-book agreement threshold, 0-1 (default: 0.90, or config file's \"gate_threshold\")")
    args = parser.parse_args()

    config = load_config(args.config_file)
    if args.spot_check is not None:
        return _spot_check(args.spot_check)
    return _generate(args, config)


def _spot_check(n: int) -> int:
    """Terminal-driven precision check: sample n books that passed the
    bulk-tier gate, print each one's PDF path and generated entries, and
    prompt for a manual Accept/Reject after visually opening the PDF --
    then report measured precision for the gate threshold. Only samples
    books whose "verified" field is False (bulk-tier, machine-gated)."""
    import random

    passing = []
    for p in sorted(corpus.ground_truth_dir().glob("*.expected.json")):
        gt = json.loads(p.read_text(encoding="utf-8"))
        if gt.get("verified") is False:
            passing.append(p.name.removesuffix(".expected.json"))
    sample = random.sample(passing, min(max(n, 0), len(passing)))
    accepted = 0
    for key in sample:
        gt = json.loads(corpus.expected_json_path(key).read_text(encoding="utf-8"))
        print(f"\n=== {key} ===\nPDF: {corpus.pdf_path(key)}")
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

- [ ] **Step 2: Write `NEW/tests/test_generate_ground_truth.py`**

Read `WT/tests/test_generate_dnb_toc_ground_truth.py` in full first — most of its coverage (rate-limit header parsing, `_call_with_retry`'s backoff/day-quota-gives-up behavior, `_run_book_entries`'s pass/fail/write-file behavior, lock acquire/release/staleness, `_is_stale_bulk_gate_entry`/`_still_needs_a_decision`) ports with only import-path changes, since none of that logic changed in Step 1 above. Port those tests verbatim with:

```python
from evaluation.scripts.generate_dnb_toc_ground_truth import (
    _acquire_lock, _binding_rate_limit_window, _call_with_retry, _is_stale_bulk_gate_entry,
    _release_lock, _retry_after_seconds, _run_book_entries, _still_needs_a_decision,
)
```

replaced by:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cli"))

from generate_ground_truth import (
    _acquire_lock, _binding_rate_limit_window, _call_with_retry, _is_stale_bulk_gate_entry,
    _release_lock, _retry_after_seconds, _run_book_entries, _still_needs_a_decision,
)
```

Any test that called `_run_book_entries(key, entries_a, entries_b, corpus_directory)` (the old 2-list signature) must be updated to the new `_run_book_entries(key, entries_by_endpoint, threshold)` signature — change `_run_book_entries(key, entries_a, entries_b, cdir)` call sites to `_run_book_entries(key, [entries_a, entries_b], 0.90)`, and drop any assertion that reads the written file from a passed-in `cdir` in favor of patching `dnb_toc_ground_truth.corpus.CORPUS_DIR` to a tempdir before the call (matching the pattern established in Task 12's port) and reading back via `corpus.expected_json_path(key)`.

Drop entirely: every test exercising `_pick_models`/`_select_best_models`/`_resolve_vision_endpoints`/`_resolve_endpoints` (the old KISSKI-discovery and env-var-alias resolution) — none of that exists anymore. Replace with new coverage for the rewritten `_resolve_endpoints`:

```python
import json
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cli"))
from generate_ground_truth import _resolve_endpoints
import argparse


def _args(**overrides) -> argparse.Namespace:
    defaults = dict(use_vision=None, use_text=None, endpoints_file=None)
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestResolveEndpoints(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.endpoints_path = Path(self._tmp.name) / ".endpoints"
        self.endpoints_path.write_text(json.dumps([
            {"url": "https://x.invalid/a", "key": "k1", "model": "model-a"},
            {"url": "https://x.invalid/b", "key": "k2", "model": "model-b"},
        ]))

    def tearDown(self):
        self._tmp.cleanup()

    def test_resolves_two_vision_models(self):
        args = _args(use_vision=["model-a", "model-b"], endpoints_file=self.endpoints_path)
        endpoints = _resolve_endpoints(args, {})
        self.assertEqual([e.model_id for e in endpoints], ["model-a", "model-b"])
        self.assertTrue(all(e.kind == "vision" for e in endpoints))

    def test_resolves_one_vision_one_text(self):
        args = _args(use_vision=["model-a"], use_text=["model-b"], endpoints_file=self.endpoints_path)
        endpoints = _resolve_endpoints(args, {})
        self.assertEqual([(e.model_id, e.kind) for e in endpoints], [("model-a", "vision"), ("model-b", "text")])

    def test_raises_without_any_vision_model(self):
        args = _args(endpoints_file=self.endpoints_path)
        with self.assertRaises(SystemExit):
            _resolve_endpoints(args, {})

    def test_falls_back_to_config_file_defaults(self):
        args = _args(endpoints_file=self.endpoints_path)
        config = {"use_vision": ["model-a", "model-b"]}
        endpoints = _resolve_endpoints(args, config)
        self.assertEqual([e.model_id for e in endpoints], ["model-a", "model-b"])


if __name__ == "__main__":
    unittest.main()
```

Merge this new class into the ported test file from the paragraph above (single `NEW/tests/test_generate_ground_truth.py`, not two files).

- [ ] **Step 3: Run and iterate to green**

```bash
cd /Users/cboulanger/Code/dnb-toc-ground-truth
uv run pytest tests/test_generate_ground_truth.py -v
```

Fix any mismatch between the ported tests' expectations and Step 1's actual implementation (signatures above are the source of truth — adjust tests, not the implementation, unless a test reveals a genuine bug in Step 1's code).

- [ ] **Step 4: Commit**

```bash
git add cli/generate_ground_truth.py tests/test_generate_ground_truth.py
git commit -m "feat: rewrite cli/generate_ground_truth.py for generic N-model endpoints"
```

---

## Phase 6: Documentation and packaging

### Task 14: Write `data/corpus/pilot/README.md`

**Files:**
- Create: `NEW/data/corpus/pilot/README.md`

Read `WT/evaluation/corpus/dnb-toc-only/README.md` in full and port it with these substitutions applied throughout: `evaluation/corpus/dnb-toc-only/` → `data/corpus/pilot/`, `dnb-toc-only` (as a corpus-name reference) → `pilot`, `src/chapter_segmentation/segmentation.py` (the `TocEntry.skip` docstring pointer) → `src/dnb_toc_ground_truth/toc_entry.py`, and any `docs/superpowers/specs/2026-08-1{4,5,6}-dnb-toc-*-design.md` cross-references → the same filenames under this repo's own `docs/superpowers/specs/` (ported in Task 17).

- [ ] **Step 1: Port the file** with the substitutions above applied.

- [ ] **Step 2: Commit**

```bash
cd /Users/cboulanger/Code/dnb-toc-ground-truth
git add data/corpus/pilot/README.md
git commit -m "docs: port pilot corpus README"
```

### Task 15: Write `cli/README.md`

**Files:**
- Create: `NEW/cli/README.md`

Same convention as `chapter-segmentation`'s `evaluation/scripts/README.md`: a one-line description plus a full `--help` dump for every script, alphabetically.

- [ ] **Step 1: Generate the `--help` dump for each script**

```bash
cd /Users/cboulanger/Code/dnb-toc-ground-truth
for f in fetch_corpus generate_ground_truth arbitrate select_eval_sample; do
  echo "## \`$f.py\`"; echo; echo '```'; uv run python cli/$f.py --help; echo '```'; echo
done
```

- [ ] **Step 2: Write `NEW/cli/README.md`**, header plus the four generated blocks:

```markdown
# CLI scripts reference

One-line description plus a full `--help` dump for every script in this
directory, alphabetically. Regenerate an entry by running
`uv run python cli/<name>.py --help` whenever that script's arguments
change.

## `arbitrate.py`

Surfaces books whose model reads didn't clear `generate_ground_truth.py`'s
agreement gate, so a human/Claude session can arbitrate the conflict
directly -- see `CLAUDE.md`'s "Arbitrating below-gate books". This
script only REPORTS and records rejections; it never decides.

<paste the `arbitrate` --help block from Step 1 here>

## `fetch_corpus.py`

Acquires real DNB-scanned table-of-contents PDFs via the lobid-resources
API into `data/corpus/pilot/pdf/`.

<paste the `fetch_corpus` --help block from Step 1 here>

## `generate_ground_truth.py`

Generates bulk-tier ground truth by sending each book's TOC pages to
every model named via `--use-vision`/`--use-text`, writing a
ground-truth file only when at least two of the resulting reads agree
well enough.

<paste the `generate_ground_truth` --help block from Step 1 here>

## `select_eval_sample.py`

Selects a stratified held-out eval-tier sample, so it isn't accidentally
dominated by one publication era or language.

<paste the `select_eval_sample` --help block from Step 1 here>
```

- [ ] **Step 3: Commit**

```bash
git add cli/README.md
git commit -m "docs: add cli/README.md --help reference"
```

### Task 16: Write `.endpoints.dist` and `.config.dist`

**Files:**
- Create: `NEW/.endpoints.dist`
- Create: `NEW/.config.dist`

- [ ] **Step 1: Write `NEW/.endpoints.dist`**

```json
[
  {
    "framework": "vLLM",
    "framework_args": "--model=mistralai/Pixtral-12B-2409 --tensor-parallel-size=2 --trust-remote-code",
    "host": "10.179.7.234:24100",
    "key": "REPLACE_WITH_REAL_API_KEY",
    "status": "Running",
    "url": "https://your-inference-provider.example/session-a"
  },
  {
    "framework": "vLLM",
    "framework_args": "--model=Qwen/Qwen3-Omni-30B-A3B-Instruct --tensor-parallel-size=2 --trust-remote-code",
    "host": "10.179.7.235:24100",
    "key": "REPLACE_WITH_REAL_API_KEY",
    "status": "Running",
    "url": "https://your-inference-provider.example/session-b"
  }
]
```

- [ ] **Step 2: Write `NEW/.config.dist`**

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

- [ ] **Step 3: Commit**

```bash
git add .endpoints.dist .config.dist
git commit -m "docs: add example .endpoints.dist and .config.dist"
```

### Task 17: Port docs (specs, plans, history)

**Files:**
- Create: `NEW/docs/superpowers/specs/*.md`, `NEW/docs/superpowers/plans/*.md` (copies of the files listed in the design spec's "Moves to dnb-toc-ground-truth" → Docs section)
- Create: `NEW/docs/history.md` (copy of `WT/evaluation/experiments/dnb-toc-ground-truth.md`)
- Create: `NEW/docs/llm-inference-providers.md` (generalized copy of `WT/evaluation/hpc/llm-mpcdf.md`)

- [ ] **Step 1: Copy the specs and plans verbatim**

```bash
cd /Users/cboulanger/Code/chapter-segmentation-dnb-migration
for f in \
  2026-08-14-dnb-toc-corpus-acquisition.md \
  2026-08-14-dnb-toc-corpus-acquisition-design.md \
  2026-08-15-dnb-toc-ground-truth-generation.md \
  2026-08-15-dnb-toc-ground-truth-generation-design.md \
  2026-08-15-dnb-toc-corpus-corrections.md \
  2026-08-15-dnb-toc-ground-truth-and-consumers-design.md \
  2026-08-16-dnb-toc-uniform-ocr-design.md \
  2026-08-16-dnb-toc-arbitration.md \
  2026-08-16-dnb-toc-arbitration-design.md \
  2026-08-16-dnb-toc-vision-extraction.md \
  2026-08-18-inference-endpoint-abstraction.md \
  2026-08-18-inference-endpoint-abstraction-design.md \
  2026-08-20-dnb-toc-vision-text-pairing-plan.md \
  2026-08-20-dnb-toc-vision-text-pairing-design.md \
  ; do
  if [ -f "docs/superpowers/plans/$f" ]; then
    cp "docs/superpowers/plans/$f" /Users/cboulanger/Code/dnb-toc-ground-truth/docs/superpowers/plans/
  fi
  if [ -f "docs/superpowers/specs/$f" ]; then
    cp "docs/superpowers/specs/$f" /Users/cboulanger/Code/dnb-toc-ground-truth/docs/superpowers/specs/
  fi
done
```

- [ ] **Step 2: Copy `docs/history.md`**

```bash
cp /Users/cboulanger/Code/chapter-segmentation-dnb-migration/evaluation/experiments/dnb-toc-ground-truth.md \
   /Users/cboulanger/Code/dnb-toc-ground-truth/docs/history.md
```

Edit the new file's opening paragraph (which currently says "not a result of the main chapter-segmentation workflow... see evaluation/RESULTS.md for that") to drop the now-meaningless cross-repo comparison — replace the first paragraph with:

```markdown
# dnb-toc-ground-truth history

Full write-up for every superseded run and diagnosis behind this
project's ground-truth-generation pipeline -- "Current status" below is
expected to go stale and be rewritten as the pipeline changes or more of
the corpus gets ground truth; the sections beneath it hold the full
history so the reasoning and dead ends behind the current numbers aren't
lost.
```

Leave everything from `## Current status` onward unchanged (it's a faithful historical record regardless of which repo it lives in).

- [ ] **Step 3: Copy and generalize the endpoint-provider notes**

```bash
cp /Users/cboulanger/Code/chapter-segmentation-dnb-migration/evaluation/hpc/llm-mpcdf.md \
   /Users/cboulanger/Code/dnb-toc-ground-truth/docs/llm-inference-providers.md
```

Edit `NEW/docs/llm-inference-providers.md`: replace every `--endpoint`/`--config-file` CLI-flag mention with `--use-vision`/`--use-text`/`--endpoints-file`, replace `<ALIAS>_BASE_URL`/`<ALIAS>_API_KEY`/`<ALIAS>_MODEL` env-var mentions (the mechanism this repo no longer has) with a pointer to `.endpoints`/`.endpoints.dist`, and remove the sentence "integrating it as a second inference endpoint alongside KISSKI" (no KISSKI in this repo at all) — replace with "as an inference endpoint for `cli/generate_ground_truth.py`". The empirical MPCDF-specific findings (endpoint URL convention, dashboard "Running" ≠ ready, confirmed-working/failing models, image-tag guidance) are unchanged, since they're true regardless of which repo calls the endpoint.

- [ ] **Step 4: Commit**

```bash
cd /Users/cboulanger/Code/dnb-toc-ground-truth
git add docs/
git commit -m "docs: port specs, plans, and history from chapter-segmentation"
```

### Task 18: Write `CLAUDE.md`

**Files:**
- Create: `NEW/CLAUDE.md`

Read `WT/evaluation/CLAUDE.md`'s "Arbitrating below-gate dnb-toc-only books" section in full (it's the source of truth for every step below) and port it with these substitutions: `arbitrate_dnb_toc.py` → `cli/arbitrate.py`, `generate_dnb_toc_ground_truth.py` → `cli/generate_ground_truth.py`, `evaluation/corpus/dnb-toc-only/<key>.expected.json` → `data/corpus/pilot/ground-truth/<key>.expected.json`, `evaluation/corpus/dnb-toc-only/llm-cache/<key>.<model>.json` → `data/corpus/pilot/llm-cache/<key>.<model>.json`, `evaluation/corpus/dnb-toc-only/arbitration-rejected.json` → `data/corpus/pilot/arbitration-rejected.json`, `TocEntry.skip`'s docstring in `src/chapter_segmentation/segmentation.py` → `src/dnb_toc_ground_truth/toc_entry.py`, and the design-spec cross-reference path stays the same relative form (`docs/superpowers/specs/2026-08-16-dnb-toc-arbitration-design.md`, now under this repo's own `docs/`).

- [ ] **Step 1: Write `NEW/CLAUDE.md`**

```markdown
# Arbitrating below-gate books

`cli/generate_ground_truth.py`'s agreement gate discards a book outright
when no pair of its resolved endpoints' reads agrees well enough (below
0.90 agreement) or fewer than two endpoints produce usable output -- but
it never deletes any endpoint's cached raw extraction
(`data/corpus/pilot/llm-cache/<key>.<model>.json`). Rather than
re-running the whole book from scratch or leaving it discarded, walk
through the following after a generation run leaves books below the
gate (design spec `docs/superpowers/specs/2026-08-16-dnb-toc-arbitration-design.md`):

1. List every book still needing a decision:

   ```bash
   uv run python cli/arbitrate.py
   ```

   This prints, per book: its title and PDF path, every cached model's
   entry count, and (for exactly two cached models) their agreement rate
   plus every entry each side found that the other didn't -- or, if only
   one model produced usable output, that model's full list with a note
   to verify it directly.

2. For each book, read the printed diff. The disagreement patterns found
   in practice so far (`docs/history.md`'s "Current status") usually
   make the right call obvious from the text alone: one side dropping
   real content, one side including front/back matter or a part-divider
   that should have been skipped, a two-line title wrongly split into
   two entries, or a deeply nested TOC segmented at different
   granularities.

3. When the text alone doesn't settle it, open the book's actual TOC
   page images directly: use the `Read` tool on the PDF with a `pages`
   parameter (1-based viewer pages).

4. Write the final `data/corpus/pilot/ground-truth/<key>.expected.json`
   yourself -- same schema as a passing book
   (`{"entries": [...], "verified": true, "source": "claude_arbitration"}`,
   each entry via `dnb_toc_ground_truth.matching.toc_entry_to_gt_dict`),
   but with `"verified": true` rather than `false`: unlike the bulk-tier
   gate's own output, this went through direct scrutiny (including the
   images, when needed) -- excluded from `_spot_check`'s sampling pool
   going forward. The `"source": "claude_arbitration"` field (vs. the
   bulk gate's own `"source": "bulk_gate"`) records that this entry's
   ground truth came from an arbitrated review, not the automated
   agreement gate.

   **Transcribe every printed line, not just the ones you'd call real
   chapters** -- part/section dividers and front/back matter (preface,
   bibliography, index, ...) get their own entry too, with
   `"skip": true`; real chapters get `"skip": false` (see `TocEntry.skip`'s
   docstring in `src/dnb_toc_ground_truth/toc_entry.py`).

5. If a book is genuinely unrecoverable (every model hallucinates, the
   scan itself is too degraded to read even directly), record that
   instead of leaving it to resurface every run:

   ```bash
   uv run python cli/arbitrate.py reject <key> "<short reason>"
   ```

   This writes to the committed `data/corpus/pilot/arbitration-rejected.json`
   -- refuses (rather than silently overwriting) if `<key>` is already
   present, so re-running this step is safe.
```

- [ ] **Step 2: Commit**

```bash
cd /Users/cboulanger/Code/dnb-toc-ground-truth
git add CLAUDE.md
git commit -m "docs: add CLAUDE.md arbitration workflow"
```

### Task 19: Write the top-level `README.md`

**Files:**
- Modify: `NEW/README.md` (replace the Task 1 stub)

Implements the design spec's "Top-level README.md" section verbatim.

- [ ] **Step 1: Write `NEW/README.md`**

```markdown
# dnb-toc-ground-truth

A pilot case for generating structured, machine-checkable ground truth
from openly available data -- the Deutsche Nationalbibliothek's
CC0-licensed "Kataloganreicherung" table-of-contents scans -- using
independent LLM reads gated against each other for agreement, with
human/Claude arbitration for the disagreements.

The output (`data/corpus/pilot/ground-truth/*.expected.json`) is meant
as an input to *other* pipelines, not an end in itself: fine-tuning a
smaller structured-extraction model, benchmarking chapter/TOC extraction
heuristics, or training a lightweight classifier could all consume this
corpus without depending on this repo's own LLM pipeline. The LLM-based
generation pipeline in this repo is the means of producing that data,
not the point -- the point is the ground-truth data itself, general
enough to feed pipelines this repo doesn't build.

## Setup

1. Install dependencies:

   ```bash
   uv sync
   ```

2. External binaries, on `PATH` or via a `--<tool>-bin` flag / env var
   (same convention as `PDFALTO_BIN` below): `ocrmypdf` (for the
   OCR-text extraction path -- `brew install ocrmypdf`), and a sibling
   [`pdfalto`](https://github.com/kermitt2/pdfalto) checkout for ALTO
   reconstruction (not vendored, not installable via brew -- build it
   next to this repo and point `PDFALTO_BIN` at the resulting binary).

3. Copy the example credential/config files and fill in real values:

   ```bash
   cp .endpoints.dist .endpoints
   cp .config.dist .config
   ```

   `.endpoints` lists every inference endpoint you can call (see
   `docs/llm-inference-providers.md`); `.config` sets defaults for
   `cli/generate_ground_truth.py`'s flags (which models to use, gate
   threshold, concurrency) so you don't have to repeat them on every
   invocation.

4. Smoke-check the install before pointing anything at a real endpoint:

   ```bash
   uv run python cli/fetch_corpus.py --help
   uv run python cli/generate_ground_truth.py --help
   ```

5. See `cli/README.md` for the full flag reference of every script, and
   `data/corpus/pilot/README.md` for the corpus's current size and
   status.
```

- [ ] **Step 2: Commit**

```bash
cd /Users/cboulanger/Code/dnb-toc-ground-truth
git add README.md
git commit -m "docs: write top-level README (purpose + setup)"
```

---

## Phase 7: Full test suite and data transfer

### Task 20: Run the full test suite and fix any remaining failures

**Files:** none new — this task fixes whatever Phase 1-6 left broken.

- [ ] **Step 1: Run everything**

```bash
cd /Users/cboulanger/Code/dnb-toc-ground-truth
uv run pytest -v
```

- [ ] **Step 2: Fix any import-path mismatch, signature drift between a ported test and Task 13's rewritten `generate_ground_truth.py`, or missing fixture left over from the mechanical ports in Phase 2/5.** Re-run after each fix.

- [ ] **Step 3: Confirm zero references to KISSKI remain anywhere in the new repo**

```bash
grep -ril kisski /Users/cboulanger/Code/dnb-toc-ground-truth || echo "clean"
```

Expected: `clean`.

- [ ] **Step 4: Commit whatever fixes were needed**

```bash
git add -A
git commit -m "test: fix remaining import/signature mismatches, full suite green"
```

### Task 21: Copy corpus data

**Files:** none tracked by git (this is a filesystem copy of gitignored/PDF content, per design spec "Migration mechanics" step 4)

- [ ] **Step 1: Copy PDFs into the new `pdf/` layout**

```bash
cp /Users/cboulanger/Code/chapter-segmentation-dnb-migration/evaluation/corpus/dnb-toc-only/*.pdf \
   /Users/cboulanger/Code/dnb-toc-ground-truth/data/corpus/pilot/pdf/
```

- [ ] **Step 2: Copy `.expected.json` files into the new `ground-truth/` layout**

```bash
cp /Users/cboulanger/Code/chapter-segmentation-dnb-migration/evaluation/corpus/dnb-toc-only/*.expected.json \
   /Users/cboulanger/Code/dnb-toc-ground-truth/data/corpus/pilot/ground-truth/
```

- [ ] **Step 3: Copy everything else (manifest, caches, locks, layout cache)**

```bash
SRC=/Users/cboulanger/Code/chapter-segmentation-dnb-migration/evaluation/corpus/dnb-toc-only
DST=/Users/cboulanger/Code/dnb-toc-ground-truth/data/corpus/pilot
cp "$SRC/manifest.json" "$DST/"
cp "$SRC/eval_tier_ids.json" "$DST/" 2>/dev/null || true
cp "$SRC/arbitration-rejected.json" "$DST/" 2>/dev/null || true
cp -R "$SRC/llm-cache" "$DST/"
cp -R "$SRC/.lobid-cache" "$DST/" 2>/dev/null || true
cp -R "$SRC/.locks" "$DST/" 2>/dev/null || true
cp -R "$SRC/.layout-cache" "$DST/" 2>/dev/null || true
```

- [ ] **Step 4: Verify counts match**

```bash
echo "old PDFs: $(ls /Users/cboulanger/Code/chapter-segmentation-dnb-migration/evaluation/corpus/dnb-toc-only/*.pdf | wc -l)"
echo "new PDFs: $(ls /Users/cboulanger/Code/dnb-toc-ground-truth/data/corpus/pilot/pdf/*.pdf | wc -l)"
echo "old ground truth: $(ls /Users/cboulanger/Code/chapter-segmentation-dnb-migration/evaluation/corpus/dnb-toc-only/*.expected.json | wc -l)"
echo "new ground truth: $(ls /Users/cboulanger/Code/dnb-toc-ground-truth/data/corpus/pilot/ground-truth/*.expected.json | wc -l)"
```

Expected: matching counts on both lines.

- [ ] **Step 5: Commit the tracked file (manifest.json only — everything else in `data/corpus/pilot/` beyond `README.md` and `manifest.json` is gitignored per Task 1's `.gitignore`, matching the old repo's own PDF/cache-gitignoring convention)**

```bash
cd /Users/cboulanger/Code/dnb-toc-ground-truth
git status  # confirm only manifest.json (and eval_tier_ids.json/arbitration-rejected.json if present) are untracked-but-not-ignored
git add data/corpus/pilot/manifest.json data/corpus/pilot/eval_tier_ids.json data/corpus/pilot/arbitration-rejected.json
git commit -m "data: transfer dnb-toc-only corpus (PDFs/caches copied, not committed)"
```

(If `eval_tier_ids.json`/`arbitration-rejected.json` don't exist in the source corpus, skip adding them — `git add` on a nonexistent path errors; add only what Step 3 actually copied.)

### Task 22: Run the real test suite once more against real data, then a dry smoke test

**Files:** none

- [ ] **Step 1: Re-run the full suite** (confirms nothing in Phase 1-6 silently assumed the old flat pdf+json-together layout)

```bash
cd /Users/cboulanger/Code/dnb-toc-ground-truth
uv run pytest -v
```

- [ ] **Step 2: Dry-run `select_eval_sample.py` against the real corpus** (pure local computation, no network/API cost)

```bash
uv run python cli/select_eval_sample.py --sample-size 75
```

Expected: prints `Wrote 75 eval-tier IDs to .../eval_tier_ids.json` with no traceback.

- [ ] **Step 3: Commit if `select_eval_sample.py` produced a fresh `eval_tier_ids.json`**

```bash
git add data/corpus/pilot/eval_tier_ids.json
git commit -m "data: regenerate eval_tier_ids.json against transferred corpus" --allow-empty
```

---

## Phase 8: Wire `chapter-segmentation`'s remaining pilots to the relocated corpus

### Task 23: Add sibling-checkout corpus override to the layout classifier and scan-noise scripts

**Files:**
- Modify: `WT/evaluation/scripts/evaluate_layout_toc_classifier.py`
- Modify: `WT/evaluation/scripts/measure_dnb_scan_noise_stats.py`
- Modify: `WT/evaluation/harness.py`

Implements design spec "Corpus access from chapter-segmentation after the move".

- [ ] **Step 1: Read `WT/evaluation/harness.py`'s `corpus_dir` function** (already read in full during brainstorming — it's `CORPUS_ROOT / corpus` where `CORPUS_ROOT = EVAL_DIR / "corpus"`). Add an override for exactly one corpus name:

In `WT/evaluation/harness.py`, change:

```python
def corpus_dir(corpus: str) -> Path:
    return CORPUS_ROOT / corpus
```

to:

```python
import os

_DNB_TOC_CORPUS_NAME = "dnb-toc-only"
_DNB_TOC_CORPUS_DIR_ENV_VAR = "DNB_TOC_CORPUS_DIR"
_DNB_TOC_CORPUS_DIR_DEFAULT = EVAL_DIR.parent.parent / "dnb-toc-ground-truth" / "data" / "corpus" / "pilot"


def corpus_dir(corpus: str) -> Path:
    # dnb-toc-only moved to the standalone dnb-toc-ground-truth repo
    # (2026-08-21 migration) -- everything under evaluation/corpus/ still
    # resolves normally; only this one corpus name is redirected to the
    # sibling checkout's data/corpus/pilot/, via DNB_TOC_CORPUS_DIR (same
    # override convention as PDFALTO_BIN) or the default sibling path.
    if corpus == _DNB_TOC_CORPUS_NAME:
        override = os.environ.get(_DNB_TOC_CORPUS_DIR_ENV_VAR)
        return Path(override) if override else _DNB_TOC_CORPUS_DIR_DEFAULT
    return CORPUS_ROOT / corpus
```

(Move the added `import os` up to the file's existing import block rather than leaving it inline, if `os` isn't already imported there — check the top of the file first.)

- [ ] **Step 2: Confirm `evaluate_layout_toc_classifier.py`/`measure_dnb_scan_noise_stats.py` need no further changes** — both already call `corpus_dir("dnb-toc-only")` (or iterate `--corpora`, which includes it) rather than constructing the path themselves, so Step 1's redirect is transparent to them. Read both files' corpus-path usage to confirm this before moving on; if either constructs a `dnb-toc-only`-specific path directly instead of going through `harness.corpus_dir`, add the same redirect logic locally in that file instead.

- [ ] **Step 3: Verify against the real relocated corpus**

```bash
cd /Users/cboulanger/Code/chapter-segmentation-dnb-migration
DNB_TOC_CORPUS_DIR=/Users/cboulanger/Code/dnb-toc-ground-truth/data/corpus/pilot \
  uv run python evaluation/scripts/measure_dnb_scan_noise_stats.py --corpus dnb-toc-only
```

Expected: runs against the relocated corpus's PDFs/manifest with no path errors (a `PDFALTO_BIN`-related failure, if `pdfalto` isn't set up in this environment, is a separate, expected pre-existing condition — not a regression from this change; note it and move on rather than trying to fix pdfalto setup here).

- [ ] **Step 4: Commit** (this modifies `WT`, i.e. the branch that will later merge/replace `OLD`'s state — see Phase 9)

```bash
cd /Users/cboulanger/Code/chapter-segmentation-dnb-migration
git add evaluation/harness.py
git commit -m "feat: redirect dnb-toc-only corpus_dir to the relocated sibling repo"
```

### Task 24: Repoint `evaluation/CLAUDE.md`'s dangling corpus cross-reference

**Files:**
- Modify: `WT/evaluation/CLAUDE.md`

- [ ] **Step 1: Find and update the cross-reference**

In `WT/evaluation/CLAUDE.md`'s Step 1 ("Transcribe the table of contents"), change:

```
look in
`evaluation/corpus/dnb-toc-only/manifest.json` (see
`evaluation/scripts/fetch_dnb_toc_corpus.py` --
```

to:

```
look in
`../dnb-toc-ground-truth/data/corpus/pilot/manifest.json` (a sibling
checkout of the standalone dnb-toc-ground-truth repo -- see
`DNB_TOC_CORPUS_DIR` in `evaluation/harness.py`'s `corpus_dir`; see also
`cli/fetch_corpus.py` in that repo --
```

(Read the surrounding paragraph in full first — this is one sentence inside a longer explanation, so adjust surrounding wording only as needed to keep the sentence grammatical after the substitution, without rewriting the rest of the paragraph.)

- [ ] **Step 2: Commit**

```bash
cd /Users/cboulanger/Code/chapter-segmentation-dnb-migration
git add evaluation/CLAUDE.md
git commit -m "docs: repoint dnb-toc-only cross-reference at the relocated repo"
```

---

## Phase 9: Checkpoint — real endpoints and smoke test

### Task 25: Stop and request real endpoints (NOT a subagent task — do this yourself, in the main session)

- [ ] **Step 1: Ask the user to start two model endpoints and provide the resulting `.endpoints` file content** (per the design spec's migration mechanics step 6). Do not proceed to Task 26 until this is provided.

- [ ] **Step 2: Once provided, write it to `/Users/cboulanger/Code/dnb-toc-ground-truth/.endpoints`** (never commit this file — it's gitignored per Task 1).

- [ ] **Step 3: Run a real smoke test**

```bash
cd /Users/cboulanger/Code/dnb-toc-ground-truth
uv run python cli/generate_ground_truth.py --use-vision <model-a>,<model-b> --limit 5
```

(substitute the two real model ids the user's endpoints actually serve). Confirm: no traceback, a sensible pass/skip summary printed, and — for at least one passed book — its `data/corpus/pilot/ground-truth/<key>.expected.json` has `"source": "bulk_gate"`, `"verified": false`, and non-empty `entries`, matching the schema `WT`'s old pipeline produced.

- [ ] **Step 4: Report the result to the user** and get explicit confirmation before proceeding to Phase 9's cleanup (Task 26) — this is the gate the design spec's "Completion criteria" requires before touching the old repo.

---

## Phase 10: Clean up `chapter-segmentation` (only after Task 25's smoke test is confirmed)

### Task 26: Remove migrated files from the worktree

**Files:** deletions only, all within `WT`

- [ ] **Step 1: Delete migrated code**

```bash
cd /Users/cboulanger/Code/chapter-segmentation-dnb-migration
git rm evaluation/dnb_toc_vision.py evaluation/dnb_toc_ocr.py evaluation/dnb_toc_matching.py
git rm evaluation/scripts/generate_dnb_toc_ground_truth.py evaluation/scripts/arbitrate_dnb_toc.py \
       evaluation/scripts/fetch_dnb_toc_corpus.py evaluation/scripts/select_dnb_toc_eval_sample.py
```

- [ ] **Step 2: Delete migrated tests**

```bash
git rm tests/test_dnb_toc_vision.py tests/test_dnb_toc_ocr.py tests/test_dnb_toc_matching.py \
       tests/test_arbitrate_dnb_toc.py tests/test_generate_dnb_toc_ground_truth.py \
       tests/test_fetch_dnb_toc_corpus.py tests/test_select_dnb_toc_eval_sample.py tests/test_kisski.py
```

- [ ] **Step 3: Delete the migrated corpus**

```bash
git rm -r evaluation/corpus/dnb-toc-only
```

- [ ] **Step 4: Delete migrated docs**

```bash
git rm evaluation/experiments/dnb-toc-ground-truth.md evaluation/hpc/llm-mpcdf.md
git rm docs/superpowers/plans/2026-08-14-dnb-toc-corpus-acquisition.md \
       docs/superpowers/specs/2026-08-14-dnb-toc-corpus-acquisition-design.md \
       docs/superpowers/plans/2026-08-15-dnb-toc-ground-truth-generation.md \
       docs/superpowers/specs/2026-08-15-dnb-toc-ground-truth-generation-design.md \
       docs/superpowers/plans/2026-08-15-dnb-toc-corpus-corrections.md \
       docs/superpowers/specs/2026-08-15-dnb-toc-ground-truth-and-consumers-design.md \
       docs/superpowers/specs/2026-08-16-dnb-toc-uniform-ocr-design.md \
       docs/superpowers/plans/2026-08-16-dnb-toc-arbitration.md \
       docs/superpowers/specs/2026-08-16-dnb-toc-arbitration-design.md \
       docs/superpowers/plans/2026-08-16-dnb-toc-vision-extraction.md \
       docs/superpowers/plans/2026-08-18-inference-endpoint-abstraction.md \
       docs/superpowers/specs/2026-08-18-inference-endpoint-abstraction-design.md \
       docs/superpowers/plans/2026-08-20-dnb-toc-vision-text-pairing-plan.md \
       docs/superpowers/specs/2026-08-20-dnb-toc-vision-text-pairing-design.md \
       docs/superpowers/specs/2026-08-21-dnb-toc-ground-truth-extraction-design.md \
       docs/superpowers/plans/2026-08-21-dnb-toc-ground-truth-extraction.md
```

(The last two — this very spec and plan — get removed too, since they document a migration that's now complete and their content lives on as the new repo's own `docs/superpowers/` history. Do this removal LAST, after every other step in this task has actually been executed using them as a checklist, not before.)

- [ ] **Step 5: Remove `evaluation/inference_endpoints.py`/`evaluation/kisski.py` from the "stays" list? No** — do not delete these; they're still used by `refresh_llm_cache.py` (confirmed in the design spec's "Stays" section). Skip this step; it exists only as an explicit reminder not to delete them by mistake.

- [ ] **Step 6: Remove `evaluation/CLAUDE.md`'s "Arbitrating below-gate dnb-toc-only books" section entirely** (its content now lives in the new repo's own `CLAUDE.md`, ported in Task 18) — open the file, find the section (a top-level `## Arbitrating below-gate dnb-toc-only books` heading through to the next `##` heading), delete it in full.

- [ ] **Step 7: Update `evaluation/experiments/README.md`**

Change:

```markdown
- [`dnb-toc-ground-truth.md`](dnb-toc-ground-truth.md) -- the
  two-independent-vision-model gate and arbitration tooling used to build
  ground truth for the `dnb-toc-only` corpus (a ground-truth-generation
  pipeline, not an accuracy experiment for `chapter_segmentation` itself).
```

to:

```markdown
- **dnb-toc-ground-truth** -- the LLM-gated ground-truth-generation
  pipeline for DNB table-of-contents scans has moved to its own
  standalone repository (`dnb-toc-ground-truth`), since it grew a real
  corpus, its own endpoint-configuration system, and no meaningful
  coupling to `chapter_segmentation` beyond a couple of vendored data
  types. See that repo's own `docs/history.md` for its experiment
  history.
```

- [ ] **Step 8: Verify the remaining test suite still passes**

```bash
cd /Users/cboulanger/Code/chapter-segmentation-dnb-migration
uv run pytest
```

Expected: PASS, with the deleted tests simply absent from collection (no import errors from any remaining file referencing something just deleted — grep to confirm):

```bash
grep -rn "dnb_toc_vision\|dnb_toc_ocr\|dnb_toc_matching\|generate_dnb_toc_ground_truth\|arbitrate_dnb_toc\|fetch_dnb_toc_corpus\|select_dnb_toc_eval_sample" \
  --include="*.py" evaluation/ tests/ src/ 2>/dev/null
```

Expected: no output (or only matches inside `evaluation/scripts/measure_dnb_scan_noise_stats.py`/`evaluate_layout_toc_classifier.py`'s own `--corpus dnb-toc-only`-style corpus-NAME string arguments, which are fine — the corpus name string itself doesn't move, only the module/script files did).

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "chore: remove dnb-toc-ground-truth pipeline (moved to its own repo)"
```

### Task 27: Finish the branch

- [ ] **Step 1: Invoke the finishing-a-development-branch skill** from the `WT` worktree to decide how `dnb-toc-ground-truth-wip`'s final state (now containing only the cleanup + sibling-checkout-wiring commits relevant to `chapter-segmentation`, since Phase 0-7's new-repo-building commits happened in a completely separate git repository at `NEW` and never touched this branch) should land in `chapter-segmentation`'s `main` — merge, PR, or otherwise, per the user's preference at that point.
