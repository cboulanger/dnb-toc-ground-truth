# Vision-LLM TOC extraction for dnb-toc-only Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `generate_dnb_toc_ground_truth.py`'s two-text-extractor
(regex heuristic + text-LLM) agreement gate with a two-vision-model
agreement gate that reads each book's page images directly, per
`docs/superpowers/specs/2026-08-16-dnb-toc-uniform-ocr-design.md`.

**Architecture:** A new `evaluation/dnb_toc_vision.py` module renders every
page of a `dnb-toc-only` PDF to PNG (via `pdftoppm`, no OCR) and sends them
in one chat-completion call to a vision-capable KISSKI model, returning
`list[TocEntry]` via a shared parsing helper extracted from
`segmentation.py`'s existing `llm_extract_toc_entries`. The GT-generation
script calls this twice per book (two independent models,
`qwen3-omni-30b-a3b-instruct` + `gemma-4-31b-it`, selected by a curated
regex allowlist since KISSKI doesn't expose a vision-capability flag) and
feeds both `TocEntry` lists into the existing `gate_book`/`align_toc_entries`
agreement gate — unchanged except for a `partial_ratio` title-matching
robustness fix and parameter renames (`heuristic`/`llm` → `a`/`b`, since
neither side is the regex heuristic anymore).

**Tech Stack:** Python, `openai` (`AsyncOpenAI`, already a dependency via
`evaluation/refresh_llm_cache.py`), `pypdf`, `pdftoppm` (poppler, already a
documented project dependency), `rapidfuzz`, `pytest`/`unittest`.

---

### Task 1: Extract `_toc_items_to_entries` shared parsing helper

**Files:**
- Modify: `src/chapter_segmentation/segmentation.py:586-644`
  (`llm_extract_toc_entries`)

This is a pure refactor — no behavior change — so the existing test suite
is the regression check; no new tests needed for this task.

- [ ] **Step 1: Extract the item-to-TocEntry parsing loop into a new module-level function**

In `src/chapter_segmentation/segmentation.py`, find `llm_extract_toc_entries`
(starts at line 586). Its body currently ends with:

```python
    try:
        items = await _extract_with_retry(prompt, llm_client)
    except Exception as exc:
        logger.warning(
            "llm_extract_toc_entries: giving up (%s)", _classify_llm_failure(exc), exc_info=True,
        )
        return []

    entries: list[TocEntry] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        if len(title) < 3:
            continue
        raw_authors = item.get("authors")
        # Guard against a malformed LLM response giving a plain string
        # instead of a list (e.g. "authors": "Jane Doe") -- iterating a
        # string yields one entry per character, silently corrupting
        # author-aware disambiguation downstream.
        authors = tuple(str(a).strip() for a in raw_authors if str(a).strip()) if isinstance(raw_authors, list) else ()
        printed = item.get("printed_page_number")
        if isinstance(printed, (int, float)):
            # Tolerate a model that ignores the string instruction and
            # returns a bare number anyway -- still unambiguous for the
            # arabic case.
            printed = str(int(printed))
        parsed_value = _parse_toc_page_number(printed.strip()) if isinstance(printed, str) else None
        # -1 is a sentinel for "unknown" (LLM returned null, an unparseable
        # value, or an implausible one, e.g. a roman numeral over
        # _ROMAN_PAGE_MAX_VALUE) -- never a real printed page number.
        printed_page_number = parsed_value if parsed_value is not None else -1
        printed_roman = parsed_value is not None and not printed.strip().isdigit()
        # source_page_index is a sentinel here -- unlike a regex-found entry,
        # an LLM-extracted entry has no single "the TOC line was on this
        # page" origin; the orchestration layer excludes the whole scanned
        # front/back-matter range instead (see _toc_scan_indices).
        entries.append(TocEntry(
            title=title, printed_page_number=printed_page_number, source_page_index=-1,
            authors=authors, printed_roman=printed_roman,
        ))
    return entries
```

Replace it with a call to a new shared function, and add that function
directly above `llm_extract_toc_entries`:

```python
def _toc_items_to_entries(items: list) -> list[TocEntry]:
    """Converts a parsed JSON array of {"title", "authors",
    "printed_page_number"} dicts -- the shape both llm_extract_toc_entries'
    text prompt and evaluation/dnb_toc_vision.py's image prompt ask an LLM
    to return -- into TocEntry objects, tolerating the malformed-response
    shapes a real model occasionally produces. Shared so both extraction
    paths parse identically instead of maintaining two copies of the same
    tolerance logic."""
    entries: list[TocEntry] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        if len(title) < 3:
            continue
        raw_authors = item.get("authors")
        # Guard against a malformed LLM response giving a plain string
        # instead of a list (e.g. "authors": "Jane Doe") -- iterating a
        # string yields one entry per character, silently corrupting
        # author-aware disambiguation downstream.
        authors = tuple(str(a).strip() for a in raw_authors if str(a).strip()) if isinstance(raw_authors, list) else ()
        printed = item.get("printed_page_number")
        if isinstance(printed, (int, float)):
            # Tolerate a model that ignores the string instruction and
            # returns a bare number anyway -- still unambiguous for the
            # arabic case.
            printed = str(int(printed))
        parsed_value = _parse_toc_page_number(printed.strip()) if isinstance(printed, str) else None
        # -1 is a sentinel for "unknown" (LLM returned null, an unparseable
        # value, or an implausible one, e.g. a roman numeral over
        # _ROMAN_PAGE_MAX_VALUE) -- never a real printed page number.
        printed_page_number = parsed_value if parsed_value is not None else -1
        printed_roman = parsed_value is not None and not printed.strip().isdigit()
        # source_page_index is a sentinel here -- unlike a regex-found entry,
        # an LLM-extracted entry has no single "the TOC line was on this
        # page" origin; the orchestration layer excludes the whole scanned
        # front/back-matter range instead (see _toc_scan_indices).
        entries.append(TocEntry(
            title=title, printed_page_number=printed_page_number, source_page_index=-1,
            authors=authors, printed_roman=printed_roman,
        ))
    return entries
```

And `llm_extract_toc_entries`'s tail becomes:

```python
    try:
        items = await _extract_with_retry(prompt, llm_client)
    except Exception as exc:
        logger.warning(
            "llm_extract_toc_entries: giving up (%s)", _classify_llm_failure(exc), exc_info=True,
        )
        return []
    return _toc_items_to_entries(items)
```

- [ ] **Step 2: Run the existing test suite to confirm no behavior change**

```bash
uv run pytest tests/test_segmentation.py -v -k TocEntries
```
Expected: all `TestLlmExtractTocEntries` tests still PASS unchanged.

- [ ] **Step 3: Commit**

```bash
git add src/chapter_segmentation/segmentation.py
git commit -m "refactor: extract _toc_items_to_entries for reuse by vision extraction"
```

---

### Task 2: New module `evaluation/dnb_toc_vision.py`

**Files:**
- Create: `evaluation/dnb_toc_vision.py`
- Test: `tests/test_dnb_toc_vision.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_dnb_toc_vision.py`:

```python
"""Unit/integration tests for evaluation/dnb_toc_vision.py -- vision-LLM
TOC extraction for dnb-toc-only, see design spec
docs/superpowers/specs/2026-08-16-dnb-toc-uniform-ocr-design.md section 3.
render_pages_to_images is integration-tested against a real (synthetic,
blank) PDF via the real pdftoppm binary -- poppler is already a documented
project dependency (evaluation/README.md). vision_extract_toc_entries is
tested with a mocked OpenAI-shaped client, no real network call."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from pypdf import PdfWriter

from evaluation.dnb_toc_vision import (
    _MAX_VISION_PAGES,
    render_pages_to_images,
    vision_extract_toc_entries,
)


def _make_pdf(path: Path, page_count: int) -> Path:
    writer = PdfWriter()
    for _ in range(page_count):
        writer.add_blank_page(width=200, height=200)
    with open(path, "wb") as f:
        writer.write(f)
    return path


class TestRenderPagesToImages(unittest.TestCase):
    def test_renders_one_png_per_page_in_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = _make_pdf(Path(tmp) / "book.pdf", 3)
            images = render_pages_to_images(pdf_path)
            self.assertEqual(len(images), 3)
            for image_bytes in images:
                self.assertTrue(image_bytes.startswith(b"\x89PNG"))

    def test_raises_on_a_nonexistent_pdf(self):
        with self.assertRaises(RuntimeError):
            render_pages_to_images(Path("/nonexistent/does-not-exist.pdf"))


def _fake_vision_client(response_text: str):
    message = MagicMock()
    message.content = response_text
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=response)
    return client


_VISION_RESPONSE = (
    '[{"title": "Einleitung", "authors": [], "printed_page_number": "9"}, '
    '{"title": "Zur Soziologie des Rechts", "authors": ["Jane Author"], "printed_page_number": "17"}]'
)


class TestVisionExtractTocEntries(unittest.IsolatedAsyncioTestCase):
    async def test_parses_response_into_toc_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = _make_pdf(Path(tmp) / "book.pdf", 1)
            client = _fake_vision_client(_VISION_RESPONSE)
            entries = await vision_extract_toc_entries(pdf_path, "some-model", client)
            self.assertEqual(len(entries), 2)
            self.assertEqual(entries[0].title, "Einleitung")
            self.assertEqual(entries[0].printed_page_number, 9)
            self.assertEqual(entries[1].authors, ("Jane Author",))

    async def test_sends_one_image_content_block_per_page(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = _make_pdf(Path(tmp) / "book.pdf", 2)
            client = _fake_vision_client("[]")
            await vision_extract_toc_entries(pdf_path, "some-model", client)
            messages = client.chat.completions.create.call_args.kwargs["messages"]
            content = messages[0]["content"]
            image_blocks = [c for c in content if c["type"] == "image_url"]
            self.assertEqual(len(image_blocks), 2)

    async def test_raises_on_malformed_json_instead_of_swallowing(self):
        # Unlike llm_extract_toc_entries (which catches internally and
        # returns [], making its own _call_with_retry wrapper dead code --
        # see generate_dnb_toc_ground_truth.py), vision_extract_toc_entries
        # deliberately propagates so the caller's retry wrapper is
        # meaningful.
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = _make_pdf(Path(tmp) / "book.pdf", 1)
            client = _fake_vision_client("not json at all")
            with self.assertRaises(Exception):
                await vision_extract_toc_entries(pdf_path, "some-model", client)

    async def test_raises_before_any_network_call_when_page_count_exceeds_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = _make_pdf(Path(tmp) / "book.pdf", _MAX_VISION_PAGES + 1)
            client = _fake_vision_client("[]")
            with self.assertRaises(ValueError):
                await vision_extract_toc_entries(pdf_path, "some-model", client)
            client.chat.completions.create.assert_not_called()
```

- [ ] **Step 2: Run the tests to verify they fail (module doesn't exist yet)**

```bash
uv run pytest tests/test_dnb_toc_vision.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'evaluation.dnb_toc_vision'`.

- [ ] **Step 3: Write the implementation**

Create `evaluation/dnb_toc_vision.py`:

```python
"""Vision-LLM TOC extraction for dnb-toc-only -- reads each book's page
images directly (no OCR, no text layer at all), per design spec
docs/superpowers/specs/2026-08-16-dnb-toc-uniform-ocr-design.md section 3.
dnb-toc-only's PDFs are pre-filtered to just their TOC pages during
acquisition (1-3 pages typically), so rendering every page unconditionally
is cheap and bounded -- this does NOT generalize to whole-book PDFs."""

import base64
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from chapter_segmentation._llm_json import parse_json_array
from chapter_segmentation.segmentation import TocEntry, _toc_items_to_entries

_VISION_TOC_EXTRACTION_PROMPT = """\
You are reading photographed/scanned page images of a book's table of \
contents. Some layouts use simple dotted leaders ("Title ..... 12"), \
others print the author's name on its own line above or below the title, \
or right-align the page number with no leader at all -- read the images \
directly rather than assuming one fixed layout.

Return ONLY a JSON array, one entry per real chapter -- skip \
acknowledgements, bibliography, index, and part-divider pages:
[{"title": "...", "authors": ["First Last", ...], "printed_page_number": "12"}]

printed_page_number is the page number exactly AS PRINTED on the page -- \
copy it verbatim, including roman numerals for front-matter chapters \
(e.g. "vii", not 7). If a chapter's printed page number is not visible, \
use null for printed_page_number. If authors are not identifiable, use an \
empty list."""

# Rendered image count this corpus's PDFs never exceed today (1-3 pages,
# per the acquisition pipeline's own TOC-only filtering) -- guards against
# silently building an arbitrarily large multi-image request if a
# mis-filtered outlier ever slips through (design spec section 5).
_MAX_VISION_PAGES = 20


def render_pages_to_images(pdf_path: Path, dpi: int = 200, pdftoppm_bin: str = "pdftoppm") -> list[bytes]:
    """Rasterizes every page of pdf_path to PNG bytes, in page order, via
    pdftoppm -- no OCR, no text extraction. Follows the same
    resolve_binary-able-external-tool and glob-then-sort conventions
    evaluation/scripts/clean_scanned_pdf.py already uses for pdftoppm."""
    with tempfile.TemporaryDirectory() as tmp:
        prefix = Path(tmp) / "page"
        result = subprocess.run(
            [pdftoppm_bin, "-r", str(dpi), "-png", str(pdf_path), str(prefix)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"pdftoppm failed on {pdf_path}: {result.stderr}")
        matches = sorted(Path(tmp).glob("page-*.png"))
        if not matches:
            raise RuntimeError(f"pdftoppm produced no output for {pdf_path}")
        return [p.read_bytes() for p in matches]


async def vision_extract_toc_entries(pdf_path: Path, model: str, client: Any, *, pdftoppm_bin: str = "pdftoppm") -> list[TocEntry]:
    """Renders every page of pdf_path and asks a vision-capable model
    (via an already-constructed openai.AsyncOpenAI-shaped `client`, model
    id `model`) to read the table of contents directly from the images.
    Same return shape as llm_extract_toc_entries, sharing its item-parsing
    tolerance logic (_toc_items_to_entries).

    Deliberately RAISES on any failure (network error, malformed JSON)
    rather than catching and returning [] the way llm_extract_toc_entries
    does -- that swallowing made the text pipeline's _call_with_retry
    wrapper dead code (llm_extract_toc_entries never actually raised to
    it). Here, the caller's retry wrapper does real work."""
    page_count = len(PdfReader(str(pdf_path)).pages)
    if page_count > _MAX_VISION_PAGES:
        raise ValueError(f"{pdf_path}: {page_count} pages exceeds vision-extraction cap of {_MAX_VISION_PAGES}")
    images = render_pages_to_images(pdf_path, pdftoppm_bin=pdftoppm_bin)
    content: list[dict] = [{"type": "text", "text": _VISION_TOC_EXTRACTION_PROMPT}]
    for image_bytes in images:
        b64 = base64.b64encode(image_bytes).decode("ascii")
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        max_tokens=4096,
        temperature=0.0,
    )
    raw = response.choices[0].message.content or ""
    items = parse_json_array(raw)
    return _toc_items_to_entries(items)
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run pytest tests/test_dnb_toc_vision.py -v
```
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add evaluation/dnb_toc_vision.py tests/test_dnb_toc_vision.py
git commit -m "feat: add vision-LLM TOC extraction module for dnb-toc-only"
```

---

### Task 3: `partial_ratio` title-matching fix and vision-neutral naming in `dnb_toc_matching.py`

**Files:**
- Modify: `evaluation/dnb_toc_matching.py`
- Test: `tests/test_dnb_toc_matching.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_dnb_toc_matching.py`, inside `TestAlignTocEntries`:

```python
    def test_matches_despite_trailing_ocr_noise_via_partial_ratio(self):
        # Real garbled-dot-leader-OCR cases measured in the investigation
        # behind docs/superpowers/specs/2026-08-16-dnb-toc-uniform-ocr-design.md
        # section 1d: token_sort_ratio alone scores these well below the
        # 70.0 threshold (a handful of garbage tokens dominates a short
        # real title's token multiset) even though one title is exactly
        # the other's real content plus a trailing noise run.
        a = [_entry("Ein Interview ss m onen een ee eee eee ees", 81)]
        b = [_entry("Ein Interview", 81)]
        self.assertEqual(align_toc_entries(a, b), [(0, 0)])

    def test_still_rejects_genuinely_different_titles_on_the_same_page(self):
        # Negative control: partial_ratio must not become so permissive
        # that two different real entries sharing a page number align.
        a = [_entry("Die Einheit der Vernunft in der Vielfalt ihrer Stimmen", 117)]
        b = [_entry("Metaphysik nach Kant", 117)]
        self.assertEqual(align_toc_entries(a, b), [])
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest tests/test_dnb_toc_matching.py -v -k "partial_ratio or genuinely_different"
```
Expected: `test_matches_despite_trailing_ocr_noise_via_partial_ratio` FAILS
(`token_sort_ratio` alone scores ~47, below the 70.0 threshold).

- [ ] **Step 3: Apply the fix**

In `evaluation/dnb_toc_matching.py`, update the import and the scoring line
inside `align_toc_entries`:

```python
from rapidfuzz import fuzz
```//unchanged, already imported

Change:

```python
            score = max(
                fuzz.token_sort_ratio(title_a.lower(), title_b.lower())
                for title_a in _candidate_titles(entry_a)
                for title_b in _candidate_titles(entry_b)
            )
```

to:

```python
            score = max(
                max(
                    fuzz.token_sort_ratio(title_a.lower(), title_b.lower()),
                    fuzz.partial_ratio(title_a.lower(), title_b.lower()),
                )
                for title_a in _candidate_titles(entry_a)
                for title_b in _candidate_titles(entry_b)
            )
```

Also update `align_toc_entries`'s docstring (currently says titles must
score `>= _ALIGN_SCORE_THRESHOLD` on "rapidfuzz's token_sort_ratio") to:
"the better of rapidfuzz's `token_sort_ratio` and `partial_ratio` (the
latter added 2026-08-16 to tolerate a trailing noise run on one side
without inflating false positives -- see design spec
docs/superpowers/specs/2026-08-16-dnb-toc-uniform-ocr-design.md section
1d/3.2 for the measurements behind this)".

- [ ] **Step 4: Generalize `gate_book`'s naming for vision-vs-vision use**

`gate_book`'s parameters (`heuristic`, `llm`) and docstring describe the
old regex-heuristic-vs-text-LLM asymmetry (e.g. "prefer the heuristic's
title... the heuristic almost never populates authors"). Both gate inputs
are now independent vision-model extractions with no such asymmetry.
Rename the parameters and update the docstring; the merge BEHAVIOR itself
is unchanged (still deterministically prefers `a`'s title, falls back to
`b`'s authors when `a`'s are empty) since all existing callers pass both
arguments positionally (verified: `generate_dnb_toc_ground_truth.py` and
`tests/test_dnb_toc_matching.py` both call `gate_book(x, y, ...)`
positionally, never by keyword).

Change the signature and docstring of `gate_book`:

```python
def gate_book(
    a: list[TocEntry], b: list[TocEntry], threshold: float = 0.90,
) -> tuple[bool, list[TocEntry]]:
    """Whole-book agreement gate (design spec section 4.2 of the original
    2026-08-15 design; the two inputs are now two independent vision-model
    extractions rather than a regex heuristic and a text-LLM pass -- see
    docs/superpowers/specs/2026-08-16-dnb-toc-uniform-ocr-design.md section
    3.1). agreement_rate = matched-pair count / max(len(a), len(b)).
    Below `threshold`, the book is rejected outright (passed=False,
    entries=[]) rather than trimmed down to just the agreeing entries -- a
    partially-agreeing book is exactly the case this design distrusts
    most, and a caller must not silently write a partial/incomplete
    result for it.

    At or above `threshold`, `entries` is the UNION of matched pairs (`a`'s
    title kept -- an arbitrary but deterministic choice between two
    equally-produced extractions -- falling back to `b`'s authors when
    `a`'s own are empty, in case one model dropped them) plus every
    singleton entry either side found alone, ordered by
    printed_page_number (the -1 "unknown" sentinel sorts last). This is
    deliberate: once a book clears the trust bar, a line only one side
    caught is far likelier a real entry the other missed than a
    hallucination -- trimming it out would silently understate the page's
    real content, which is exactly the "incomplete training target"
    failure mode this design exists to avoid."""
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

Also update the module docstring at the top of the file (currently:
"Pure functions over TocEntry lists produced by the two existing
extractors (find_toc_candidates, llm_extract_toc_entries...)") to say the
two lists now come from two independent `vision_extract_toc_entries` calls
(`evaluation/dnb_toc_vision.py`), linking to the new design spec instead.

- [ ] **Step 5: Run the tests to verify they pass**

```bash
uv run pytest tests/test_dnb_toc_matching.py -v
```
Expected: all PASS, including the two new tests.

- [ ] **Step 6: Commit**

```bash
git add evaluation/dnb_toc_matching.py tests/test_dnb_toc_matching.py
git commit -m "fix: tolerate trailing OCR noise in title matching via partial_ratio"
```

---

### Task 4: Add vision model selection alongside the existing text-model selection

**Files:**
- Modify: `evaluation/scripts/generate_dnb_toc_ground_truth.py`
- Test: `tests/test_generate_dnb_toc_ground_truth.py`

This task only ADDS the new `_select_best_models`/`_pick_models`/
`_VISION_MODEL_PATTERNS` functions — it deliberately does NOT touch or
remove `_PREFERRED_MODEL_PATTERNS`/`_select_best_model`/`_pick_model`, nor
the cache function signatures, nor `_run_book_pages`/`_run_book`/
`_generate` (all still text-based at this point). Changing the cache
functions' signature here would break `_run_book_pages`'s own (still
in-place) calls to them before Task 5 replaces that function — so that
change, and removal of the now-superseded text-model-selection functions,
both happen together in Task 5, in the same commit as the code that stops
using them. This task's own tests are purely additive: nothing existing
should change behavior.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_generate_dnb_toc_ground_truth.py`, alongside (not
replacing) the existing `TestSelectBestModel` class. Add `KisskiModel` and
`_select_best_models` to the existing import block first.

```python
class TestSelectBestModels(unittest.TestCase):
    def test_picks_one_from_each_pattern_in_order(self):
        models = [
            KisskiModel(id="qwen3-omni-30b-a3b-instruct", name="Qwen Omni", demand=0),
            KisskiModel(id="gemma-4-31b-it", name="Gemma", demand=1),
        ]
        self.assertEqual(
            _select_best_models(models),
            ["qwen3-omni-30b-a3b-instruct", "gemma-4-31b-it"],
        )

    def test_matches_omni_family_regardless_of_version(self):
        models = [
            KisskiModel(id="qwen5-omni-99b-instruct", name="Qwen Omni next", demand=0),
            KisskiModel(id="gemma-7-40b-it", name="Gemma next", demand=0),
        ]
        self.assertEqual(
            _select_best_models(models),
            ["qwen5-omni-99b-instruct", "gemma-7-40b-it"],
        )

    def test_skips_very_busy_candidate_within_a_pattern(self):
        models = [
            KisskiModel(id="qwen3-omni-30b-a3b-instruct", name="Qwen Omni busy", demand=10),
            KisskiModel(id="gemma-4-31b-it", name="Gemma", demand=0),
        ]
        with self.assertRaises(RuntimeError):
            _select_best_models(models)

    def test_raises_when_fewer_than_two_vision_models_available(self):
        models = [
            KisskiModel(id="glm-4.7", name="GLM (not vision)", demand=0),
        ]
        with self.assertRaises(RuntimeError):
            _select_best_models(models)

    def test_picks_least_busy_among_multiple_matches_in_the_same_pattern(self):
        models = [
            KisskiModel(id="qwen3-omni-30b-a3b-instruct", name="Qwen Omni A", demand=2),
            KisskiModel(id="qwen4-omni-30b-a3b-instruct", name="Qwen Omni B", demand=0),
            KisskiModel(id="gemma-4-31b-it", name="Gemma", demand=0),
        ]
        self.assertEqual(
            _select_best_models(models),
            ["qwen4-omni-30b-a3b-instruct", "gemma-4-31b-it"],
        )
```

Add `KisskiModel` and `_select_best_models` to the test file's existing
imports, without removing anything already imported:

```python
from evaluation.kisski import KisskiModel
```
(add alongside the existing imports at the top of the file, if not
already present)
```python
from evaluation.scripts.generate_dnb_toc_ground_truth import (
    _call_with_retry,
    _load_cached_llm_entries,
    _pad_pages_for_scan,
    _run_book_pages,
    _select_best_model,
    _select_best_models,
    _toc_entries_for_scan,
    _write_cached_llm_entries,
)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
uv run pytest tests/test_generate_dnb_toc_ground_truth.py -v -k SelectBestModels
```
Expected: FAIL — `_select_best_models` doesn't exist yet.

- [ ] **Step 3: Implement**

In `evaluation/scripts/generate_dnb_toc_ground_truth.py`, ADD the
following block immediately after the existing `_pick_model` function —
do not remove or modify `_PREFERRED_MODEL_PATTERNS`, `_select_best_model`,
or `_pick_model` themselves; they stay exactly as they are, and
`_generate` keeps calling `_pick_model` (not `_pick_models`) until Task 5:

```python
# Vision-capable KISSKI model families, confirmed by direct experiment
# (design spec docs/superpowers/specs/2026-08-16-dnb-toc-uniform-ocr-design.md
# section 2.1) -- KISSKI's /models endpoint has no "supports vision" flag,
# so this is a curated allowlist, not something discoverable from the API
# response. Tried in this order: qwen-omni was faster and more accurate
# than gemma in the tested cases.
_VISION_MODEL_PATTERNS = (
    re.compile(r"^qwen\d+-omni"),
    re.compile(r"^gemma-\d+-"),
)


def _select_best_models(models: list, patterns=_VISION_MODEL_PATTERNS, count: int = 2) -> list[str]:
    """Picks `count` DISTINCT vision-capable model ids, one per pattern in
    preference order. Deliberately does NOT fall back to an arbitrary
    global least-busy model: a non-vision-capable model given image
    content would either error or silently ignore the images, and the
    whole point of the agreement gate is two INDEPENDENT reads -- gating a
    single model against itself (or against a model that never saw the
    images at all) would measure something other than what it claims to.
    Raises loudly rather than silently degrading to fewer models."""
    selected: list[str] = []
    for pattern in patterns:
        candidates = [
            m for m in models
            if pattern.match(m.id) and m.availability != "very busy" and m.id not in selected
        ]
        if candidates:
            selected.append(min(candidates, key=lambda m: m.demand).id)
        if len(selected) >= count:
            break
    if len(selected) < count:
        raise RuntimeError(f"Need {count} distinct vision-capable models, found {len(selected)}: {selected}")
    return selected


def _pick_models(base_url: str, api_key: str) -> list[str]:
    return _select_best_models(fetch_kisski_models(base_url, api_key))
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run pytest tests/test_generate_dnb_toc_ground_truth.py -v
```
Expected: `TestSelectBestModels` PASSes; every previously-existing test
(including `TestSelectBestModel`, `TestRunBookPages`,
`TestTocEntriesForScan`, `TestPadPagesForScan`) still PASSes unchanged,
since nothing they exercise was touched.

- [ ] **Step 5: Commit**

```bash
git add evaluation/scripts/generate_dnb_toc_ground_truth.py tests/test_generate_dnb_toc_ground_truth.py
git commit -m "feat: add vision model selection alongside existing text-model selection"
```

---

### Task 5: Replace the text-extraction orchestration with the two-vision-model flow

**Files:**
- Modify: `evaluation/scripts/generate_dnb_toc_ground_truth.py`
- Test: `tests/test_generate_dnb_toc_ground_truth.py`

- [ ] **Step 1: Write the failing tests**

Replace the ENTIRE contents of `tests/test_generate_dnb_toc_ground_truth.py`
with:

```python
"""Unit tests for evaluation/scripts/generate_dnb_toc_ground_truth.py's
pure logic. The real PDF-reading/vision-LLM-calling main() is exercised
manually against the real corpus with a real KISSKI_API_KEY -- see design
spec docs/superpowers/specs/2026-08-16-dnb-toc-uniform-ocr-design.md."""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from pypdf import PdfWriter

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


def _entry(title: str, page: int, authors: tuple[str, ...] = ()) -> TocEntry:
    return TocEntry(title=title, printed_page_number=page, source_page_index=-1, authors=authors)


class TestLlmCacheRoundTrip(unittest.TestCase):
    def test_round_trips_entries_through_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            entries = [
                TocEntry(title="Einleitung", printed_page_number=9, source_page_index=0, authors=("Jane Author",)),
                TocEntry(title="Bibliographie", printed_page_number=-1, source_page_index=1),
            ]
            self.assertIsNone(_load_cached_llm_entries(cache_dir, "book1", "model-a"))
            _write_cached_llm_entries(cache_dir, "book1", "model-a", entries)
            loaded = _load_cached_llm_entries(cache_dir, "book1", "model-a")
            self.assertEqual(loaded, entries)

    def test_round_trip_preserves_printed_roman(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            entries = [
                TocEntry(title="Vorwort", printed_page_number=7, source_page_index=0, printed_roman=True),
            ]
            _write_cached_llm_entries(cache_dir, "book2", "model-a", entries)
            loaded = _load_cached_llm_entries(cache_dir, "book2", "model-a")
            self.assertEqual(loaded, entries)
            self.assertTrue(loaded[0].printed_roman)

    def test_different_models_get_independent_cache_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            entries_a = [TocEntry(title="From model A", printed_page_number=1, source_page_index=0)]
            entries_b = [TocEntry(title="From model B", printed_page_number=1, source_page_index=0)]
            _write_cached_llm_entries(cache_dir, "book3", "model-a", entries_a)
            _write_cached_llm_entries(cache_dir, "book3", "model-b", entries_b)
            self.assertEqual(_load_cached_llm_entries(cache_dir, "book3", "model-a"), entries_a)
            self.assertEqual(_load_cached_llm_entries(cache_dir, "book3", "model-b"), entries_b)


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


class TestRunBookEntries(unittest.TestCase):
    def test_passing_book_writes_expected_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus_directory = Path(tmp) / "corpus"
            corpus_directory.mkdir()
            a = [_entry("Einleitung", 9), _entry("Schluss", 40)]
            b = [_entry("Einleitung", 9), _entry("Schluss", 40)]

            key, passed, reason = _run_book_entries("book1", a, b, corpus_directory)

            self.assertEqual(key, "book1")
            self.assertTrue(passed)
            self.assertEqual(reason, "ok")
            gt_path = corpus_directory / "book1.expected.json"
            self.assertTrue(gt_path.exists())
            data = json.loads(gt_path.read_text(encoding="utf-8"))
            self.assertFalse(data["verified"])
            self.assertEqual(len(data["entries"]), 2)

    def test_below_threshold_book_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus_directory = Path(tmp) / "corpus"
            corpus_directory.mkdir()
            a = [_entry("Einleitung", 9), _entry("A", 20), _entry("B", 30), _entry("C", 40)]
            b = [_entry("Einleitung", 9)]

            key, passed, reason = _run_book_entries("book2", a, b, corpus_directory)

            self.assertFalse(passed)
            self.assertEqual(reason, "below_threshold")
            self.assertFalse((corpus_directory / "book2.expected.json").exists())

    def test_no_entries_from_either_side_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus_directory = Path(tmp) / "corpus"
            corpus_directory.mkdir()

            key, passed, reason = _run_book_entries("book3", [], [], corpus_directory)

            self.assertFalse(passed)
            self.assertEqual(reason, "no_entries")
            self.assertFalse((corpus_directory / "book3.expected.json").exists())


def _fake_vision_client(response_text: str):
    message = MagicMock()
    message.content = response_text
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=response)
    return client


def _make_pdf(path: Path) -> Path:
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    with open(path, "wb") as f:
        writer.write(f)
    return path


_VISION_RESPONSE = (
    '[{"title": "Einleitung", "authors": [], "printed_page_number": "9"}, '
    '{"title": "Schluss", "authors": [], "printed_page_number": "40"}]'
)


class TestRunBook(unittest.IsolatedAsyncioTestCase):
    async def test_calls_each_model_once_and_writes_on_agreement(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            corpus_directory = tmp_path / "corpus"
            cache_directory = tmp_path / "cache"
            corpus_directory.mkdir()
            pdf_path = _make_pdf(tmp_path / "book.pdf")
            client = _fake_vision_client(_VISION_RESPONSE)
            semaphore = asyncio.Semaphore(1)

            key, passed, reason = await _run_book(
                "book1", pdf_path, ("model-a", "model-b"), client, semaphore, corpus_directory, cache_directory,
            )

            self.assertTrue(passed)
            self.assertEqual(reason, "ok")
            self.assertEqual(client.chat.completions.create.await_count, 2)
            self.assertTrue((corpus_directory / "book1.expected.json").exists())

    async def test_cached_model_entries_are_reused_without_a_new_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            corpus_directory = tmp_path / "corpus"
            cache_directory = tmp_path / "cache"
            corpus_directory.mkdir()
            pdf_path = _make_pdf(tmp_path / "book.pdf")
            entries = [_entry("Einleitung", 9), _entry("Schluss", 40)]
            _write_cached_llm_entries(cache_directory, "book2", "model-a", entries)
            _write_cached_llm_entries(cache_directory, "book2", "model-b", entries)
            client = _fake_vision_client(_VISION_RESPONSE)
            semaphore = asyncio.Semaphore(1)

            key, passed, reason = await _run_book(
                "book2", pdf_path, ("model-a", "model-b"), client, semaphore, corpus_directory, cache_directory,
            )

            self.assertTrue(passed)
            client.chat.completions.create.assert_not_called()

    async def test_a_corrupt_pdf_is_reported_not_raised(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            corpus_directory = tmp_path / "corpus"
            cache_directory = tmp_path / "cache"
            corpus_directory.mkdir()
            bad_pdf = tmp_path / "not-a-pdf.pdf"
            bad_pdf.write_text("this is not a pdf")
            client = _fake_vision_client(_VISION_RESPONSE)
            semaphore = asyncio.Semaphore(1)

            key, passed, reason = await _run_book(
                "book3", bad_pdf, ("model-a", "model-b"), client, semaphore, corpus_directory, cache_directory,
            )

            self.assertFalse(passed)
            self.assertTrue(reason.startswith("error:"))


class TestSelectBestModels(unittest.TestCase):
    def test_picks_one_from_each_pattern_in_order(self):
        models = [
            KisskiModel(id="qwen3-omni-30b-a3b-instruct", name="Qwen Omni", demand=0),
            KisskiModel(id="gemma-4-31b-it", name="Gemma", demand=1),
        ]
        self.assertEqual(_select_best_models(models), ["qwen3-omni-30b-a3b-instruct", "gemma-4-31b-it"])

    def test_matches_omni_family_regardless_of_version(self):
        models = [
            KisskiModel(id="qwen5-omni-99b-instruct", name="Qwen Omni next", demand=0),
            KisskiModel(id="gemma-7-40b-it", name="Gemma next", demand=0),
        ]
        self.assertEqual(_select_best_models(models), ["qwen5-omni-99b-instruct", "gemma-7-40b-it"])

    def test_skips_very_busy_candidate_within_a_pattern(self):
        models = [
            KisskiModel(id="qwen3-omni-30b-a3b-instruct", name="Qwen Omni busy", demand=10),
            KisskiModel(id="gemma-4-31b-it", name="Gemma", demand=0),
        ]
        with self.assertRaises(RuntimeError):
            _select_best_models(models)

    def test_raises_when_fewer_than_two_vision_models_available(self):
        models = [KisskiModel(id="glm-4.7", name="GLM (not vision)", demand=0)]
        with self.assertRaises(RuntimeError):
            _select_best_models(models)

    def test_picks_least_busy_among_multiple_matches_in_the_same_pattern(self):
        models = [
            KisskiModel(id="qwen3-omni-30b-a3b-instruct", name="Qwen Omni A", demand=2),
            KisskiModel(id="qwen4-omni-30b-a3b-instruct", name="Qwen Omni B", demand=0),
            KisskiModel(id="gemma-4-31b-it", name="Gemma", demand=0),
        ]
        self.assertEqual(_select_best_models(models), ["qwen4-omni-30b-a3b-instruct", "gemma-4-31b-it"])
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest tests/test_generate_dnb_toc_ground_truth.py -v
```
Expected: FAIL/ERROR — `_run_book_entries` and the new `_run_book` shape
don't exist yet.

- [ ] **Step 3: Rewrite the script**

In `evaluation/scripts/generate_dnb_toc_ground_truth.py`:

Replace the module docstring with:

```python
"""Generates bulk-tier structured ground truth for dnb-toc-only (design
spec docs/superpowers/specs/2026-08-16-dnb-toc-uniform-ocr-design.md,
which supersedes the two-text-extractor design in
docs/superpowers/specs/2026-08-15-dnb-toc-ground-truth-generation-design.md).
For every manifest book not held out in eval_tier_ids.json (see
select_dnb_toc_eval_sample.py and evaluation/README.md's "Building
dnb-toc-only ground truth"), sends the book's page images to two
independent vision-capable KISSKI models
(evaluation.dnb_toc_vision.vision_extract_toc_entries) and writes
<id>.expected.json with "verified": false only when they agree well
enough (evaluation.dnb_toc_matching.gate_book, >=0.90 whole-book
agreement). Books that don't clear the gate are skipped and reported, not
partially written.

Spends real KISSKI API budget (two calls per book, one per vision model --
see evaluation/refresh_llm_cache.py's docstring for the shared
KISSKI_API_KEY setup this script reuses). Not a pytest test.

    uv run python evaluation/scripts/generate_dnb_toc_ground_truth.py --limit 50   # smoke test
    uv run python evaluation/scripts/generate_dnb_toc_ground_truth.py               # full corpus
    uv run python evaluation/scripts/generate_dnb_toc_ground_truth.py --spot-check 30

`--spot-check N` does not generate anything -- instead it samples N books
that already passed the bulk-tier gate (i.e. have "verified": false;
already-human-verified eval-tier entries are excluded) and walks through a
manual, terminal-driven visual Accept/Reject check against the real PDF for
each, then prints the measured accept rate as an estimate of the gate's
real precision.
"""

import argparse
import asyncio
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Optional

from openai import AsyncOpenAI

from chapter_segmentation.segmentation import TocEntry
from evaluation.dnb_toc_matching import gate_book, toc_entry_to_gt_dict
from evaluation.dnb_toc_vision import vision_extract_toc_entries
from evaluation.harness import corpus_dir, llm_cache_dir, load_manifest_books
from evaluation.kisski import DEFAULT_KISSKI_BASE_URL, fetch_kisski_models
from evaluation.scripts.select_dnb_toc_eval_sample import manifest_key
```

Delete entirely: `_PAGE_NUMBER_GUARD_PADDING`, `_pad_pages_for_scan`,
`_toc_entries_for_scan` (no longer relevant — nothing reads page text
anymore), and `_PREFERRED_MODEL_PATTERNS`/`_select_best_model`/
`_pick_model` (the old text-model selection, superseded by Task 4's
`_VISION_MODEL_PATTERNS`/`_select_best_models`/`_pick_models`, which stay
and are NOT touched by this task).

Replace `_cache_path`, `_load_cached_llm_entries`, `_write_cached_llm_entries`
(adding the `model` parameter — safe now, since this task also replaces
every remaining call site below):

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

`_call_with_retry` is unchanged — leave it exactly as it is.

Replace `_GATE_THRESHOLD` onward — everything from the old
`_run_book_pages` through `_run_book` — with:

```python
_GATE_THRESHOLD = 0.90


def _run_book_entries(
    key: str, entries_a: list[TocEntry], entries_b: list[TocEntry], corpus_directory: Path,
) -> tuple[str, bool, str]:
    """Core per-book gating logic, given two already-extracted TocEntry
    lists -- kept separate from PDF/vision-call I/O so it's directly
    unit-testable with synthetic entries, no real PDF or network call
    needed. Returns (key, passed, reason); reason is "ok" on success, else
    why the book was skipped/rejected ("no_entries", "below_threshold")."""
    if not entries_a and not entries_b:
        return key, False, "no_entries"
    passed, entries = gate_book(entries_a, entries_b, threshold=_GATE_THRESHOLD)
    if not passed:
        return key, False, "below_threshold"
    gt_path = corpus_directory / f"{key}.expected.json"
    gt_path.write_text(
        json.dumps({"entries": [toc_entry_to_gt_dict(e) for e in entries], "verified": False}, indent=2) + "\n",
        encoding="utf-8",
    )
    return key, True, "ok"


_CORPUS_NAME = "dnb-toc-only"


async def _run_book(
    key: str, pdf_path: Path, models: tuple[str, str], client, semaphore: asyncio.Semaphore,
    corpus_directory: Path, cache_directory: Path,
) -> tuple[str, bool, str]:
    """Thin I/O wrapper around _run_book_entries -- calls
    vision_extract_toc_entries once per model (through the cache, then
    _call_with_retry on a miss), and delegates the two resulting entry
    lists to _run_book_entries. Catches any exception (a corrupt/unreadable
    PDF, a network error that survives _call_with_retry's own retries,
    etc.) and reports it as a failed-but-tuple-shaped result instead of
    letting it propagate -- same "catch-log-continue" convention
    evaluation/refresh_llm_cache.py already established for this kind of
    long, unattended, budget-spending batch job. One book's failure must
    never abort the rest of a ~1000-book run."""
    try:
        entries_by_model = []
        for model in models:
            cached = _load_cached_llm_entries(cache_directory, key, model)
            async with semaphore:
                if cached is not None:
                    entries = cached
                else:
                    entries = await _call_with_retry(
                        lambda m=model: vision_extract_toc_entries(pdf_path, m, client)
                    )
                    # Only cache a non-empty result -- an empty list here
                    # could be a genuine "no TOC content on these pages" or
                    # a transient failure already exhausted by
                    # _call_with_retry; caching it either way would make a
                    # later re-run trust a possibly-transient empty result
                    # forever instead of retrying.
                    if entries:
                        _write_cached_llm_entries(cache_directory, key, model, entries)
            entries_by_model.append(entries)
        return _run_book_entries(key, entries_by_model[0], entries_by_model[1], corpus_directory)
    except Exception as exc:  # noqa: BLE001 -- must never let one book crash the whole batch
        print(f"[error] {key}: {exc}")
        return key, False, f"error: {type(exc).__name__}"


# Vision-capable KISSKI model families, confirmed by direct experiment
# (design spec docs/superpowers/specs/2026-08-16-dnb-toc-uniform-ocr-design.md
# section 2.1) -- KISSKI's /models endpoint has no "supports vision" flag,
# so this is a curated allowlist, not something discoverable from the API
# response. Tried in this order: qwen-omni was faster and more accurate
# than gemma in the tested cases.
_VISION_MODEL_PATTERNS = (
    re.compile(r"^qwen\d+-omni"),
    re.compile(r"^gemma-\d+-"),
)


def _select_best_models(models: list, patterns=_VISION_MODEL_PATTERNS, count: int = 2) -> list[str]:
    """Picks `count` DISTINCT vision-capable model ids, one per pattern in
    preference order. Deliberately does NOT fall back to an arbitrary
    global least-busy model: a non-vision-capable model given image
    content would either error or silently ignore the images, and the
    whole point of the agreement gate is two INDEPENDENT reads -- gating a
    single model against itself (or against a model that never saw the
    images at all) would measure something other than what it claims to.
    Raises loudly rather than silently degrading to fewer models."""
    selected: list[str] = []
    for pattern in patterns:
        candidates = [
            m for m in models
            if pattern.match(m.id) and m.availability != "very busy" and m.id not in selected
        ]
        if candidates:
            selected.append(min(candidates, key=lambda m: m.demand).id)
        if len(selected) >= count:
            break
    if len(selected) < count:
        raise RuntimeError(f"Need {count} distinct vision-capable models, found {len(selected)}: {selected}")
    return selected


def _pick_models(base_url: str, api_key: str) -> list[str]:
    return _select_best_models(fetch_kisski_models(base_url, api_key))


async def _run_all(
    keys_and_paths: list[tuple[str, Path]], models: tuple[str, str], client, concurrency: int,
    corpus_directory: Path, cache_directory: Path,
) -> list[tuple[str, bool, str]]:
    semaphore = asyncio.Semaphore(concurrency)
    return list(await asyncio.gather(*[
        _run_book(key, path, models, client, semaphore, corpus_directory, cache_directory)
        for key, path in keys_and_paths
    ]))


def _generate(args: argparse.Namespace) -> int:
    cdir = corpus_dir(_CORPUS_NAME)
    eval_tier_path = cdir / "eval_tier_ids.json"
    eval_tier_ids = set(json.loads(eval_tier_path.read_text(encoding="utf-8"))) if eval_tier_path.exists() else set()

    books = load_manifest_books(_CORPUS_NAME)
    eligible = [b for b in books if manifest_key(b) not in eval_tier_ids]
    if args.limit is not None:
        eligible = eligible[: args.limit]
    candidates = [(manifest_key(b), cdir / b["filename"]) for b in eligible if (cdir / b["filename"]).exists()]
    missing_pdf_count = len(eligible) - len(candidates)

    api_key = os.environ["KISSKI_API_KEY"]
    models = tuple(_pick_models(DEFAULT_KISSKI_BASE_URL, api_key))
    client = AsyncOpenAI(base_url=DEFAULT_KISSKI_BASE_URL, api_key=api_key)

    results = asyncio.run(_run_all(candidates, models, client, args.concurrency, cdir, llm_cache_dir(_CORPUS_NAME)))
    passed = [r for r in results if r[1]]
    by_reason: dict[str, int] = {}
    for _, ok, reason in results:
        if not ok:
            by_reason[reason] = by_reason.get(reason, 0) + 1
    print(f"Vision models used: {models[0]}, {models[1]}")
    print(f"{len(passed)}/{len(results)} books passed the gate and got .expected.json written.")
    for reason, count in sorted(by_reason.items()):
        print(f"  {count} skipped: {reason}")
    if missing_pdf_count:
        print(f"  {missing_pdf_count} skipped: missing_pdf (not downloaded locally)")
    return 0
```

`main()` and `_spot_check()` are unchanged — leave them exactly as they
are.

- [ ] **Step 4: Run the full test file to verify it passes**

```bash
uv run pytest tests/test_generate_dnb_toc_ground_truth.py -v
```
Expected: all PASS.

- [ ] **Step 5: Run the full project test suite**

```bash
uv run pytest
```
Expected: all PASS (no test anywhere still imports a name this rewrite
removed).

- [ ] **Step 6: Commit**

```bash
git add evaluation/scripts/generate_dnb_toc_ground_truth.py tests/test_generate_dnb_toc_ground_truth.py
git commit -m "feat: switch dnb-toc-only GT generation to two-vision-model gate"
```

---

### Task 6: Update documentation

**Files:**
- Modify: `evaluation/README.md`
- Modify: `evaluation/scripts/README.md`

- [ ] **Step 1: Update `evaluation/README.md`'s "Building dnb-toc-only ground truth" section**

In `evaluation/README.md`, find the "**Bulk tier**" paragraph (currently
reads "Runs two independent extractors per book (the regex heuristic and
a KISSKI LLM pass) and writes `.expected.json` only when they agree on at
least 90% of the book's entries"). Replace the spec reference and that
paragraph:

```markdown
See
`docs/superpowers/specs/2026-08-16-dnb-toc-uniform-ocr-design.md`
for the full design (supersedes the two-text-extractor design in
`docs/superpowers/specs/2026-08-15-dnb-toc-ground-truth-generation-design.md`).
Two tiers, both writing
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

Sends each book's page images (rendered via `pdftoppm`, no OCR) to two
independent vision-capable KISSKI models and writes `.expected.json` only
when they agree on at least 90% of the book's entries -- see
`evaluation/dnb_toc_matching.py` and `evaluation/dnb_toc_vision.py`. Books
that don't clear that bar are skipped and reported, not partially written.
Requires `pdftoppm` (poppler) on `PATH` -- see this file's "Cleaning a
badly-scanned PDF" section for the install command.
```

- [ ] **Step 2: Regenerate the `--help` dump reference**

```bash
uv run python evaluation/scripts/generate_dnb_toc_ground_truth.py --help
```

Find `generate_dnb_toc_ground_truth.py`'s existing `--help` dump in
`evaluation/scripts/README.md` and replace it with this command's current
output (the flags themselves — `--limit`, `--concurrency`, `--spot-check`
— are unchanged by this plan, so this should be a no-op diff; only run it
to confirm).

- [ ] **Step 3: Commit**

```bash
git add evaluation/README.md evaluation/scripts/README.md
git commit -m "docs: describe the two-vision-model dnb-toc-only GT pipeline"
```
