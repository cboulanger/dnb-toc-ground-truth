# Vision+Text Model Pairing for dnb-toc-only Ground Truth Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `generate_dnb_toc_ground_truth.py` pair one vision-model endpoint with one text-only-model endpoint (fed freshly-OCR'd, reading-order-reconstructed page text) as an alternative to today's two-vision-model gate, via new `--text-endpoint`/`--text-config-file` CLI flags.

**Architecture:** A new `evaluation/dnb_toc_ocr.py` module (structurally parallel to `evaluation/dnb_toc_vision.py`) turns a book's TOC pages into OCR'd text rows via `ocrmypdf` + `pdfalto`, then asks a text-only LLM the same verbatim/`skip`-flag extraction question the vision prompt already asks. `evaluation/dnb_toc_vision.py`'s cache gains an optional `"kind"` field so `arbitrate_dnb_toc.py` can label which side of a disagreement came from OCR'd text vs. a direct image read. `gate_book`/`diff_toc_entries` (`evaluation/dnb_toc_matching.py`) are untouched — they already just compare two `list[TocEntry]`.

**Tech Stack:** Python 3.12, `openai` AsyncOpenAI client, `ocrmypdf`, `pdfalto` (sibling checkout), `xml.etree.ElementTree`, `pytest`/`unittest`.

**Design spec:** `docs/superpowers/specs/2026-08-20-dnb-toc-vision-text-pairing-design.md` — read it for full rationale; this plan implements it section by section (§2 → Task 1, §3 → Tasks 2-3, §4 → Task 4, §1/§5 → Tasks 5-6, §6 → Task 7).

---

## Task 1: Relocate `OpenAICompatibleLLMClient` to `evaluation/inference_endpoints.py`

Pure move (spec §2): `_OpenAICompatibleLLMClient` in `evaluation/refresh_llm_cache.py:114` already has zero KISSKI/MPCDF-specific knowledge. Moving it (renamed, non-private) into `evaluation/inference_endpoints.py` lets the new `evaluation/dnb_toc_ocr.py` module (Task 3) import it without importing a script module.

**Files:**
- Modify: `evaluation/inference_endpoints.py`
- Modify: `evaluation/refresh_llm_cache.py`
- Modify: `tests/test_refresh_llm_cache.py`
- Modify: `tests/test_inference_endpoints.py`

- [ ] **Step 1: Add a failing test for the new location**

Append to `tests/test_inference_endpoints.py` (add `unittest.mock` to its imports — currently only `from unittest.mock import patch` is imported, so change that line to also import `AsyncMock` and `MagicMock`):

```python
from unittest.mock import AsyncMock, MagicMock, patch
```

Append this test class at the end of the file:

```python
class TestOpenAICompatibleLLMClient(unittest.IsolatedAsyncioTestCase):
    async def test_uses_the_given_client_not_a_new_one(self):
        fake_client = MagicMock()
        message = MagicMock()
        message.content = "hello"
        choice = MagicMock()
        choice.message = message
        response = MagicMock()
        response.choices = [choice]
        fake_client.chat.completions.create = AsyncMock(return_value=response)

        llm_client = OpenAICompatibleLLMClient(model="model-x", client=fake_client)
        result = await llm_client.generate("prompt", max_tokens=10, temperature=0.0)

        self.assertEqual(result, "hello")
        fake_client.chat.completions.create.assert_awaited_once_with(
            model="model-x", messages=[{"role": "user", "content": "prompt"}], max_tokens=10, temperature=0.0,
        )
```

Add `OpenAICompatibleLLMClient` to the existing `from evaluation.inference_endpoints import (...)` line at the top of the file, e.g.:

```python
from evaluation.inference_endpoints import (
    ModelEndpoint, OpenAICompatibleLLMClient, load_mpcdf_sessions, resolve_endpoint_from_env,
    resolve_endpoints_from_config_file,
)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_inference_endpoints.py::TestOpenAICompatibleLLMClient -v`
Expected: FAIL with `ImportError: cannot import name 'OpenAICompatibleLLMClient'`

- [ ] **Step 3: Add the class to `evaluation/inference_endpoints.py`**

Change the top import line from:

```python
import os
import re
from dataclasses import dataclass
from pathlib import Path

from openai import AsyncOpenAI
```

to:

```python
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from openai import AsyncOpenAI
```

Append this class at the end of `evaluation/inference_endpoints.py`:

```python
class OpenAICompatibleLLMClient:
    """Minimal LLMClient (see chapter_segmentation.llm.LLMClient) wrapping
    an already-built AsyncOpenAI client -- callers construct the client
    themselves (KISSKI's own base_url/api_key, or a ModelEndpoint's client
    resolved above), so this class has no provider-specific knowledge at
    all. Relocated here (2026-08-20, see design spec
    docs/superpowers/specs/2026-08-20-dnb-toc-vision-text-pairing-design.md
    section 2) from evaluation/refresh_llm_cache.py, its original home --
    a pure move, not a behavior change: evaluation/dnb_toc_ocr.py's
    text_extract_toc_entries needs the exact same LLMClient-shaped
    adapter, and importing it from refresh_llm_cache.py (a script module)
    would be backwards for a library module to depend on."""

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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_inference_endpoints.py::TestOpenAICompatibleLLMClient -v`
Expected: PASS

- [ ] **Step 5: Remove the old class and re-point `evaluation/refresh_llm_cache.py` at the new location**

In `evaluation/refresh_llm_cache.py`, delete the class definition entirely. Change:

```python
class _OpenAICompatibleLLMClient:
    """Minimal LLMClient (see chapter_segmentation.llm.LLMClient) wrapping
    an already-built AsyncOpenAI client -- callers construct the client
    themselves (KISSKI's own base_url/api_key, or a ModelEndpoint's own
    client from evaluation.inference_endpoints), so this class has no
    provider-specific knowledge at all."""

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


def _fully_covered_model_ids(book_specs: list[tuple[Path, str]]) -> set[str]:
```

to:

```python
def _fully_covered_model_ids(book_specs: list[tuple[Path, str]]) -> set[str]:
```

(i.e. delete the class and the two blank lines that separated it from `_fully_covered_model_ids`, leaving that function definition directly where the class used to be.)

Change the import block from:

```python
from evaluation.inference_endpoints import (
    DEFAULT_SESSIONS_FILENAME, ModelEndpoint, resolve_endpoint_from_env, resolve_endpoints_from_config_file,
)
```

to:

```python
from evaluation.inference_endpoints import (
    DEFAULT_SESSIONS_FILENAME, ModelEndpoint, OpenAICompatibleLLMClient, resolve_endpoint_from_env,
    resolve_endpoints_from_config_file,
)
```

Change `_model_and_client_for_endpoint`'s signature and body from:

```python
def _model_and_client_for_endpoint(endpoint: ModelEndpoint) -> tuple[KisskiModel, _OpenAICompatibleLLMClient]:
    """Wraps a resolved ModelEndpoint into the (model, llm_client) shape
    _run_book_for_model/_upsert_cache expect. demand=0 -- KisskiModel's
    own `demand` field has no meaning for an --endpoint-selected model
    (no shared pool, nothing to be busy relative to); 0 is also what
    KisskiModel.availability reads as "available", the only sensible
    default for a model you deployed yourself and know is up. Reuses
    KisskiModel itself rather than inventing a second (id, name, demand)
    type -- despite the name, it's just a model-identity-plus-demand
    record, not KISSKI-specific in shape."""
    model = KisskiModel(id=endpoint.model_id, name=endpoint.model_id, demand=0)
    llm_client = _OpenAICompatibleLLMClient(model=endpoint.model_id, client=endpoint.client)
    return model, llm_client
```

to:

```python
def _model_and_client_for_endpoint(endpoint: ModelEndpoint) -> tuple[KisskiModel, OpenAICompatibleLLMClient]:
    """Wraps a resolved ModelEndpoint into the (model, llm_client) shape
    _run_book_for_model/_upsert_cache expect. demand=0 -- KisskiModel's
    own `demand` field has no meaning for an --endpoint-selected model
    (no shared pool, nothing to be busy relative to); 0 is also what
    KisskiModel.availability reads as "available", the only sensible
    default for a model you deployed yourself and know is up. Reuses
    KisskiModel itself rather than inventing a second (id, name, demand)
    type -- despite the name, it's just a model-identity-plus-demand
    record, not KISSKI-specific in shape."""
    model = KisskiModel(id=endpoint.model_id, name=endpoint.model_id, demand=0)
    llm_client = OpenAICompatibleLLMClient(model=endpoint.model_id, client=endpoint.client)
    return model, llm_client
```

In `_main`, change:

```python
        llm_client = _OpenAICompatibleLLMClient(model=model.id, client=AsyncOpenAI(base_url=base_url, api_key=api_key))
```

to:

```python
        llm_client = OpenAICompatibleLLMClient(model=model.id, client=AsyncOpenAI(base_url=base_url, api_key=api_key))
```

- [ ] **Step 6: Update `tests/test_refresh_llm_cache.py` to match the new location**

Remove the `TestOpenAICompatibleLLMClient` test class entirely (it now lives in `tests/test_inference_endpoints.py`, Step 1 above).

Change the import block from:

```python
from evaluation.inference_endpoints import ModelEndpoint
from evaluation.refresh_llm_cache import (
    _OpenAICompatibleLLMClient,
    _all_cached_model_ids,
    _call_with_retry,
    _fully_covered_model_ids,
    _has_cached_entry,
    _main,
    _model_and_client_for_endpoint,
    _process_model,
    _run_book_for_model,
    _upsert_cache,
)
```

to:

```python
from evaluation.inference_endpoints import ModelEndpoint, OpenAICompatibleLLMClient
from evaluation.refresh_llm_cache import (
    _all_cached_model_ids,
    _call_with_retry,
    _fully_covered_model_ids,
    _has_cached_entry,
    _main,
    _model_and_client_for_endpoint,
    _process_model,
    _run_book_for_model,
    _upsert_cache,
)
```

In `TestModelAndClientForEndpoint.test_wraps_endpoint_with_demand_zero_and_the_endpoints_own_client`, change:

```python
        self.assertIsInstance(llm_client, _OpenAICompatibleLLMClient)
```

to:

```python
        self.assertIsInstance(llm_client, OpenAICompatibleLLMClient)
```

- [ ] **Step 7: Run the full test suite for both files**

Run: `uv run pytest tests/test_inference_endpoints.py tests/test_refresh_llm_cache.py -v`
Expected: PASS, no `_OpenAICompatibleLLMClient` references remain anywhere.

Run: `grep -rn "_OpenAICompatibleLLMClient" evaluation/ tests/` — expect no output (confirms the rename is complete, no stray references).

- [ ] **Step 8: Commit**

```bash
git add evaluation/inference_endpoints.py evaluation/refresh_llm_cache.py tests/test_refresh_llm_cache.py tests/test_inference_endpoints.py
git commit -m "refactor: relocate OpenAICompatibleLLMClient into inference_endpoints.py

Pure move, no behavior change -- evaluation/dnb_toc_ocr.py's upcoming
text_extract_toc_entries needs the same LLMClient adapter and shouldn't
import it from a script module. See design spec
docs/superpowers/specs/2026-08-20-dnb-toc-vision-text-pairing-design.md
section 2."
```

---

## Task 2: `evaluation/dnb_toc_ocr.py` — ALTO row reconstruction and `ocr_pages_to_rows`

New module, new file. Implements spec §3's `ocr_pages_to_rows`: runs `ocrmypdf --force-ocr` unconditionally, then `pdfalto`, then reconstructs printed reading order by clustering raw `<String>` ALTO tokens by `VPOS` (8px tolerance) and sorting by `HPOS` within each cluster — deliberately NOT trusting `pdfalto`'s own `<TextBlock>`/`<TextLine>` grouping, which the 2026-08-16 uniform-ocr-design investigation (§1b) found groups a dot-leader TOC's titles and page numbers into separate blocks.

The ALTO namespace and parsing conventions below match the two other ALTO-consuming modules already in this repo (`evaluation/scripts/alto_scan_noise.py`, `evaluation/scripts/layout_features.py`): `xml.etree.ElementTree`, explicit Clark-notation namespace prefixing (`"{http://www.loc.gov/standards/alto/ns-v3#}"`), `root.iter(_ALTO_NS + "Page")` to walk pages, `page.iter(_ALTO_NS + "String")` to reach tokens regardless of intermediate `TextBlock`/`TextLine` nesting.

**Files:**
- Create: `evaluation/dnb_toc_ocr.py`
- Create: `tests/test_dnb_toc_ocr.py`

- [ ] **Step 1: Write the failing test for row reconstruction**

Create `tests/test_dnb_toc_ocr.py`:

```python
"""Unit tests for evaluation/dnb_toc_ocr.py -- OCR'd-text TOC extraction
for dnb-toc-only's vision+text pairing, see design spec
docs/superpowers/specs/2026-08-20-dnb-toc-vision-text-pairing-design.md
section 3. _rows_from_alto_xml is tested against a hand-written fixture
ALTO XML file (no real ocrmypdf/pdfalto dependency, matching how
evaluation/dnb_toc_vision.py's own render_pages_to_images test is the
only one of that module's tests that shells out to a real binary).
text_extract_toc_entries is tested with ocr_pages_to_rows mocked out and a
mocked OpenAI-shaped client, no real network call or OCR."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from evaluation.dnb_toc_ocr import _rows_from_alto_xml


_ALTO_NS_URI = "http://www.loc.gov/standards/alto/ns-v3#"


def _write_alto_fixture(path: Path) -> Path:
    # Page 1: a dot-leader TOC line whose title and page number pdfalto's
    # own TextBlock segmentation put in SEPARATE TextBlocks (mirroring the
    # real tesseract failure mode the 2026-08-16 investigation found) but
    # whose VPOS values (100, 102) are within the 8px tolerance -- the row
    # reconstruction must still merge them into one row, sorted by HPOS
    # regardless of which TextBlock each token came from.
    # Page 2: two genuinely separate rows (VPOS 200 and 260, 60px apart --
    # well outside tolerance), each with its title and number already in
    # the same TextLine, must NOT merge into one row.
    # Page 3: an empty PrintSpace (a blank page pdfalto still emits a bare
    # <Page> element for) must produce an empty string, not crash.
    content = f"""<?xml version="1.0" encoding="UTF-8"?>
<alto xmlns="{_ALTO_NS_URI}">
  <Layout>
    <Page ID="Page1" PHYSICAL_IMG_NR="1" WIDTH="400" HEIGHT="600">
      <PrintSpace>
        <TextBlock ID="p1_b1">
          <TextLine ID="p1_t1">
            <String CONTENT="Einleitung" HPOS="50" VPOS="100" WIDTH="80" HEIGHT="12"/>
          </TextLine>
        </TextBlock>
        <TextBlock ID="p1_b2">
          <TextLine ID="p1_t2">
            <String CONTENT="9" HPOS="300" VPOS="102" WIDTH="10" HEIGHT="12"/>
          </TextLine>
        </TextBlock>
      </PrintSpace>
    </Page>
    <Page ID="Page2" PHYSICAL_IMG_NR="2" WIDTH="400" HEIGHT="600">
      <PrintSpace>
        <TextBlock ID="p2_b1">
          <TextLine ID="p2_t1">
            <String CONTENT="Schluss" HPOS="50" VPOS="200" WIDTH="80" HEIGHT="12"/>
            <String CONTENT="40" HPOS="300" VPOS="200" WIDTH="10" HEIGHT="12"/>
          </TextLine>
          <TextLine ID="p2_t2">
            <String CONTENT="Bibliographie" HPOS="50" VPOS="260" WIDTH="80" HEIGHT="12"/>
            <String CONTENT="45" HPOS="300" VPOS="260" WIDTH="10" HEIGHT="12"/>
          </TextLine>
        </TextBlock>
      </PrintSpace>
    </Page>
    <Page ID="Page3" PHYSICAL_IMG_NR="3" WIDTH="400" HEIGHT="600">
      <PrintSpace/>
    </Page>
  </Layout>
</alto>
"""
    path.write_text(content, encoding="utf-8")
    return path


class TestRowsFromAltoXml(unittest.TestCase):
    def test_reconstructs_one_row_per_page_in_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            alto_path = _write_alto_fixture(Path(tmp) / "test.alto.xml")

            rows = _rows_from_alto_xml(alto_path)

            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0], "Einleitung 9")

    def test_tokens_across_different_text_blocks_merge_when_vpos_is_close(self):
        # The actual regression this function exists to fix: pdfalto's own
        # TextBlock boundaries put "Einleitung" and "9" in separate blocks,
        # but their VPOS values (100, 102) are within the 8px tolerance --
        # row reconstruction must ignore the TextBlock boundary entirely.
        with tempfile.TemporaryDirectory() as tmp:
            alto_path = _write_alto_fixture(Path(tmp) / "test.alto.xml")
            rows = _rows_from_alto_xml(alto_path)
            self.assertEqual(rows[0], "Einleitung 9")

    def test_rows_further_apart_than_tolerance_stay_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            alto_path = _write_alto_fixture(Path(tmp) / "test.alto.xml")
            rows = _rows_from_alto_xml(alto_path)
            self.assertEqual(rows[1], "Schluss 40\nBibliographie 45")

    def test_empty_page_produces_an_empty_string_not_a_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            alto_path = _write_alto_fixture(Path(tmp) / "test.alto.xml")
            rows = _rows_from_alto_xml(alto_path)
            self.assertEqual(rows[2], "")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_dnb_toc_ocr.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'evaluation.dnb_toc_ocr'`

- [ ] **Step 3: Create `evaluation/dnb_toc_ocr.py` with the ALTO row-reconstruction logic and `ocr_pages_to_rows`**

```python
"""OCR'd-text TOC extraction for dnb-toc-only's vision+text-model pairing
-- feeds a text-only LLM the book's TOC pages reconstructed as plain text
via ocrmypdf + pdfalto, instead of vision_extract_toc_entries' page
images. See design spec
docs/superpowers/specs/2026-08-20-dnb-toc-vision-text-pairing-design.md
section 3. Structurally parallel to evaluation/dnb_toc_vision.py, reusing
its cache_path/load_cached_llm_entries/write_cached_llm_entries directly
(both extraction paths share one cache, keyed by (book, model) -- see
that module's own "kind" field for how a cached entry records which
extraction path produced it)."""

import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from chapter_segmentation._llm_json import parse_json_array
from chapter_segmentation.segmentation import TocEntry, _toc_items_to_entries
from evaluation.inference_endpoints import OpenAICompatibleLLMClient
from evaluation.scripts import pdfalto_runner

_ALTO_NS = "{http://www.loc.gov/standards/alto/ns-v3#}"

# VPOS tolerance (ALTO points/px) for clustering <String> tokens into one
# reconstructed row -- see design spec section 3 and
# docs/superpowers/specs/2026-08-16-dnb-toc-uniform-ocr-design.md section
# 1b, which found pdfalto's own <TextBlock>/<TextLine> segmentation groups
# a dot-leader TOC's title column and page-number column into SEPARATE
# blocks (all titles, then all page numbers) rather than one block per
# printed line -- exactly the failure this raw-<String>-position
# reclustering avoids by ignoring ALTO's own TextBlock/TextLine nesting
# entirely and re-deriving rows purely from token geometry.
_ROW_VPOS_TOLERANCE = 8.0


def _rows_from_alto_xml(alto_path: Path) -> list[str]:
    """One reading-order-reconstructed text block per ALTO <Page>, in page
    order. Clusters every <String> token on a page by VPOS (tolerance
    _ROW_VPOS_TOLERANCE) into rows, sorts each row's tokens by HPOS, and
    joins them with a single space -- deliberately ignores ALTO's own
    <TextBlock>/<TextLine> nesting (see _ROW_VPOS_TOLERANCE's docstring for
    why trusting it doesn't work for this corpus's dot-leader TOCs). A page
    with no <String> tokens at all (e.g. a blank page pdfalto still emits
    an empty <Page> element for) produces an empty string, not an error --
    ocr_pages_to_rows's caller needs one entry per page to keep its
    per-page-list shape aligned with render_pages_to_images'."""
    root = ET.parse(alto_path).getroot()
    page_rows: list[str] = []
    for page in root.iter(_ALTO_NS + "Page"):
        tokens = sorted(
            (
                (float(string.get("VPOS", "0")), float(string.get("HPOS", "0")), string.get("CONTENT", ""))
                for string in page.iter(_ALTO_NS + "String")
                if string.get("CONTENT")
            ),
            key=lambda token: (token[0], token[1]),
        )
        clusters: list[list[tuple[float, float, str]]] = []
        for token in tokens:
            if clusters and token[0] - clusters[-1][0][0] <= _ROW_VPOS_TOLERANCE:
                clusters[-1].append(token)
            else:
                clusters.append([token])
        rows = [
            " ".join(content for _, _, content in sorted(cluster, key=lambda t: t[1]))
            for cluster in clusters
        ]
        page_rows.append("\n".join(rows))
    return page_rows


def ocr_pages_to_rows(pdf_path: Path, *, pdfalto_bin: str | None = None) -> list[str]:
    """Forces fresh OCR on pdf_path (ocrmypdf --force-ocr, unconditionally
    -- this corpus's PDFs are pre-filtered to 1-3 TOC pages, so re-OCRing
    even an already-text-layered PDF is cheap and keeps behavior uniform
    regardless of the source PDF's own text layer quality), then runs
    pdfalto and reconstructs reading order via _rows_from_alto_xml. Returns
    one string per page, in page order -- the same per-page-list shape
    render_pages_to_images (evaluation/dnb_toc_vision.py) returns for
    images, so the vision and text extraction paths stay visually parallel
    in any calling code. `pdfalto_bin` is passed straight through to
    pdfalto_runner.resolve_pdfalto_binary -- None (the default) resolves
    via the PDFALTO_BIN environment variable, then a bare "pdfalto" on
    PATH; pdfalto is a sibling checkout, not on PATH by default (see
    CLAUDE.local.md/evaluation/CLAUDE.md's pdfalto notes). Raises
    RuntimeError if ocrmypdf exits non-zero -- propagates to the caller
    exactly like any other extraction failure, no special-casing."""
    resolved_pdfalto_bin = pdfalto_runner.resolve_pdfalto_binary(pdfalto_bin)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        ocr_pdf_path = tmp_dir / f"{pdf_path.stem}.ocr.pdf"
        result = subprocess.run(
            ["ocrmypdf", "--force-ocr", "-l", "deu+eng", str(pdf_path), str(ocr_pdf_path)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ocrmypdf failed on {pdf_path}: {result.stderr}")
        alto_path = pdfalto_runner.ensure_alto_xml(ocr_pdf_path, tmp_dir, resolved_pdfalto_bin)
        return _rows_from_alto_xml(alto_path)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_dnb_toc_ocr.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add evaluation/dnb_toc_ocr.py tests/test_dnb_toc_ocr.py
git commit -m "feat: add ALTO-based reading-order reconstruction for OCR'd TOC text

New evaluation/dnb_toc_ocr.py: ocr_pages_to_rows runs ocrmypdf + pdfalto
and reconstructs each page's printed reading order by clustering raw
<String> tokens by VPOS rather than trusting pdfalto's own TextBlock/
TextLine grouping, which the 2026-08-16 investigation found scrambles a
dot-leader TOC's title and page-number columns into separate blocks."
```

---

## Task 2.5: Optional `tessdata_best` support for OCR quality

Added after Task 2 landed, in response to a user request to use tesseract's
higher-accuracy `tessdata_best` language models instead of whatever ships by
default (`tessdata_fast`, on this machine's Homebrew `tesseract-lang`
formula -- confirmed by inspection: Homebrew has no `tessdata_best` formula
at all, so this is never present unless a human downloads it by hand from
https://github.com/tesseract-ocr/tessdata_best per language). A prior
investigation (`docs/superpowers/specs/2026-08-16-dnb-toc-uniform-ocr-design.md`
§1c) tested `tessdata_best` for a *different* downstream consumer (a fragile
regex heuristic) and found it "a mixed, marginal improvement... not
adopted" -- but `text_extract_toc_entries` (Task 3) feeds OCR'd text to an
LLM instructed to read past OCR artifacts, a more tolerant consumer, so the
tradeoff is worth re-offering as an opt-in here rather than assumed settled.

Purely opt-in via a `TESSDATA_BEST_DIR` environment variable pointing at a
directory of `.traineddata` files -- unset (the default) changes nothing,
`ocr_pages_to_rows` behaves exactly as Task 2 built it. Set-but-invalid
(missing a required language's file) raises immediately with a clear
message, rather than either silently falling back or failing deep inside a
cryptic tesseract subprocess error -- same "raise naming exactly what's
wrong" convention `evaluation/inference_endpoints.py`'s
`resolve_endpoint_from_env` already established for a similar
env-var-driven, human-diagnosable setup step.

**Files:**
- Modify: `evaluation/dnb_toc_ocr.py`
- Modify: `tests/test_dnb_toc_ocr.py`
- Modify: `evaluation/README.md`

- [ ] **Step 1: Write the failing tests**

Change `tests/test_dnb_toc_ocr.py`'s imports from:

```python
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from evaluation.dnb_toc_ocr import _rows_from_alto_xml, text_extract_toc_entries
```

to:

```python
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from evaluation.dnb_toc_ocr import _resolve_tessdata_best_env, _rows_from_alto_xml, ocr_pages_to_rows
```

(Note: at this point in the branch's history, `text_extract_toc_entries` doesn't exist yet -- it's Task 3's deliverable, and Task 2.5 was spliced in between Task 2 and Task 3. Do NOT import it here; none of this task's new tests need it, and importing it would break collection.)

Append this test class to the end of `tests/test_dnb_toc_ocr.py`:

```python
class TestResolveTessdataBestEnv(unittest.TestCase):
    def test_returns_none_when_env_var_absent(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TESSDATA_BEST_DIR", None)
            self.assertIsNone(_resolve_tessdata_best_env())

    def test_returns_tessdata_prefix_pointing_at_the_directory_when_all_languages_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "deu.traineddata").write_bytes(b"fake")
            (Path(tmp) / "eng.traineddata").write_bytes(b"fake")
            with patch.dict(os.environ, {"TESSDATA_BEST_DIR": tmp}, clear=False):
                env = _resolve_tessdata_best_env()

        self.assertIsNotNone(env)
        self.assertEqual(env["TESSDATA_PREFIX"], tmp)

    def test_raises_naming_the_missing_language_when_dir_is_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "deu.traineddata").write_bytes(b"fake")
            with patch.dict(os.environ, {"TESSDATA_BEST_DIR": tmp}, clear=False):
                with self.assertRaises(RuntimeError) as ctx:
                    _resolve_tessdata_best_env()

        self.assertIn("eng", str(ctx.exception))


class TestOcrPagesToRowsTessdataWiring(unittest.TestCase):
    def test_passes_the_resolved_tessdata_env_through_to_ocrmypdf(self):
        fake_env = {"TESSDATA_PREFIX": "/fake/tessdata_best"}
        fake_result = MagicMock(returncode=0)
        with patch("evaluation.dnb_toc_ocr._resolve_tessdata_best_env", return_value=fake_env), \
             patch("evaluation.dnb_toc_ocr.subprocess.run", return_value=fake_result) as mock_run, \
             patch("evaluation.dnb_toc_ocr.pdfalto_runner.ensure_alto_xml", return_value=Path("/fake/out.alto.xml")), \
             patch("evaluation.dnb_toc_ocr._rows_from_alto_xml", return_value=["row"]):
            ocr_pages_to_rows(Path("/fake/book.pdf"))

        mock_run.assert_called_once()
        self.assertEqual(mock_run.call_args.kwargs["env"], fake_env)

    def test_passes_none_env_when_tessdata_best_is_not_configured(self):
        fake_result = MagicMock(returncode=0)
        with patch("evaluation.dnb_toc_ocr._resolve_tessdata_best_env", return_value=None), \
             patch("evaluation.dnb_toc_ocr.subprocess.run", return_value=fake_result) as mock_run, \
             patch("evaluation.dnb_toc_ocr.pdfalto_runner.ensure_alto_xml", return_value=Path("/fake/out.alto.xml")), \
             patch("evaluation.dnb_toc_ocr._rows_from_alto_xml", return_value=["row"]):
            ocr_pages_to_rows(Path("/fake/book.pdf"))

        mock_run.assert_called_once()
        self.assertIsNone(mock_run.call_args.kwargs["env"])
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_dnb_toc_ocr.py -v`
Expected: FAIL with `ImportError: cannot import name '_resolve_tessdata_best_env'`

- [ ] **Step 3: Add `_resolve_tessdata_best_env` and wire it into `ocr_pages_to_rows`**

In `evaluation/dnb_toc_ocr.py`, change the imports from:

```python
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
```

to:

```python
import os
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
```

Add this constant and function directly above `def ocr_pages_to_rows`:

```python
_TESSDATA_BEST_DIR_ENV_VAR = "TESSDATA_BEST_DIR"


def _resolve_tessdata_best_env(languages: tuple[str, ...] = ("deu", "eng")) -> dict[str, str] | None:
    """Resolves an optional subprocess environment override pointing
    ocrmypdf/tesseract at a tessdata_best directory instead of whatever
    ships by default (Homebrew's tesseract-lang formula ships
    tessdata_fast only -- there is no Homebrew formula for tessdata_best,
    so this is opt-in via the TESSDATA_BEST_DIR environment variable after
    a manual per-language download from
    https://github.com/tesseract-ocr/tessdata_best -- see
    evaluation/README.md's "Building dnb-toc-only ground truth"). Returns
    None (no env override -- ocrmypdf uses whatever tessdata is already on
    PATH/its default location) when the variable isn't set at all; purely
    opt-in, no default guessed. When it IS set, validates the directory
    actually contains every requested language's .traineddata file and
    raises RuntimeError naming exactly what's missing if not -- a
    misconfigured explicit request should fail loudly with an actionable
    message, not silently fall back to the default or surface as a
    cryptic tesseract error deep inside a subprocess (same
    raise-naming-what's-wrong convention
    evaluation/inference_endpoints.py's resolve_endpoint_from_env already
    established for a similar env-var-driven setup step)."""
    directory = os.environ.get(_TESSDATA_BEST_DIR_ENV_VAR)
    if not directory:
        return None
    missing = [lang for lang in languages if not (Path(directory) / f"{lang}.traineddata").exists()]
    if missing:
        raise RuntimeError(
            f"{_TESSDATA_BEST_DIR_ENV_VAR}={directory} is missing traineddata for: {', '.join(missing)} -- "
            f"download from https://github.com/tesseract-ocr/tessdata_best"
        )
    return {**os.environ, "TESSDATA_PREFIX": directory}
```

Change `ocr_pages_to_rows`'s `subprocess.run` call from:

```python
        result = subprocess.run(
            ["ocrmypdf", "--force-ocr", "-l", "deu+eng", str(pdf_path), str(ocr_pdf_path)],
            capture_output=True, text=True,
        )
```

to:

```python
        result = subprocess.run(
            ["ocrmypdf", "--force-ocr", "-l", "deu+eng", str(pdf_path), str(ocr_pdf_path)],
            capture_output=True, text=True, env=_resolve_tessdata_best_env(),
        )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_dnb_toc_ocr.py -v`
Expected: PASS (all tests, including the 4 pre-existing ones)

- [ ] **Step 5: Document the setup step**

In `evaluation/README.md`, in the "Building dnb-toc-only ground truth" section, insert this new paragraph directly after the existing bulk-tier paragraph that ends "...see this file's 'Cleaning a badly-scanned PDF' section for the install command." (i.e. right before the "**Eval tier**" heading). Insert:

```
The text-extraction side of a vision+text pairing (`--text-endpoint`/
`--text-config-file`, `evaluation/dnb_toc_ocr.py`) OCRs each book's TOC
pages via `ocrmypdf` -- by default using whatever tesseract language data
is already installed (Homebrew's `tesseract-lang` formula ships
`tessdata_fast`). For higher OCR accuracy, download `tessdata_best`'s
`deu.traineddata`/`eng.traineddata` by hand from
https://github.com/tesseract-ocr/tessdata_best into one directory and set
`TESSDATA_BEST_DIR` to that directory's path -- picked up automatically,
with no code change, the next time `ocr_pages_to_rows` runs. Unset (the
default) uses the system's normal tessdata.
```

- [ ] **Step 6: Run the full test suite**

Run: `uv run pytest -q`
Expected: PASS, no regressions.

- [ ] **Step 7: Commit**

```bash
git add evaluation/dnb_toc_ocr.py tests/test_dnb_toc_ocr.py evaluation/README.md
git commit -m "feat: support an opt-in tessdata_best directory for OCR quality

TESSDATA_BEST_DIR, when set, points ocrmypdf at higher-accuracy tesseract
language data instead of the Homebrew default (tessdata_fast) -- a prior
investigation found tessdata_best only marginally helped a fragile regex
heuristic, but the text-extraction pairing feeds an LLM instructed to
read past OCR artifacts, a more tolerant consumer, so it's worth
re-offering as an opt-in. Unset changes nothing; set-but-incomplete
raises immediately naming the missing language."
```

---

## Task 3: `evaluation/dnb_toc_ocr.py` — text extraction prompt and `text_extract_toc_entries`

Adds the LLM-calling half of the module (spec §3): a text-reading variant of `_VISION_TOC_EXTRACTION_PROMPT` and `text_extract_toc_entries`, structurally parallel to `vision_extract_toc_entries` (`evaluation/dnb_toc_vision.py:194`).

**Files:**
- Modify: `evaluation/dnb_toc_ocr.py`
- Modify: `tests/test_dnb_toc_ocr.py`

- [ ] **Step 1: Write the failing tests**

Change `tests/test_dnb_toc_ocr.py`'s imports from:

```python
from evaluation.dnb_toc_ocr import _rows_from_alto_xml
```

to:

```python
from evaluation.dnb_toc_ocr import _rows_from_alto_xml, text_extract_toc_entries
```

Append these helpers and test class to the end of `tests/test_dnb_toc_ocr.py`:

```python
def _fake_text_client(response_text: str):
    message = MagicMock()
    message.content = response_text
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=response)
    return client


_TEXT_RESPONSE = (
    '[{"title": "Einleitung", "authors": [], "printed_page_number": "9", "skip": false}, '
    '{"title": "Schluss", "authors": [], "printed_page_number": "40", "skip": false}]'
)


class TestTextExtractTocEntries(unittest.IsolatedAsyncioTestCase):
    async def test_parses_a_clean_response_into_entries(self):
        client = _fake_text_client(_TEXT_RESPONSE)
        with patch("evaluation.dnb_toc_ocr.ocr_pages_to_rows", return_value=["Einleitung 9", "Schluss 40"]):
            entries = await text_extract_toc_entries(Path("/tmp/book.pdf"), "text-model", client)

        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0].title, "Einleitung")
        self.assertEqual(entries[1].printed_page_number, "40")
        client.chat.completions.create.assert_awaited_once()
        call_kwargs = client.chat.completions.create.await_args.kwargs
        self.assertEqual(call_kwargs["model"], "text-model")
        self.assertIn("Einleitung 9", call_kwargs["messages"][0]["content"])

    async def test_escalates_max_tokens_once_on_a_truncated_first_response(self):
        client = MagicMock()
        good_message = MagicMock()
        good_message.content = _TEXT_RESPONSE
        good_choice = MagicMock()
        good_choice.message = good_message
        good_response = MagicMock()
        good_response.choices = [good_choice]
        bad_message = MagicMock()
        bad_message.content = "[{\"title\": \"truncated"  # not valid JSON, no closing bracket
        bad_choice = MagicMock()
        bad_choice.message = bad_message
        bad_response = MagicMock()
        bad_response.choices = [bad_choice]
        client.chat.completions.create = AsyncMock(side_effect=[bad_response, good_response])

        with patch("evaluation.dnb_toc_ocr.ocr_pages_to_rows", return_value=["Einleitung 9"]):
            entries = await text_extract_toc_entries(Path("/tmp/book.pdf"), "text-model", client)

        self.assertEqual(len(entries), 2)
        self.assertEqual(client.chat.completions.create.await_count, 2)
        first_max_tokens = client.chat.completions.create.await_args_list[0].kwargs["max_tokens"]
        second_max_tokens = client.chat.completions.create.await_args_list[1].kwargs["max_tokens"]
        self.assertLess(first_max_tokens, second_max_tokens)

    async def test_raises_after_both_attempts_fail_to_parse(self):
        client = _fake_text_client("not json at all")
        with patch("evaluation.dnb_toc_ocr.ocr_pages_to_rows", return_value=["garbage"]):
            with self.assertRaises(Exception):
                await text_extract_toc_entries(Path("/tmp/book.pdf"), "text-model", client)
        self.assertEqual(client.chat.completions.create.await_count, 2)

    async def test_ocr_failure_propagates_uncaught(self):
        client = _fake_text_client(_TEXT_RESPONSE)
        with patch("evaluation.dnb_toc_ocr.ocr_pages_to_rows", side_effect=RuntimeError("ocrmypdf failed")):
            with self.assertRaises(RuntimeError):
                await text_extract_toc_entries(Path("/tmp/book.pdf"), "text-model", client)
        client.chat.completions.create.assert_not_called()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_dnb_toc_ocr.py::TestTextExtractTocEntries -v`
Expected: FAIL with `ImportError: cannot import name 'text_extract_toc_entries'`

- [ ] **Step 3: Add the prompt and `text_extract_toc_entries` to `evaluation/dnb_toc_ocr.py`**

Append to `evaluation/dnb_toc_ocr.py`:

```python
_TEXT_TOC_EXTRACTION_PROMPT = """\
You are reading a machine-OCR'd transcription of a book's table of \
contents pages, with each printed line's reading order already \
reconstructed for you. The OCR process can still introduce scanning \
artifacts: misrecognized characters, a run of garbled tokens where a \
printed dot leader ("....") was misread as text, or an occasional \
dropped or duplicated word. Read past such artifacts and report the \
actual title/page-number content a human would recognize the line as \
saying -- do not transcribe OCR noise literally as if it were real text.

A heading can have indented, numbered, or lettered sub-points listed \
below it (e.g. "I.", "II.", "1.", "2.") that each carry their OWN page \
number -- each such sub-point is its own separate entry too, exactly as \
printed, not merged into its parent heading. Do not collapse or omit a \
sub-point just because it is indented under a larger heading.

A single chapter's title sometimes spans two printed lines -- a short \
main title followed by a longer explanatory subtitle right below it \
(or vice versa) -- with only ONE page number for the pair. That is ONE \
entry, not two: join both lines into a single title string. Do not \
create a separate entry for the subtitle line, and do not create a \
separate entry with no page number just because a line of text sits \
above a chapter's title.

Return ONLY a JSON array, one entry for EVERY line in the transcription \
that names a titled section and (usually) a page number -- transcribe \
what is actually printed, do not decide which lines matter. This \
includes lines you might not think of as a "real chapter": a \
part/section divider (e.g. "Teil 1", "I. Historische Grundlagen", an \
unnumbered section-title line that groups several chapters under it), \
front matter (preface, foreword, acknowledgements, list of \
contributors/authors), and back matter (bibliography, index, an \
appendix listing an author's or honoree's own prior publications) all \
get their own entry too, exactly like any other line, even when they \
carry no page number of their own. Mark each entry "skip": true if it is \
one of these non-chapter lines (a divider, front matter, or back \
matter) and "skip": false if it is an actual chapter -- but include the \
entry either way; never omit a printed line because of what "skip" \
value it would get:
[{"title": "...", "authors": ["First Last", ...], "printed_page_number": "12", "skip": false}]

printed_page_number is the page number exactly AS PRINTED on the page -- \
copy it verbatim, including roman numerals for front-matter chapters \
(e.g. "vii", not 7). If a line's printed page number is not visible, use \
null for printed_page_number -- never leave the line out just because it \
has no page number. If authors are not identifiable, use an empty list.

If a title is printed with a leading number, letter, or label (e.g. "1 ", \
"2.3 ", "I. ", "a) "), that label is part of the title -- include it \
verbatim as the start of the title string. Do not strip, renumber, or \
omit any such printed label."""

# Same escalation shape as evaluation/dnb_toc_vision.py's
# _VISION_MAX_TOKENS/_VISION_MAX_TOKENS_RETRY -- a truncated JSON array
# reliably fails parse_json_array regardless of cause, so JSON-parseability
# alone is a sufficient retry trigger. A text response has no image tokens
# inflating the prompt, so budget pressure here is milder than vision, but
# the escalation costs nothing to keep for consistency.
_TEXT_MAX_TOKENS = 4096
_TEXT_MAX_TOKENS_RETRY = 8192


async def text_extract_toc_entries(
    pdf_path: Path, model: str, client: Any, *, pdfalto_bin: str | None = None,
) -> list[TocEntry]:
    """OCRs pdf_path (ocr_pages_to_rows) and asks a text-only model (via an
    already-constructed openai.AsyncOpenAI-shaped `client`, model id
    `model`) to extract the table of contents from the reconstructed page
    text. Same return shape as vision_extract_toc_entries, sharing its
    item-parsing tolerance logic (_toc_items_to_entries) and its
    raises-on-failure/max_tokens-escalation contract -- see that
    function's own docstring in evaluation/dnb_toc_vision.py for why
    swallowing failures internally would be wrong here too. Does not catch
    exceptions from ocr_pages_to_rows -- an OCR failure propagates exactly
    like any other extraction failure, no special-casing (design spec
    section "Error handling")."""
    page_texts = ocr_pages_to_rows(pdf_path, pdfalto_bin=pdfalto_bin)
    pages_block = "\n\n".join(f"--- Page {i + 1} ---\n{text}" for i, text in enumerate(page_texts))
    prompt = f"{_TEXT_TOC_EXTRACTION_PROMPT}\n\n{pages_block}"
    llm_client = OpenAICompatibleLLMClient(model=model, client=client)

    last_error: Exception | None = None
    for max_tokens in (_TEXT_MAX_TOKENS, _TEXT_MAX_TOKENS_RETRY):
        raw = await llm_client.generate(prompt, max_tokens=max_tokens, temperature=0.0)
        try:
            items = parse_json_array(raw)
            return _toc_items_to_entries(items)
        except Exception as exc:  # noqa: BLE001 -- any parse failure triggers the escalation retry
            last_error = exc
    raise last_error
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_dnb_toc_ocr.py -v`
Expected: PASS (8 tests total)

- [ ] **Step 5: Commit**

```bash
git add evaluation/dnb_toc_ocr.py tests/test_dnb_toc_ocr.py
git commit -m "feat: add text_extract_toc_entries for OCR'd-text TOC extraction

Structurally parallel to vision_extract_toc_entries: same verbatim/skip
JSON schema, same max_tokens escalation on a parse failure, same
raises-on-failure contract. The prompt is a text-reading adaptation of
_VISION_TOC_EXTRACTION_PROMPT (evaluation/dnb_toc_vision.py), not the
older, stale _LLM_TOC_EXTRACTION_PROMPT."
```

---

## Task 4: `"kind"` cache field in `evaluation/dnb_toc_vision.py`

Adds an optional `"kind": "vision"|"text"` field to the cache schema (spec §4) so `arbitrate_dnb_toc.py` (Task 7) can label which extraction path produced a cached entry. Absent on any pre-existing cache file, treated as `"vision"` -- fully backward compatible, no cache-schema version bump.

**Files:**
- Modify: `evaluation/dnb_toc_vision.py`
- Modify: `tests/test_dnb_toc_vision.py`

- [ ] **Step 1: Write the failing tests**

In `tests/test_dnb_toc_vision.py`, change the top-level imports from:

```python
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from pypdf import PdfWriter

from chapter_segmentation.segmentation import TocEntry
from evaluation.dnb_toc_vision import (
    _MAX_VISION_PAGES,
    cache_path,
    load_cached_llm_entries,
    render_pages_to_images,
    versioned_cache_dir,
    vision_extract_toc_entries,
    write_cached_llm_entries,
)
```

to:

```python
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from pypdf import PdfWriter

from chapter_segmentation.segmentation import TocEntry
from evaluation.dnb_toc_vision import (
    _MAX_VISION_PAGES,
    cache_path,
    load_cached_kind,
    load_cached_llm_entries,
    render_pages_to_images,
    versioned_cache_dir,
    vision_extract_toc_entries,
    write_cached_llm_entries,
)
```

Append this test class to the end of `tests/test_dnb_toc_vision.py`:

```python
class TestLoadCachedKind(unittest.TestCase):
    def test_defaults_to_vision_when_kind_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            path = cache_path(cache_dir, "book1", "model-a")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"generated_at": 0, "entries": []}), encoding="utf-8")

            self.assertEqual(load_cached_kind(cache_dir, "book1", "model-a"), "vision")

    def test_returns_the_written_kind(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            entries = [TocEntry(title="X", printed_page_number=1, source_page_index=0)]
            write_cached_llm_entries(cache_dir, "book2", "model-a", entries, kind="text")

            self.assertEqual(load_cached_kind(cache_dir, "book2", "model-a"), "text")

    def test_defaults_to_vision_for_a_missing_cache_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(load_cached_kind(Path(tmp), "book3", "model-a"), "vision")

    def test_write_without_kind_argument_defaults_to_vision(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            entries = [TocEntry(title="X", printed_page_number=1, source_page_index=0)]
            write_cached_llm_entries(cache_dir, "book4", "model-a", entries)

            self.assertEqual(load_cached_kind(cache_dir, "book4", "model-a"), "vision")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_dnb_toc_vision.py::TestLoadCachedKind -v`
Expected: FAIL with `ImportError: cannot import name 'load_cached_kind'`

- [ ] **Step 3: Add the `kind` field to `write_cached_llm_entries` and add `load_cached_kind`**

In `evaluation/dnb_toc_vision.py`, change `write_cached_llm_entries` from:

```python
def write_cached_llm_entries(cache_directory: Path, key: str, model: str, entries: list[TocEntry]) -> None:
    """Caches entries for (key, model). Callers should only call this
    with a non-empty entries list -- an empty result could be a genuine
    "no TOC content" or a transient failure, and caching it either way
    would make a later re-run trust a possibly-transient empty result
    forever instead of retrying."""
    path = cache_path(cache_directory, key, model)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "generated_at": time.time(),
        "entries": [
            {
                "title": e.title, "printed_page_number": e.printed_page_number,
                "source_page_index": e.source_page_index, "authors": list(e.authors),
                "printed_roman": e.printed_roman, "skip": e.skip,
            }
            for e in entries
        ],
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)
```

to:

```python
def write_cached_llm_entries(
    cache_directory: Path, key: str, model: str, entries: list[TocEntry], *, kind: str = "vision",
) -> None:
    """Caches entries for (key, model). Callers should only call this
    with a non-empty entries list -- an empty result could be a genuine
    "no TOC content" or a transient failure, and caching it either way
    would make a later re-run trust a possibly-transient empty result
    forever instead of retrying. `kind` ("vision" or "text", default
    "vision") records which extraction path produced these entries --
    see load_cached_kind's own docstring for how a caller reads it back."""
    path = cache_path(cache_directory, key, model)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "generated_at": time.time(),
        "kind": kind,
        "entries": [
            {
                "title": e.title, "printed_page_number": e.printed_page_number,
                "source_page_index": e.source_page_index, "authors": list(e.authors),
                "printed_roman": e.printed_roman, "skip": e.skip,
            }
            for e in entries
        ],
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)
```

Add this function directly below `load_cached_llm_entries`:

```python
def load_cached_kind(cache_directory: Path, key: str, model: str) -> str:
    """Which extraction path produced (key, model)'s cached entries --
    "vision" or "text". Reads the "kind" field write_cached_llm_entries
    writes; absent on any cache file written before this field existed
    (every cache file to date came from vision_extract_toc_entries), which
    is treated as "vision" for full backward compatibility -- see design
    spec docs/superpowers/specs/2026-08-20-dnb-toc-vision-text-pairing-
    design.md section 4. Also returns "vision" if the cache file doesn't
    exist at all -- callers (arbitrate_dnb_toc.py) only ever call this for
    a (key, model) pair they already know has a cache file, but this keeps
    the function total rather than raising on a caller's bug."""
    path = cache_path(cache_directory, key, model)
    if not path.exists():
        return "vision"
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("kind", "vision")
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_dnb_toc_vision.py -v`
Expected: PASS, all tests including the pre-existing round-trip tests (they don't pass `kind`, so they exercise the new default).

- [ ] **Step 5: Commit**

```bash
git add evaluation/dnb_toc_vision.py tests/test_dnb_toc_vision.py
git commit -m "feat: add optional kind field to the dnb-toc-only LLM cache schema

write_cached_llm_entries gains a kind='vision'|'text' keyword (default
'vision', fully backward compatible with every existing cache file);
load_cached_kind reads it back, defaulting to 'vision' when absent. Sets
up arbitrate_dnb_toc.py's source-kind labeling (next task)."
```

---

## Task 5: `_resolve_endpoints` and CLI flags in `generate_dnb_toc_ground_truth.py`

Implements spec §1's combination table: `--text-endpoint`/`--text-config-file` pair one vision endpoint with one text endpoint; today's zero/two-vision-endpoint paths are unchanged (delegated to the existing `_resolve_vision_endpoints`, untouched).

**Files:**
- Modify: `evaluation/scripts/generate_dnb_toc_ground_truth.py`
- Modify: `tests/test_generate_dnb_toc_ground_truth.py`

- [ ] **Step 1: Write the failing tests**

In `tests/test_generate_dnb_toc_ground_truth.py`, change the import from `evaluation.scripts.generate_dnb_toc_ground_truth` from:

```python
from evaluation.scripts.generate_dnb_toc_ground_truth import (
    _acquire_lock,
    _binding_rate_limit_window,
    _call_with_retry,
    _is_stale_bulk_gate_entry,
    _lock_path,
    _release_lock,
    _resolve_vision_endpoints,
    _retry_after_seconds,
    _run_book,
    _run_book_entries,
    _select_best_models,
    _still_needs_a_decision,
)
```

to:

```python
from evaluation.scripts.generate_dnb_toc_ground_truth import (
    _acquire_lock,
    _binding_rate_limit_window,
    _call_with_retry,
    _is_stale_bulk_gate_entry,
    _lock_path,
    _release_lock,
    _resolve_endpoints,
    _resolve_vision_endpoints,
    _retry_after_seconds,
    _run_book,
    _run_book_entries,
    _select_best_models,
    _still_needs_a_decision,
)
```

Append this test class to the end of `tests/test_generate_dnb_toc_ground_truth.py`:

```python
class TestResolveEndpoints(unittest.TestCase):
    def test_no_text_side_delegates_to_resolve_vision_endpoints(self):
        env = {
            "MPCDF_A_BASE_URL": "https://a.invalid/v1", "MPCDF_A_API_KEY": "ka", "MPCDF_A_MODEL": "model-a",
            "MPCDF_B_BASE_URL": "https://b.invalid/v1", "MPCDF_B_API_KEY": "kb", "MPCDF_B_MODEL": "model-b",
        }
        with patch.dict(os.environ, env, clear=False):
            vision, second, kind = _resolve_endpoints(["MPCDF_A", "MPCDF_B"], None, None, None)

        self.assertEqual(kind, "vision")
        self.assertEqual(vision.model_id, "model-a")
        self.assertEqual(second.model_id, "model-b")

    def test_one_vision_alias_plus_one_text_alias_pairs_them(self):
        env = {
            "MPCDF_A_BASE_URL": "https://a.invalid/v1", "MPCDF_A_API_KEY": "ka", "MPCDF_A_MODEL": "vision-model",
            "TEXT_A_BASE_URL": "https://c.invalid/v1", "TEXT_A_API_KEY": "kc", "TEXT_A_MODEL": "text-model",
        }
        with patch.dict(os.environ, env, clear=False):
            vision, second, kind = _resolve_endpoints(["MPCDF_A"], None, "TEXT_A", None)

        self.assertEqual(kind, "text")
        self.assertEqual(vision.model_id, "vision-model")
        self.assertEqual(second.model_id, "text-model")

    def test_vision_config_file_plus_text_config_file_pairs_them(self):
        vision_table = "framework_args\t--model=vision-model\nkey\tkv\nurl\thttps://v.invalid/v1\n"
        text_table = "framework_args\t--model=text-model\nkey\tkt\nurl\thttps://t.invalid/v1\n"
        with tempfile.TemporaryDirectory() as tmp:
            vision_path = Path(tmp) / "vision.txt"
            text_path = Path(tmp) / "text.txt"
            vision_path.write_text(vision_table)
            text_path.write_text(text_table)

            vision, second, kind = _resolve_endpoints(None, vision_path, None, text_path)

        self.assertEqual(kind, "text")
        self.assertEqual(vision.model_id, "vision-model")
        self.assertEqual(second.model_id, "text-model")

    def test_vision_endpoint_alias_plus_text_config_file_can_be_mixed(self):
        env = {"MPCDF_A_BASE_URL": "https://a.invalid/v1", "MPCDF_A_API_KEY": "ka", "MPCDF_A_MODEL": "vision-model"}
        text_table = "framework_args\t--model=text-model\nkey\tkt\nurl\thttps://t.invalid/v1\n"
        with tempfile.TemporaryDirectory() as tmp:
            text_path = Path(tmp) / "text.txt"
            text_path.write_text(text_table)
            with patch.dict(os.environ, env, clear=False):
                vision, second, kind = _resolve_endpoints(["MPCDF_A"], None, None, text_path)

        self.assertEqual(kind, "text")
        self.assertEqual(vision.model_id, "vision-model")
        self.assertEqual(second.model_id, "text-model")

    def test_two_vision_aliases_plus_a_text_alias_is_a_user_error(self):
        with self.assertRaises(SystemExit):
            _resolve_endpoints(["MPCDF_A", "MPCDF_B"], None, "TEXT_A", None)

    def test_text_alias_with_no_vision_side_is_a_user_error(self):
        with self.assertRaises(SystemExit):
            _resolve_endpoints(None, None, "TEXT_A", None)

    def test_text_config_file_with_two_tables_is_a_user_error(self):
        one_table = "framework_args\t--model=vision-model\nkey\tkv\nurl\thttps://v.invalid/v1\n"
        with tempfile.TemporaryDirectory() as tmp:
            vision_path = Path(tmp) / "vision.txt"
            text_path = Path(tmp) / "text.txt"
            vision_path.write_text(one_table)
            text_path.write_text(one_table + "\n" + one_table)

            with self.assertRaises(SystemExit):
                _resolve_endpoints(None, vision_path, None, text_path)

    def test_vision_config_file_with_two_tables_plus_a_text_alias_is_a_user_error(self):
        one_table = "framework_args\t--model=x\nkey\tk\nurl\thttps://a.invalid/v1\n"
        with tempfile.TemporaryDirectory() as tmp:
            vision_path = Path(tmp) / "vision.txt"
            vision_path.write_text(one_table + "\n" + one_table)

            with self.assertRaises(SystemExit):
                _resolve_endpoints(None, vision_path, "TEXT_A", None)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_generate_dnb_toc_ground_truth.py::TestResolveEndpoints -v`
Expected: FAIL with `ImportError: cannot import name '_resolve_endpoints'`

- [ ] **Step 3: Add `_resolve_endpoints` and the new CLI flags**

In `evaluation/scripts/generate_dnb_toc_ground_truth.py`, add `text_extract_toc_entries` to the `dnb_toc_ocr`/`dnb_toc_vision` imports. Change:

```python
from chapter_segmentation.segmentation import TocEntry
from evaluation.dnb_toc_matching import gate_book, toc_entry_to_gt_dict
from evaluation.dnb_toc_vision import load_cached_llm_entries, vision_extract_toc_entries, write_cached_llm_entries
from evaluation.harness import corpus_dir, llm_cache_dir, load_manifest_books
```

to:

```python
from chapter_segmentation.segmentation import TocEntry
from evaluation.dnb_toc_matching import gate_book, toc_entry_to_gt_dict
from evaluation.dnb_toc_ocr import text_extract_toc_entries
from evaluation.dnb_toc_vision import load_cached_llm_entries, vision_extract_toc_entries, write_cached_llm_entries
from evaluation.harness import corpus_dir, llm_cache_dir, load_manifest_books
```

Add this function directly below `_resolve_vision_endpoints`:

```python
def _resolve_endpoints(
    endpoint_aliases: Optional[list[str]],
    config_file: Optional[Path],
    text_endpoint_alias: Optional[str],
    text_config_file: Optional[Path],
) -> tuple[ModelEndpoint, ModelEndpoint, str]:
    """Resolves the gate's two endpoints plus which extraction path the
    second one needs ("vision" for vision_extract_toc_entries, "text" for
    text_extract_toc_entries) -- see design spec
    docs/superpowers/specs/2026-08-20-dnb-toc-vision-text-pairing-design.md
    section 1's combination table. Neither --text-endpoint nor
    --text-config-file given delegates entirely to _resolve_vision_endpoints
    (today's two-vision-model behavior, completely unchanged), second_kind
    "vision". Either text flag given pairs exactly one vision-side endpoint
    (--endpoint with exactly 1 alias, or --config-file with exactly 1
    pasted session table) with exactly one text-side endpoint
    (--text-endpoint, or --text-config-file with exactly 1 table),
    second_kind "text" -- the vision and text sides may use different
    sourcing mechanisms freely (e.g. vision via --endpoint, text via
    --text-config-file). Any other shape (e.g. 2 vision endpoints ALSO
    given a text endpoint, or a text flag with no vision side at all) is a
    user error, raising SystemExit naming exactly what's wrong -- same
    style as _resolve_vision_endpoints' own existing errors."""
    if not text_endpoint_alias and not text_config_file:
        vision_a, vision_b = _resolve_vision_endpoints(endpoint_aliases, config_file)
        return vision_a, vision_b, "vision"

    if config_file:
        vision_endpoints = resolve_endpoints_from_config_file(config_file)
        if len(vision_endpoints) != 1:
            raise SystemExit(
                f"--config-file paired with --text-endpoint/--text-config-file requires exactly 1 pasted "
                f"session table for the vision side, got {len(vision_endpoints)} in {config_file}"
            )
        vision_endpoint = vision_endpoints[0]
    elif endpoint_aliases:
        if len(endpoint_aliases) != 1:
            raise SystemExit(
                f"--endpoint paired with --text-endpoint/--text-config-file requires exactly 1 alias for the "
                f"vision side, got {len(endpoint_aliases)}: {endpoint_aliases}"
            )
        vision_endpoint = resolve_endpoint_from_env(endpoint_aliases[0])
    else:
        raise SystemExit(
            "--text-endpoint/--text-config-file requires a vision-side --endpoint or --config-file too -- "
            "the gate needs one vision read and one text read"
        )

    if text_config_file:
        text_endpoints = resolve_endpoints_from_config_file(text_config_file)
        if len(text_endpoints) != 1:
            raise SystemExit(
                f"--text-config-file requires exactly 1 pasted session table, got {len(text_endpoints)} in "
                f"{text_config_file}"
            )
        text_endpoint = text_endpoints[0]
    else:
        text_endpoint = resolve_endpoint_from_env(text_endpoint_alias)

    return vision_endpoint, text_endpoint, "text"
```

Add the new flags in `main()`. Change:

```python
    endpoint_group = parser.add_mutually_exclusive_group()
    endpoint_group.add_argument(
        "--endpoint", action="append", default=None, metavar="ALIAS",
        help="Use an explicit OpenAI-compatible endpoint instead of KISSKI auto-discovery -- pass exactly twice "
             "(the gate needs two independent reads), e.g. --endpoint MPCDF_A --endpoint MPCDF_B. Each ALIAS must "
             "have <ALIAS>_BASE_URL, <ALIAS>_API_KEY, <ALIAS>_MODEL set in the environment.",
    )
    endpoint_group.add_argument(
        "--config-file", nargs="?", const=DEFAULT_SESSIONS_FILENAME, default=None, metavar="PATH",
        help="Same as --endpoint, but sources both endpoints from a pasted-session-table file instead of env "
             f"vars -- PATH defaults to {DEFAULT_SESSIONS_FILENAME} when omitted; must contain exactly 2 pasted "
             "session tables. See evaluation/hpc/llm-mpcdf.md.",
    )
    args = parser.parse_args()
    if args.config_file:
        args.config_file = Path(args.config_file)
```

to:

```python
    endpoint_group = parser.add_mutually_exclusive_group()
    endpoint_group.add_argument(
        "--endpoint", action="append", default=None, metavar="ALIAS",
        help="Use an explicit OpenAI-compatible endpoint instead of KISSKI auto-discovery for the VISION side -- "
             "pass exactly twice for two independent vision reads (e.g. --endpoint MPCDF_A --endpoint MPCDF_B), "
             "or exactly once when paired with --text-endpoint/--text-config-file. Each ALIAS must have "
             "<ALIAS>_BASE_URL, <ALIAS>_API_KEY, <ALIAS>_MODEL set in the environment.",
    )
    endpoint_group.add_argument(
        "--config-file", nargs="?", const=DEFAULT_SESSIONS_FILENAME, default=None, metavar="PATH",
        help="Same as --endpoint, but sources the vision endpoint(s) from a pasted-session-table file instead of "
             f"env vars -- PATH defaults to {DEFAULT_SESSIONS_FILENAME} when omitted; must contain exactly 2 "
             "pasted session tables (two vision reads), or exactly 1 when paired with "
             "--text-endpoint/--text-config-file. See evaluation/hpc/llm-mpcdf.md.",
    )
    text_group = parser.add_mutually_exclusive_group()
    text_group.add_argument(
        "--text-endpoint", default=None, metavar="ALIAS",
        help="Pair the vision endpoint (--endpoint or --config-file, exactly 1 either way) with a text-only "
             "endpoint fed freshly-OCR'd page text instead of a second vision read -- ALIAS must have "
             "<ALIAS>_BASE_URL, <ALIAS>_API_KEY, <ALIAS>_MODEL set in the environment. See design spec "
             "docs/superpowers/specs/2026-08-20-dnb-toc-vision-text-pairing-design.md.",
    )
    text_group.add_argument(
        "--text-config-file", nargs="?", const=DEFAULT_SESSIONS_FILENAME, default=None, metavar="PATH",
        help="Same as --text-endpoint, but sources the text endpoint from a pasted-session-table file -- PATH "
             f"defaults to {DEFAULT_SESSIONS_FILENAME} when omitted; must contain exactly 1 pasted session table.",
    )
    args = parser.parse_args()
    if args.config_file:
        args.config_file = Path(args.config_file)
    if args.text_config_file:
        args.text_config_file = Path(args.text_config_file)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_generate_dnb_toc_ground_truth.py::TestResolveEndpoints tests/test_generate_dnb_toc_ground_truth.py::TestResolveVisionEndpoints -v`
Expected: PASS (existing `TestResolveVisionEndpoints` tests still pass unchanged; new `TestResolveEndpoints` tests pass)

Run: `uv run python evaluation/scripts/generate_dnb_toc_ground_truth.py --help`
Expected: shows `--text-endpoint`/`--text-config-file` alongside `--endpoint`/`--config-file`, no argparse errors.

- [ ] **Step 5: Commit**

```bash
git add evaluation/scripts/generate_dnb_toc_ground_truth.py tests/test_generate_dnb_toc_ground_truth.py
git commit -m "feat: add --text-endpoint/--text-config-file to generate_dnb_toc_ground_truth.py

_resolve_endpoints implements the vision+text combination table: no text
flag delegates unchanged to the existing two-vision-model
_resolve_vision_endpoints; either text flag pairs exactly one vision
endpoint with exactly one text endpoint (sourcing mechanisms may be
mixed freely); any other shape is a SystemExit naming what's wrong. Not
yet wired into _run_book/_generate -- next task."
```

---

## Task 6: Wire `second_kind` through `_run_book`/`_run_all`/`_generate`

Dispatches the second endpoint's extraction call to `vision_extract_toc_entries` or `text_extract_toc_entries` based on `_resolve_endpoints`' `second_kind`, and tags cache writes with the right `kind`. `second_kind` is added as a **keyword-only parameter defaulting to `"vision"`** everywhere, so every existing `_run_book(...)` call site in the test suite (none of which pass `sleep` positionally) keeps working unchanged.

**Files:**
- Modify: `evaluation/scripts/generate_dnb_toc_ground_truth.py`
- Modify: `tests/test_generate_dnb_toc_ground_truth.py`

- [ ] **Step 1: Write the failing test**

In `tests/test_generate_dnb_toc_ground_truth.py`, change the `from evaluation.dnb_toc_vision import ...` line from:

```python
from evaluation.dnb_toc_vision import load_cached_llm_entries, write_cached_llm_entries
```

to:

```python
from evaluation.dnb_toc_vision import load_cached_kind, load_cached_llm_entries, write_cached_llm_entries
```

Append this test to the `TestRunBook` class (after `test_a_book_whose_lock_is_already_held_is_skipped_without_calling_any_model` or anywhere else inside the class body):

```python
    async def test_second_kind_text_dispatches_to_text_extract_toc_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            corpus_directory = tmp_path / "corpus"
            cache_directory = tmp_path / "cache"
            corpus_directory.mkdir()
            pdf_path = _make_pdf(tmp_path / "book.pdf")
            client = _fake_vision_client(_VISION_RESPONSE)
            endpoints = (_endpoint("vision-model", client), _endpoint("text-model", client))
            semaphore = asyncio.Semaphore(1)

            with patch(
                "evaluation.scripts.generate_dnb_toc_ground_truth.text_extract_toc_entries",
                new=AsyncMock(return_value=[_entry("Einleitung", 9), _entry("Schluss", 40)]),
            ) as mock_text_extract:
                key, passed, reason = await _run_book(
                    "book10", pdf_path, endpoints, semaphore, corpus_directory, cache_directory,
                    second_kind="text", sleep=AsyncMock(),
                )

            self.assertTrue(passed)
            mock_text_extract.assert_awaited_once()
            self.assertEqual(client.chat.completions.create.await_count, 1)
            self.assertEqual(load_cached_llm_entries(cache_directory, "book10", "text-model")[0].title, "Einleitung")
            self.assertEqual(load_cached_kind(cache_directory, "book10", "vision-model"), "vision")
            self.assertEqual(load_cached_kind(cache_directory, "book10", "text-model"), "text")

    async def test_second_kind_defaults_to_vision_when_not_given(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            corpus_directory = tmp_path / "corpus"
            cache_directory = tmp_path / "cache"
            corpus_directory.mkdir()
            pdf_path = _make_pdf(tmp_path / "book.pdf")
            client = _fake_vision_client(_VISION_RESPONSE)
            endpoints = (_endpoint("model-a", client), _endpoint("model-b", client))
            semaphore = asyncio.Semaphore(1)

            key, passed, reason = await _run_book(
                "book11", pdf_path, endpoints, semaphore, corpus_directory, cache_directory, sleep=AsyncMock(),
            )

            self.assertTrue(passed)
            self.assertEqual(load_cached_kind(cache_directory, "book11", "model-a"), "vision")
            self.assertEqual(load_cached_kind(cache_directory, "book11", "model-b"), "vision")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_generate_dnb_toc_ground_truth.py::TestRunBook::test_second_kind_text_dispatches_to_text_extract_toc_entries -v`
Expected: FAIL -- `_run_book()` has no `second_kind` keyword yet (`TypeError: _run_book() got an unexpected keyword argument 'second_kind'`), and there's nothing at `evaluation.scripts.generate_dnb_toc_ground_truth.text_extract_toc_entries` to patch yet (it's imported as a module-level name by Task 5's Step 3, so the patch target itself already resolves -- the failure here is the `TypeError`).

- [ ] **Step 3: Update `_run_book`, `_run_all`, and `_generate`**

In `evaluation/scripts/generate_dnb_toc_ground_truth.py`, change `_run_book`'s signature and its endpoint-processing loop. From:

```python
async def _run_book(
    key: str, pdf_path: Path, endpoints: tuple[ModelEndpoint, ModelEndpoint], semaphore: asyncio.Semaphore,
    corpus_directory: Path, cache_directory: Path, sleep=asyncio.sleep,
) -> tuple[str, bool, str]:
```

to:

```python
async def _run_book(
    key: str, pdf_path: Path, endpoints: tuple[ModelEndpoint, ModelEndpoint], semaphore: asyncio.Semaphore,
    corpus_directory: Path, cache_directory: Path, *, second_kind: str = "vision", sleep=asyncio.sleep,
) -> tuple[str, bool, str]:
```

Add one sentence to `_run_book`'s existing docstring (insert right after its first paragraph, before the `semaphore` paragraph) -- change:

```python
    """Thin I/O wrapper around _run_book_entries -- calls
    vision_extract_toc_entries once per endpoint (through the cache, then
    _call_with_retry on a miss), and delegates the two resulting entry
    lists to _run_book_entries. `endpoints` carries each side's own
    client, not a single shared one -- the two independent vision reads
    can come from entirely different inference endpoints (e.g. two MPCDF
    sessions, or one MPCDF + one KISSKI model), not just two models
    behind KISSKI's single base URL. Catches any exception (a corrupt/
    unreadable PDF, a network error that survives _call_with_retry's own
    retries, etc.) and reports it as a failed-but-tuple-shaped result
    instead of letting it propagate -- same "catch-log-continue"
    convention evaluation/refresh_llm_cache.py already established for
    this kind of long, unattended, budget-spending batch job. One book's
    failure must never abort the rest of a ~1000-book run.
```

to:

```python
    """Thin I/O wrapper around _run_book_entries -- calls
    vision_extract_toc_entries for the first endpoint and, per
    `second_kind`, either vision_extract_toc_entries ("vision", the
    default) or text_extract_toc_entries ("text") for the second endpoint
    (through the cache, then _call_with_retry on a miss), and delegates
    the two resulting entry lists to _run_book_entries. `endpoints` carries
    each side's own client, not a single shared one -- the two independent
    reads can come from entirely different inference endpoints (e.g. two
    MPCDF sessions, or one MPCDF + one KISSKI model), not just two models
    behind KISSKI's single base URL. Catches any exception (a corrupt/
    unreadable PDF, a network error that survives _call_with_retry's own
    retries, etc.) and reports it as a failed-but-tuple-shaped result
    instead of letting it propagate -- same "catch-log-continue"
    convention evaluation/refresh_llm_cache.py already established for
    this kind of long, unattended, budget-spending batch job. One book's
    failure must never abort the rest of a ~1000-book run.
```

Change the body's endpoint loop. From:

```python
    try:
        entries_by_model = []
        for endpoint in endpoints:
            cached = load_cached_llm_entries(cache_directory, key, endpoint.model_id)
            if cached is not None:
                entries = cached
            else:
                async def _call(ep=endpoint):
                    async with semaphore:
                        return await vision_extract_toc_entries(pdf_path, ep.model_id, ep.client)
                entries = await _call_with_retry(_call, sleep=sleep)
                # Only cache a non-empty result -- an empty list here
                # could be a genuine "no TOC content on these pages" or
                # a transient failure already exhausted by
                # _call_with_retry; caching it either way would make a
                # later re-run trust a possibly-transient empty result
                # forever instead of retrying.
                if entries:
                    write_cached_llm_entries(cache_directory, key, endpoint.model_id, entries)
            entries_by_model.append(entries)
        return _run_book_entries(key, entries_by_model[0], entries_by_model[1], corpus_directory)
```

to:

```python
    try:
        entries_by_model = []
        for endpoint, kind in zip(endpoints, ("vision", second_kind)):
            cached = load_cached_llm_entries(cache_directory, key, endpoint.model_id)
            if cached is not None:
                entries = cached
            else:
                async def _call(ep=endpoint, k=kind):
                    async with semaphore:
                        if k == "text":
                            return await text_extract_toc_entries(pdf_path, ep.model_id, ep.client)
                        return await vision_extract_toc_entries(pdf_path, ep.model_id, ep.client)
                entries = await _call_with_retry(_call, sleep=sleep)
                # Only cache a non-empty result -- an empty list here
                # could be a genuine "no TOC content on these pages" or
                # a transient failure already exhausted by
                # _call_with_retry; caching it either way would make a
                # later re-run trust a possibly-transient empty result
                # forever instead of retrying.
                if entries:
                    write_cached_llm_entries(cache_directory, key, endpoint.model_id, entries, kind=kind)
            entries_by_model.append(entries)
        return _run_book_entries(key, entries_by_model[0], entries_by_model[1], corpus_directory)
```

Change `_run_all`. From:

```python
async def _run_all(
    keys_and_paths: list[tuple[str, Path]], endpoints: tuple[ModelEndpoint, ModelEndpoint], concurrency: int,
    corpus_directory: Path, cache_directory: Path,
) -> list[tuple[str, bool, str]]:
    semaphore = asyncio.Semaphore(concurrency)
    return list(await asyncio.gather(*[
        _run_book(key, path, endpoints, semaphore, corpus_directory, cache_directory)
        for key, path in keys_and_paths
    ]))
```

to:

```python
async def _run_all(
    keys_and_paths: list[tuple[str, Path]], endpoints: tuple[ModelEndpoint, ModelEndpoint], concurrency: int,
    corpus_directory: Path, cache_directory: Path, *, second_kind: str = "vision",
) -> list[tuple[str, bool, str]]:
    semaphore = asyncio.Semaphore(concurrency)
    return list(await asyncio.gather(*[
        _run_book(key, path, endpoints, semaphore, corpus_directory, cache_directory, second_kind=second_kind)
        for key, path in keys_and_paths
    ]))
```

Change `_generate`'s endpoint resolution and reporting. From:

```python
    endpoints = _resolve_vision_endpoints(args.endpoint, args.config_file)

    results = asyncio.run(_run_all(candidates, endpoints, args.concurrency, cdir, llm_cache_dir(_CORPUS_NAME)))
    passed = [r for r in results if r[1]]
    by_reason: dict[str, int] = {}
    for _, ok, reason in results:
        if not ok:
            by_reason[reason] = by_reason.get(reason, 0) + 1
    print(
        f"Vision models used: {endpoints[0].label}:{endpoints[0].model_id}, "
        f"{endpoints[1].label}:{endpoints[1].model_id}"
    )
```

to:

```python
    vision_endpoint, second_endpoint, second_kind = _resolve_endpoints(
        args.endpoint, args.config_file, args.text_endpoint, args.text_config_file,
    )
    endpoints = (vision_endpoint, second_endpoint)

    results = asyncio.run(
        _run_all(candidates, endpoints, args.concurrency, cdir, llm_cache_dir(_CORPUS_NAME), second_kind=second_kind)
    )
    passed = [r for r in results if r[1]]
    by_reason: dict[str, int] = {}
    for _, ok, reason in results:
        if not ok:
            by_reason[reason] = by_reason.get(reason, 0) + 1
    print(
        f"Endpoints used: vision={endpoints[0].label}:{endpoints[0].model_id}, "
        f"{second_kind}={endpoints[1].label}:{endpoints[1].model_id}"
    )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_generate_dnb_toc_ground_truth.py -v`
Expected: PASS, all tests (existing `TestRunBook` tests unaffected, since `second_kind` defaults to `"vision"` and none of them pass it).

- [ ] **Step 5: Commit**

```bash
git add evaluation/scripts/generate_dnb_toc_ground_truth.py tests/test_generate_dnb_toc_ground_truth.py
git commit -m "feat: dispatch the second endpoint's extraction on second_kind

_run_book/_run_all gain a keyword-only second_kind='vision' parameter
(default preserves every existing call site's behavior unchanged);
_generate now resolves via _resolve_endpoints and threads second_kind
through, so --text-endpoint/--text-config-file (previous task) actually
drives text_extract_toc_entries end to end."
```

---

## Task 7: Kind labeling in `arbitrate_dnb_toc.py`

Implements spec §6: `format_book_report` labels each side by its cached `"kind"` (e.g. `"vision: Qwen/..."` vs. `"text (OCR'd): meta-llama/..."`) so a human arbitrating a mixed-source disagreement knows to suspect OCR quality first.

**Files:**
- Modify: `evaluation/scripts/arbitrate_dnb_toc.py`
- Modify: `tests/test_arbitrate_dnb_toc.py`

- [ ] **Step 1: Write the failing tests**

In `tests/test_arbitrate_dnb_toc.py`, change the imports from:

```python
from chapter_segmentation.segmentation import TocEntry
from evaluation.dnb_toc_vision import write_cached_llm_entries
from evaluation.scripts.arbitrate_dnb_toc import (
    _cached_models_for_book,
    books_needing_arbitration,
    format_book_report,
    reject_book,
)
```

to:

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

In `TestFormatBookReport.test_single_surviving_model_lists_its_entries_with_a_note`, change:

```python
        self.assertIn("Only model-a returned usable output", report)
```

to:

```python
        self.assertIn("Only vision: model-a returned usable output", report)
```

(This is the pre-existing test exercising the no-`kinds`-given default, which now renders every model as `"vision: <model>"`.)

Append these test classes to the end of `tests/test_arbitrate_dnb_toc.py`:

```python
class TestFormatBookReportKindLabels(unittest.TestCase):
    def test_kind_labels_distinguish_vision_from_text(self):
        report = format_book_report(
            "book5", "Some Title", Path("/tmp/book5.pdf"),
            {
                "model-a": [_entry("Einleitung", 9)],
                "model-b": [_entry("Einleitung", 9)],
            },
            {"model-a": "vision", "model-b": "text"},
        )
        self.assertIn("vision: model-a", report)
        self.assertIn("text (OCR'd): model-b", report)

    def test_kind_defaults_to_vision_for_a_model_missing_from_the_kinds_map(self):
        report = format_book_report(
            "book6", "Some Title", Path("/tmp/book6.pdf"),
            {"model-a": [_entry("Einleitung", 9)]},
            {},
        )
        self.assertIn("Only vision: model-a returned usable output", report)


class TestCachedKindsForBook(unittest.TestCase):
    def test_reads_each_models_own_kind(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_directory = Path(tmp)
            write_cached_llm_entries(cache_directory, "book1", "model-a", [_entry("X", 1)], kind="vision")
            write_cached_llm_entries(cache_directory, "book1", "model-b", [_entry("Y", 2)], kind="text")

            result = _cached_kinds_for_book(cache_directory, "book1")

            self.assertEqual(result, {"model-a": "vision", "model-b": "text"})
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_arbitrate_dnb_toc.py -v`
Expected: FAIL -- `ImportError: cannot import name '_cached_kinds_for_book'`, plus the updated `test_single_surviving_model_lists_its_entries_with_a_note` assertion fails against the current unlabeled output.

- [ ] **Step 3: Add kind labeling to `evaluation/scripts/arbitrate_dnb_toc.py`**

Change the `evaluation.dnb_toc_vision` import from:

```python
from evaluation.dnb_toc_vision import load_cached_llm_entries, versioned_cache_dir
```

to:

```python
from evaluation.dnb_toc_vision import load_cached_kind, load_cached_llm_entries, versioned_cache_dir
```

Add this function directly below `_cached_models_for_book`:

```python
def _cached_kinds_for_book(cache_directory: Path, key: str) -> dict[str, str]:
    """Every cached model's extraction "kind" ("vision"/"text") for one
    book key -- same globbing convention as _cached_models_for_book, read
    via load_cached_kind (evaluation/dnb_toc_vision.py). Used only to
    label format_book_report's output (see its own docstring)."""
    result: dict[str, str] = {}
    for path in sorted(versioned_cache_dir(cache_directory).glob(f"{key}.*.json")):
        model = path.name[len(key) + 1: -len(".json")]
        result[model] = load_cached_kind(cache_directory, key, model)
    return result
```

Change `format_book_report`'s signature and body. From:

```python
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
```

to:

```python
def _kind_label(model: str, kind: str) -> str:
    prefix = "vision" if kind == "vision" else "text (OCR'd)"
    return f"{prefix}: {model}"


def format_book_report(
    key: str, title: str, pdf_path: Path, models_to_entries: dict[str, list[TocEntry]],
    kinds: dict[str, str] | None = None,
) -> str:
    """Human-readable diff for one book -- the actual disagreement, ready
    for Claude (or a human) to arbitrate. Handles the normal two-model
    case, the single-surviving-model case (the other model's response
    was empty/malformed), and defensively falls back to a plain per-model
    listing for any other count. Each model name in the rendered report is
    prefixed with its extraction "kind" via _kind_label
    (kinds.get(model, "vision")) -- e.g. "vision: Qwen/..." vs.
    "text (OCR'd): meta-llama/..." -- see design spec
    docs/superpowers/specs/2026-08-20-dnb-toc-vision-text-pairing-design.md
    section 6: a mixed-source disagreement is legible to a human arbitrator
    at a glance instead of looking like plain model disagreement. `kinds`
    defaults every model to "vision" when omitted or when a model is
    missing from it -- the pre-existing, all-vision behavior."""
    kinds = kinds or {}

    def label(model: str) -> str:
        return _kind_label(model, kinds.get(model, "vision"))

    lines = [f"=== {key} -- {title} ===", f"PDF: {pdf_path}"]
    model_names = sorted(models_to_entries)
    if len(model_names) == 1:
        model = model_names[0]
        entries = models_to_entries[model]
        lines.append(f"Only {label(model)} returned usable output ({len(entries)} entries) -- verify directly against the page images:")
        for entry in entries:
            lines.append(_format_entry(entry))
        return "\n".join(lines)
    if len(model_names) != 2:
        lines.append(f"Expected 1 or 2 cached models, found {len(model_names)}: {model_names} -- review each list directly:")
        for model in model_names:
            lines.append(f"  -- {label(model)} ({len(models_to_entries[model])} entries) --")
            for entry in models_to_entries[model]:
                lines.append(_format_entry(entry))
        return "\n".join(lines)
    model_a, model_b = model_names
    entries_a, entries_b = models_to_entries[model_a], models_to_entries[model_b]
    matched, only_a, only_b = diff_toc_entries(entries_a, entries_b)
    rate = len(matched) / max(len(entries_a), len(entries_b))
    lines.append(f"{label(model_a)}: {len(entries_a)} entries, {label(model_b)}: {len(entries_b)} entries -- {len(matched)} matched, rate={rate:.2f}")
    if only_a:
        lines.append(f"  Only in {label(model_a)}:")
        for entry in only_a:
            lines.append(_format_entry(entry))
    if only_b:
        lines.append(f"  Only in {label(model_b)}:")
        for entry in only_b:
            lines.append(_format_entry(entry))
    return "\n".join(lines)
```

Change `_list` to pass `kinds` through. From:

```python
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
```

to:

```python
def _list(cdir: Path, cache_directory: Path) -> int:
    needing = books_needing_arbitration(cdir, cache_directory)
    if not needing:
        print("No books currently need arbitration.")
        return 0
    titles = {manifest_key(book): book.get("title", "") for book in load_manifest_books(_CORPUS_NAME)}
    for key in needing:
        models_to_entries = _cached_models_for_book(cache_directory, key)
        kinds = _cached_kinds_for_book(cache_directory, key)
        print(format_book_report(key, titles.get(key, ""), cdir / f"{key}.pdf", models_to_entries, kinds))
        print()
    return 0
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_arbitrate_dnb_toc.py -v`
Expected: PASS, all tests.

- [ ] **Step 5: Commit**

```bash
git add evaluation/scripts/arbitrate_dnb_toc.py tests/test_arbitrate_dnb_toc.py
git commit -m "feat: label each side of an arbitration report by its extraction kind

format_book_report now prefixes every model name with 'vision: ' or
'text (OCR'd): ' (via the new _kind_label/_cached_kinds_for_book), so a
disagreement between a vision read and an OCR'd-text read is legible at
a glance -- a human arbitrator's first instinct should be to suspect OCR
quality, not assume the two models simply disagree."
```

---

## Task 8: Update the `--help` dump reference and run the full suite

`evaluation/scripts/README.md` keeps a literal `--help` dump for every script -- regenerate `generate_dnb_toc_ground_truth.py`'s entry now that it has two new flags (per that file's own header: "Regenerate an entry by running `uv run python evaluation/scripts/<name>.py --help` ... whenever that script's arguments change").

**Files:**
- Modify: `evaluation/scripts/README.md`

- [ ] **Step 1: Regenerate the `--help` dump**

Run: `uv run python evaluation/scripts/generate_dnb_toc_ground_truth.py --help`

Replace the fenced code block under the `## \`generate_dnb_toc_ground_truth.py\`` heading in `evaluation/scripts/README.md` (currently starting `usage: generate_dnb_toc_ground_truth.py [-h] [--limit LIMIT]` and ending after the `--spot-check N` help line) with the real output of that command -- copy it verbatim into the fence, matching the surrounding wrapped-line style already used for the other entries in this file.

Also update the one-line prose description directly above that fenced block. Change:

```
Generates bulk-tier `dnb-toc-only` ground truth by sending each book's page
images to two independent vision-capable KISSKI models and writing
`.expected.json` only when they agree well enough -- see
`evaluation/README.md`'s "Building dnb-toc-only ground truth".
```

to:

```
Generates bulk-tier `dnb-toc-only` ground truth by sending each book's page
images to two independent vision-capable KISSKI models (or, with
`--text-endpoint`/`--text-config-file`, one vision-capable model paired
with a text-only model fed freshly-OCR'd page text) and writing
`.expected.json` only when they agree well enough -- see
`evaluation/README.md`'s "Building dnb-toc-only ground truth" and design
spec `docs/superpowers/specs/2026-08-20-dnb-toc-vision-text-pairing-design.md`.
```

- [ ] **Step 2: Run the complete test suite**

Run: `uv run pytest`
Expected: PASS, no failures, no new warnings introduced by this plan's changes.

- [ ] **Step 3: Commit**

```bash
git add evaluation/scripts/README.md
git commit -m "docs: regenerate generate_dnb_toc_ground_truth.py's --help dump

Reflects the new --text-endpoint/--text-config-file flags added across
this branch's vision+text-model-pairing work."
```
