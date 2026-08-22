# NuExtract Template-Mode Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `--use-vision`/`--use-text` resolve to a NuExtract-family endpoint (`numind/NuExtract3` today, a finetuned NuExtract2 later) and get correct `TocEntry` extraction from it via NuExtract's own template-mode API, declared explicitly per endpoint in `.endpoints`.

**Architecture:** A new module `nuextract.py`, structurally parallel to `vision.py`/`ocr.py`, provides `nuextract_vision_extract_toc_entries`/`nuextract_text_extract_toc_entries`, reusing `render_pages_to_images`/`ocr_pages_to_rows` for input construction and `parse_json_array`/`_toc_items_to_entries` for response parsing. `inference.py` gains two new endpoint fields (`extraction_api`, `extraction_instructions`) plus a one-off convenience default for the already-running `numind/NuExtract3` endpoint. `generate_ground_truth.py`'s `_run_book` dispatches to the new functions when an endpoint declares `extraction_api == "nuextract"`.

**Tech Stack:** Python 3.12, pytest + pytest-asyncio (`asyncio_mode = "auto"`), `openai` AsyncOpenAI client, `unittest.mock` (`AsyncMock`/`MagicMock`/`patch`) for all test doubles -- no real network calls in any test here.

**Reference:** Design spec `docs/superpowers/specs/2026-08-22-nuextract-template-mode-integration-design.md` -- read it first for the empirical findings and rationale behind every decision below.

---

### Task 1: `inference.py` -- parse `extraction_api`/`extraction_instructions`, both formats

**Files:**
- Modify: `src/dnb_toc_ground_truth/inference.py`
- Test: `tests/test_inference.py`

- [ ] **Step 1: Write the failing tests**

Add this new test class to `tests/test_inference.py`, right after `TestLoadEndpointEntriesPlainText` (before `TestResolveModelEndpoints`):

```python
class TestExtractionApiField(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_json_format_parses_explicit_extraction_api(self):
        path = _write(self.tmp_path, ".endpoints", json.dumps([
            {"url": "https://x.invalid/a", "key": "k", "model": "acme/finetuned-nuextract2",
             "extraction_api": "nuextract"},
        ]))
        entries = load_endpoint_entries(path)
        self.assertEqual(entries[0].extraction_api, "nuextract")

    def test_json_format_defaults_extraction_api_to_empty_for_an_ordinary_model(self):
        path = _write(self.tmp_path, ".endpoints", json.dumps([
            {"url": "https://x.invalid/a", "key": "k", "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct"},
        ]))
        entries = load_endpoint_entries(path)
        self.assertEqual(entries[0].extraction_api, "")

    def test_extraction_instructions_defaults_to_true(self):
        path = _write(self.tmp_path, ".endpoints", json.dumps([
            {"url": "https://x.invalid/a", "key": "k", "model": "model-a", "extraction_api": "nuextract"},
        ]))
        entries = load_endpoint_entries(path)
        self.assertTrue(entries[0].extraction_instructions)

    def test_json_format_parses_explicit_extraction_instructions_false(self):
        path = _write(self.tmp_path, ".endpoints", json.dumps([
            {"url": "https://x.invalid/a", "key": "k", "model": "acme/finetuned-nuextract2",
             "extraction_api": "nuextract", "extraction_instructions": "false"},
        ]))
        entries = load_endpoint_entries(path)
        self.assertFalse(entries[0].extraction_instructions)

    def test_plain_text_format_parses_both_fields(self):
        path = _write(self.tmp_path, ".endpoints", (
            "framework_args\t--model=acme/finetuned-nuextract2\n"
            "extraction_api\tnuextract\n"
            "extraction_instructions\tfalse\n"
            "key\tsecret\n"
            "url\thttps://x.invalid/a\n"
        ))
        entries = load_endpoint_entries(path)
        self.assertEqual(entries[0].extraction_api, "nuextract")
        self.assertFalse(entries[0].extraction_instructions)

    def test_numind_nuextract3_gets_the_convenience_default_when_unset(self):
        path = _write(self.tmp_path, ".endpoints", json.dumps([
            {"url": "https://x.invalid/a", "key": "k", "model": "numind/NuExtract3"},
        ]))
        entries = load_endpoint_entries(path)
        self.assertEqual(entries[0].extraction_api, "nuextract")
        self.assertTrue(entries[0].extraction_instructions)

    def test_other_model_ids_do_not_get_the_convenience_default(self):
        path = _write(self.tmp_path, ".endpoints", json.dumps([
            {"url": "https://x.invalid/a", "key": "k", "model": "numind/NuExtract2-finetuned"},
        ]))
        entries = load_endpoint_entries(path)
        self.assertEqual(entries[0].extraction_api, "")

    def test_explicit_empty_extraction_api_overrides_the_convenience_default(self):
        # An explicit "" for numind/NuExtract3 forces the old free-text
        # path even though the convenience default would otherwise apply
        # -- the design spec commits to this override behavior explicitly.
        path = _write(self.tmp_path, ".endpoints", json.dumps([
            {"url": "https://x.invalid/a", "key": "k", "model": "numind/NuExtract3", "extraction_api": ""},
        ]))
        entries = load_endpoint_entries(path)
        self.assertEqual(entries[0].extraction_api, "")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_inference.py::TestExtractionApiField -v`
Expected: FAIL -- `AttributeError: '_EndpointEntry' object has no attribute 'extraction_api'` (or similar) on every test in the new class.

- [ ] **Step 3: Implement `_EndpointEntry`'s new fields and the resolution helper**

In `src/dnb_toc_ground_truth/inference.py`, modify the `_EndpointEntry` dataclass (currently at lines 58-68):

```python
@dataclass(frozen=True)
class _EndpointEntry:
    """One parsed row from an --endpoints-file, before resolution against
    a requested model id. `status` is the JSON format's raw status string
    ("Running", "Stopped", ...) used only to break a multi-match tie --
    always "" for the plain-text format, which has no equivalent field.
    `extraction_api` ("" or "nuextract") and `extraction_instructions`
    select and configure the NuExtract-family template-mode extraction
    path in nuextract.py -- see _resolve_extraction_fields below and
    design spec docs/superpowers/specs/2026-08-22-nuextract-template-
    mode-integration-design.md."""

    base_url: str
    api_key: str
    model: str
    status: str = ""
    extraction_api: str = ""
    extraction_instructions: bool = True
```

Then add this new constant and helper function right after `_normalize_base_url` (currently at lines 71-72):

```python
_NUEXTRACT_CONVENIENCE_MODEL = "numind/NuExtract3"


def _resolve_extraction_fields(fields: dict, model: str) -> tuple[str, bool]:
    """Resolves (extraction_api, extraction_instructions) for one endpoint
    entry from its raw parsed fields dict -- works identically for the
    JSON-row dict and the plain-text session-block dict, both plain
    str-keyed dicts by the time this is called. An explicitly-PRESENT
    "extraction_api" key always wins, even an explicit "" override --
    only an ABSENT key falls back to the numind/NuExtract3 convenience
    default (the endpoint already running before this field existed
    keeps working without editing .endpoints). Every other model with an
    absent key defaults to "" (today's free-text-prompt path, unchanged).
    "extraction_instructions" defaults to True unless explicitly set to
    "false"/"0"/"no" (case-insensitive) -- only meaningful when
    extraction_api == "nuextract". See design spec 2026-08-22-nuextract-
    template-mode-integration-design.md."""
    if "extraction_api" in fields:
        extraction_api = str(fields["extraction_api"]).strip()
    elif model == _NUEXTRACT_CONVENIENCE_MODEL:
        extraction_api = "nuextract"
    else:
        extraction_api = ""
    extraction_instructions = str(fields.get("extraction_instructions", "")).strip().lower() not in ("false", "0", "no")
    return extraction_api, extraction_instructions
```

Then modify `_parse_plain_text_endpoints` (currently at lines 98-112) to call it:

```python
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
        extraction_api, extraction_instructions = _resolve_extraction_fields(fields, model)
        entries.append(_EndpointEntry(
            base_url=_normalize_base_url(url), api_key=api_key, model=model,
            extraction_api=extraction_api, extraction_instructions=extraction_instructions,
        ))
    return entries
```

And `_parse_json_endpoints` (currently at lines 115-136):

```python
def _parse_json_endpoints(data: list[dict]) -> list[_EndpointEntry]:
    """Officially-supported endpoints-file format: a JSON array of
    objects as pasted from a provider dashboard. Consumes only `url`,
    `key`, and the model id (from `model` if present, else parsed out of
    `framework_args`'s `--model=...` token), plus `status` for
    tie-breaking -- every other field (framework, gpus, job_id, ...) is
    ignored except `extraction_api`/`extraction_instructions`, see
    _resolve_extraction_fields. An entry missing url/key/model is
    skipped."""
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
        extraction_api, extraction_instructions = _resolve_extraction_fields(row, model)
        entries.append(_EndpointEntry(
            base_url=_normalize_base_url(url), api_key=api_key, model=model,
            status=str(row.get("status", "")),
            extraction_api=extraction_api, extraction_instructions=extraction_instructions,
        ))
    return entries
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_inference.py -v`
Expected: PASS -- every test in the file, including the new `TestExtractionApiField` class.

- [ ] **Step 5: Commit**

```bash
git add src/dnb_toc_ground_truth/inference.py tests/test_inference.py
git commit -m "feat: parse extraction_api/extraction_instructions fields in .endpoints"
```

---

### Task 2: `inference.py` -- thread the new fields onto `ModelEndpoint`

**Files:**
- Modify: `src/dnb_toc_ground_truth/inference.py`
- Test: `tests/test_inference.py`

- [ ] **Step 1: Write the failing tests**

Add this new test class to `tests/test_inference.py`, right after `TestResolveModelEndpoints` (before `TestLoadConfig`):

```python
class TestResolveModelEndpointsExtractionFields(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _entries(self, rows):
        path = _write(self.tmp_path, ".endpoints", json.dumps(rows))
        return load_endpoint_entries(path)

    def test_extraction_fields_are_threaded_onto_the_resolved_endpoint(self):
        entries = self._entries([
            {"url": "https://x.invalid/a", "key": "k", "model": "acme/finetuned-nuextract2",
             "extraction_api": "nuextract", "extraction_instructions": "false"},
        ])
        resolved = resolve_model_endpoints(["acme/finetuned-nuextract2"], "text", entries)
        self.assertEqual(resolved[0].extraction_api, "nuextract")
        self.assertFalse(resolved[0].extraction_instructions)

    def test_default_extraction_fields_for_an_ordinary_model(self):
        entries = self._entries([{"url": "https://x.invalid/a", "key": "k", "model": "model-a"}])
        resolved = resolve_model_endpoints(["model-a"], "vision", entries)
        self.assertEqual(resolved[0].extraction_api, "")
        self.assertTrue(resolved[0].extraction_instructions)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_inference.py::TestResolveModelEndpointsExtractionFields -v`
Expected: FAIL -- `AttributeError: 'ModelEndpoint' object has no attribute 'extraction_api'`.

- [ ] **Step 3: Implement**

In `src/dnb_toc_ground_truth/inference.py`, modify the `ModelEndpoint` dataclass (currently at lines 24-33):

```python
@dataclass(frozen=True)
class ModelEndpoint:
    """One ready-to-call (client, model_id) pair, plus which extraction
    path it was requested for ("vision" or "text"). `label` is the
    resolved model id, used only for log/print output.
    `extraction_api`/`extraction_instructions` mirror _EndpointEntry's
    own fields of the same name -- see that dataclass's docstring."""

    label: str
    model_id: str
    kind: str
    client: AsyncOpenAI
    extraction_api: str = ""
    extraction_instructions: bool = True
```

Then modify `resolve_model_endpoints` (currently at lines 158-187) -- only the final `resolved.append(...)` line changes:

```python
        entry = matches[0]
        client = AsyncOpenAI(base_url=entry.base_url, api_key=entry.api_key, timeout=timeout)
        resolved.append(ModelEndpoint(
            label=entry.model, model_id=entry.model, kind=kind, client=client,
            extraction_api=entry.extraction_api, extraction_instructions=entry.extraction_instructions,
        ))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_inference.py -v`
Expected: PASS -- all tests in the file.

- [ ] **Step 5: Commit**

```bash
git add src/dnb_toc_ground_truth/inference.py tests/test_inference.py
git commit -m "feat: thread extraction_api/extraction_instructions onto ModelEndpoint"
```

---

### Task 3: New module `nuextract.py` -- vision-mode template extraction

**Files:**
- Create: `src/dnb_toc_ground_truth/nuextract.py`
- Test: Create `tests/test_nuextract.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_nuextract.py`:

```python
"""Unit tests for nuextract.py -- NuExtract-family template-mode TOC
extraction, see design spec
docs/superpowers/specs/2026-08-22-nuextract-template-mode-integration-design.md.
Both extraction functions are tested with a mocked OpenAI-shaped client,
no real network call; nuextract_text_extract_toc_entries mocks
ocr_pages_to_rows the same way test_ocr.py's text_extract_toc_entries
tests do."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from pypdf import PdfWriter

from dnb_toc_ground_truth.nuextract import _MAX_PAGES, nuextract_vision_extract_toc_entries


def _make_pdf(path: Path, page_count: int) -> Path:
    writer = PdfWriter()
    for _ in range(page_count):
        writer.add_blank_page(width=200, height=200)
    with open(path, "wb") as f:
        writer.write(f)
    return path


def _fake_response(response_text: str):
    message = MagicMock()
    message.content = response_text
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


def _fake_client(response_text: str):
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=_fake_response(response_text))
    return client


def _fake_client_sequence(*response_texts: str):
    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=[_fake_response(t) for t in response_texts])
    return client


_NUEXTRACT_RESPONSE = json.dumps({
    "entries": [
        {"title": "Einleitung", "authors": [], "printed_page_number": "9", "skip": False},
        {"title": "Bibliographie", "authors": [], "printed_page_number": "200", "skip": True},
    ]
})


class TestNuextractVisionExtractTocEntries(unittest.IsolatedAsyncioTestCase):
    async def test_parses_the_entries_wrapper_shape_into_toc_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = _make_pdf(Path(tmp) / "book.pdf", 1)
            client = _fake_client(_NUEXTRACT_RESPONSE)
            entries = await nuextract_vision_extract_toc_entries(pdf_path, "numind/NuExtract3", client)
            self.assertEqual(len(entries), 2)
            self.assertEqual(entries[0].title, "Einleitung")
            self.assertFalse(entries[0].skip)
            self.assertTrue(entries[1].skip)

    async def test_sends_template_and_instructions_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = _make_pdf(Path(tmp) / "book.pdf", 1)
            client = _fake_client(_NUEXTRACT_RESPONSE)
            await nuextract_vision_extract_toc_entries(pdf_path, "numind/NuExtract3", client)
            extra_body = client.chat.completions.create.call_args.kwargs["extra_body"]
            self.assertIn("template", extra_body["chat_template_kwargs"])
            self.assertIn("instructions", extra_body["chat_template_kwargs"])

    async def test_omits_instructions_when_use_instructions_is_false(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = _make_pdf(Path(tmp) / "book.pdf", 1)
            client = _fake_client(_NUEXTRACT_RESPONSE)
            await nuextract_vision_extract_toc_entries(pdf_path, "numind/NuExtract3", client, use_instructions=False)
            extra_body = client.chat.completions.create.call_args.kwargs["extra_body"]
            self.assertNotIn("instructions", extra_body["chat_template_kwargs"])

    async def test_sends_one_image_content_block_per_page(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = _make_pdf(Path(tmp) / "book.pdf", 2)
            client = _fake_client(json.dumps({"entries": []}))
            await nuextract_vision_extract_toc_entries(pdf_path, "numind/NuExtract3", client)
            messages = client.chat.completions.create.call_args.kwargs["messages"]
            content = messages[0]["content"]
            image_blocks = [c for c in content if c["type"] == "image_url"]
            self.assertEqual(len(image_blocks), 2)

    async def test_raises_on_malformed_json_instead_of_swallowing(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = _make_pdf(Path(tmp) / "book.pdf", 1)
            client = _fake_client("not json at all")
            with self.assertRaises(Exception):
                await nuextract_vision_extract_toc_entries(pdf_path, "numind/NuExtract3", client)

    async def test_raises_before_any_network_call_when_page_count_exceeds_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = _make_pdf(Path(tmp) / "book.pdf", _MAX_PAGES + 1)
            client = _fake_client(json.dumps({"entries": []}))
            with self.assertRaises(ValueError):
                await nuextract_vision_extract_toc_entries(pdf_path, "numind/NuExtract3", client)
            client.chat.completions.create.assert_not_called()

    async def test_escalates_max_tokens_and_recovers_from_a_truncated_first_response(self):
        truncated = '{"entries": [{"title": "Einleitung"'
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = _make_pdf(Path(tmp) / "book.pdf", 1)
            client = _fake_client_sequence(truncated, _NUEXTRACT_RESPONSE)
            entries = await nuextract_vision_extract_toc_entries(pdf_path, "numind/NuExtract3", client)
            self.assertEqual(len(entries), 2)
            self.assertEqual(client.chat.completions.create.await_count, 2)
            first_call_kwargs = client.chat.completions.create.call_args_list[0].kwargs
            second_call_kwargs = client.chat.completions.create.call_args_list[1].kwargs
            self.assertEqual(first_call_kwargs["max_tokens"], 4096)
            self.assertEqual(second_call_kwargs["max_tokens"], 8192)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_nuextract.py -v`
Expected: FAIL -- `ModuleNotFoundError: No module named 'dnb_toc_ground_truth.nuextract'`.

- [ ] **Step 3: Implement `nuextract.py` (vision function only for now)**

Create `src/dnb_toc_ground_truth/nuextract.py`:

```python
"""NuExtract-family (numind/NuExtract3 today, a finetuned NuExtract2
later) template-mode TOC extraction -- an alternative to vision.py's/
ocr.py's free-text-prompt extraction path, for endpoints that declare
"extraction_api": "nuextract" in .endpoints (see inference.py). Ad hoc
testing against a live numind/NuExtract3 endpoint found that a free-text
prompt sent as ordinary chat content is effectively ignored by this
model family -- it only follows its own bespoke chat_template_kwargs
convention (an explicit JSON "template" plus an optional prose
"instructions" string), not a prompt embedded in the message content.
Calls client.chat.completions.create directly rather than through
inference.py's OpenAICompatibleLLMClient wrapper, since that wrapper's
generate() has no way to pass extra_body. See design spec
docs/superpowers/specs/2026-08-22-nuextract-template-mode-integration-design.md.

Text-mode quality is noticeably weaker than vision-mode on this
project's OCR'd text (dropped divider labels, one missed page number, an
author name duplicated into its title, in ad hoc testing against a real
book) -- documented here as a known limitation, not engineered around."""

import base64
import json
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from dnb_toc_ground_truth.toc_entry import TocEntry, _toc_items_to_entries, parse_json_array
from dnb_toc_ground_truth.vision import render_pages_to_images

_TEMPLATE = {
    "entries": [
        {
            "title": "verbatim-string",
            "authors": ["string"],
            "printed_page_number": "verbatim-string",
            "skip": "boolean",
        }
    ]
}

# Ported from vision.py's/ocr.py's free-text prompt rules, minus the
# "return ONLY a JSON array..." boilerplate -- template mode already
# governs output shape, so only the extraction RULES need spelling out
# here.
_INSTRUCTIONS = """\
Include EVERY printed line that names a titled section, even lines you \
might not think of as a real chapter: part/section dividers (e.g. "Part \
I", "Teil 1"), front matter (preface, list of contributors), and back \
matter (bibliography, index). Mark `skip` true for these non-chapter \
lines and false for actual chapters -- but always include the entry \
either way, never omit it.

A single chapter's title sometimes spans two printed lines (a short \
main title plus a longer subtitle right below it) with only ONE page \
number for the pair -- that is ONE entry, not two; join both lines into \
a single title string.

An indented, numbered, or lettered sub-point under a heading (e.g. \
"I.", "1.") that carries its OWN page number is its own separate entry \
too, not merged into its parent heading.

If a title is printed with a leading number, letter, or label (e.g. \
"1 ", "2.3 ", "I. ", "a) "), that label is part of the title -- include \
it verbatim as the start of the title string.

`printed_page_number` must be copied exactly as printed, including \
roman numerals (e.g. "vii", not 7). Use null only if no page number is \
visible anywhere on the line."""

_VISION_PROMPT_TEXT = "Extract every table-of-contents entry from this page."

# Same guard/escalation shape as vision.py's _MAX_VISION_PAGES/
# _VISION_MAX_TOKENS/_VISION_MAX_TOKENS_RETRY and ocr.py's equivalents --
# see those constants' own docstrings for the reasoning, unchanged here.
_MAX_PAGES = 20
_MAX_TOKENS = 4096
_MAX_TOKENS_RETRY = 8192


def _extra_body(use_instructions: bool) -> dict:
    kwargs: dict = {"template": json.dumps(_TEMPLATE), "enable_thinking": False}
    if use_instructions:
        kwargs["instructions"] = _INSTRUCTIONS
    return {"chat_template_kwargs": kwargs}


async def nuextract_vision_extract_toc_entries(
    pdf_path: Path, model: str, client: Any, *, use_instructions: bool = True, pdftoppm_bin: str = "pdftoppm",
) -> list[TocEntry]:
    """Template-mode counterpart to vision.py's vision_extract_toc_entries
    -- same page-rendering, page-count cap, max-tokens escalation, and
    raises-on-failure contract, but sends NuExtract's own
    chat_template_kwargs (template + optional instructions) instead of a
    free-text prompt, per this module's own docstring."""
    page_count = len(PdfReader(str(pdf_path)).pages)
    if page_count > _MAX_PAGES:
        raise ValueError(f"{pdf_path}: {page_count} pages exceeds vision-extraction cap of {_MAX_PAGES}")
    images = render_pages_to_images(pdf_path, pdftoppm_bin=pdftoppm_bin)
    content: list[dict] = [{"type": "text", "text": _VISION_PROMPT_TEXT}]
    for image_bytes in images:
        b64 = base64.b64encode(image_bytes).decode("ascii")
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
    messages = [{"role": "user", "content": content}]
    extra_body = _extra_body(use_instructions)

    last_error: Exception | None = None
    for max_tokens in (_MAX_TOKENS, _MAX_TOKENS_RETRY):
        response = await client.chat.completions.create(
            model=model, messages=messages, max_tokens=max_tokens, temperature=0.0, extra_body=extra_body,
        )
        raw = response.choices[0].message.content or ""
        try:
            items = parse_json_array(raw)
            return _toc_items_to_entries(items)
        except Exception as exc:  # noqa: BLE001 -- any parse failure triggers the escalation retry
            last_error = exc
    raise last_error
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_nuextract.py -v`
Expected: PASS -- all tests in `TestNuextractVisionExtractTocEntries`.

- [ ] **Step 5: Commit**

```bash
git add src/dnb_toc_ground_truth/nuextract.py tests/test_nuextract.py
git commit -m "feat: add nuextract_vision_extract_toc_entries template-mode extraction"
```

---

### Task 4: `nuextract.py` -- text-mode template extraction

**Files:**
- Modify: `src/dnb_toc_ground_truth/nuextract.py`
- Test: `tests/test_nuextract.py`

- [ ] **Step 1: Write the failing tests**

In `tests/test_nuextract.py`, change the import line to also bring in the new function:

```python
from dnb_toc_ground_truth.nuextract import (
    _MAX_PAGES, nuextract_text_extract_toc_entries, nuextract_vision_extract_toc_entries,
)
```

Then append this new test class at the end of the file:

```python
class TestNuextractTextExtractTocEntries(unittest.IsolatedAsyncioTestCase):
    async def test_parses_a_clean_response_into_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = _make_pdf(Path(tmp) / "book.pdf", 1)
            client = _fake_client(_NUEXTRACT_RESPONSE)
            with patch("dnb_toc_ground_truth.nuextract.ocr_pages_to_rows", return_value=["Einleitung 9"]):
                entries = await nuextract_text_extract_toc_entries(pdf_path, "acme/finetuned-nuextract2", client)
            self.assertEqual(len(entries), 2)
            self.assertEqual(entries[0].title, "Einleitung")

    async def test_sends_the_ocrd_text_as_plain_string_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = _make_pdf(Path(tmp) / "book.pdf", 1)
            client = _fake_client(json.dumps({"entries": []}))
            with patch("dnb_toc_ground_truth.nuextract.ocr_pages_to_rows", return_value=["Einleitung 9"]):
                await nuextract_text_extract_toc_entries(pdf_path, "acme/finetuned-nuextract2", client)
            messages = client.chat.completions.create.call_args.kwargs["messages"]
            self.assertIn("Einleitung 9", messages[0]["content"])
            self.assertIsInstance(messages[0]["content"], str)

    async def test_omits_instructions_when_use_instructions_is_false(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = _make_pdf(Path(tmp) / "book.pdf", 1)
            client = _fake_client(json.dumps({"entries": []}))
            with patch("dnb_toc_ground_truth.nuextract.ocr_pages_to_rows", return_value=["Einleitung 9"]):
                await nuextract_text_extract_toc_entries(
                    pdf_path, "acme/finetuned-nuextract2", client, use_instructions=False,
                )
            extra_body = client.chat.completions.create.call_args.kwargs["extra_body"]
            self.assertNotIn("instructions", extra_body["chat_template_kwargs"])

    async def test_ocr_failure_propagates_uncaught(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = _make_pdf(Path(tmp) / "book.pdf", 1)
            client = _fake_client(_NUEXTRACT_RESPONSE)
            with patch(
                "dnb_toc_ground_truth.nuextract.ocr_pages_to_rows", side_effect=RuntimeError("ocrmypdf failed"),
            ):
                with self.assertRaises(RuntimeError):
                    await nuextract_text_extract_toc_entries(pdf_path, "acme/finetuned-nuextract2", client)
            client.chat.completions.create.assert_not_called()

    async def test_raises_before_any_ocr_or_network_call_when_page_count_exceeds_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = _make_pdf(Path(tmp) / "book.pdf", _MAX_PAGES + 1)
            client = _fake_client(_NUEXTRACT_RESPONSE)
            with patch("dnb_toc_ground_truth.nuextract.ocr_pages_to_rows") as mock_ocr:
                with self.assertRaises(ValueError):
                    await nuextract_text_extract_toc_entries(pdf_path, "acme/finetuned-nuextract2", client)
            mock_ocr.assert_not_called()
            client.chat.completions.create.assert_not_called()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_nuextract.py -v`
Expected: FAIL -- `ImportError: cannot import name 'nuextract_text_extract_toc_entries'`.

- [ ] **Step 3: Implement**

In `src/dnb_toc_ground_truth/nuextract.py`, add this import alongside the existing `from dnb_toc_ground_truth.vision import render_pages_to_images` line:

```python
from dnb_toc_ground_truth.ocr import ocr_pages_to_rows
```

Then append this function at the end of the file:

```python
async def nuextract_text_extract_toc_entries(
    pdf_path: Path, model: str, client: Any, *, use_instructions: bool = True, pdfalto_bin: str | None = None,
) -> list[TocEntry]:
    """Template-mode counterpart to ocr.py's text_extract_toc_entries --
    same OCR step, page-count cap, max-tokens escalation, and
    raises-on-failure contract, sending the OCR'd text as plain chat
    content alongside NuExtract's chat_template_kwargs instead of a
    free-text prompt."""
    page_count = len(PdfReader(str(pdf_path)).pages)
    if page_count > _MAX_PAGES:
        raise ValueError(f"{pdf_path}: {page_count} pages exceeds text-extraction cap of {_MAX_PAGES}")
    page_texts = ocr_pages_to_rows(pdf_path, pdfalto_bin=pdfalto_bin)
    pages_block = "\n\n".join(f"--- Page {i + 1} ---\n{text}" for i, text in enumerate(page_texts))
    messages = [{"role": "user", "content": pages_block}]
    extra_body = _extra_body(use_instructions)

    last_error: Exception | None = None
    for max_tokens in (_MAX_TOKENS, _MAX_TOKENS_RETRY):
        response = await client.chat.completions.create(
            model=model, messages=messages, max_tokens=max_tokens, temperature=0.0, extra_body=extra_body,
        )
        raw = response.choices[0].message.content or ""
        try:
            items = parse_json_array(raw)
            return _toc_items_to_entries(items)
        except Exception as exc:  # noqa: BLE001 -- any parse failure triggers the escalation retry
            last_error = exc
    raise last_error
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_nuextract.py -v`
Expected: PASS -- every test in the file, both classes.

- [ ] **Step 5: Commit**

```bash
git add src/dnb_toc_ground_truth/nuextract.py tests/test_nuextract.py
git commit -m "feat: add nuextract_text_extract_toc_entries template-mode extraction"
```

---

### Task 5: `generate_ground_truth.py` -- dispatch on `extraction_api`

**Files:**
- Modify: `cli/generate_ground_truth.py`
- Test: `tests/test_generate_ground_truth.py`

- [ ] **Step 1: Write the failing tests**

In `tests/test_generate_ground_truth.py`, modify the `_endpoint` helper (currently at lines 230-231) to accept the two new fields:

```python
def _endpoint(
    model_id: str, client, kind: str = "vision", extraction_api: str = "", extraction_instructions: bool = True,
) -> ModelEndpoint:
    return ModelEndpoint(
        label="test", model_id=model_id, kind=kind, client=client,
        extraction_api=extraction_api, extraction_instructions=extraction_instructions,
    )
```

Then append these three new test methods inside the existing `TestRunBook` class (after `test_a_cache_entry_written_under_a_different_kind_is_not_trusted`, still inside the class body -- keep the same indentation as the other methods in that class):

```python
    async def test_extraction_api_nuextract_dispatches_to_nuextract_vision_function(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp) / "corpus"):
            tmp_path = Path(tmp)
            cache_directory = tmp_path / "cache"
            pdf_path = _make_pdf(tmp_path / "book.pdf")
            client = _fake_vision_client(_VISION_RESPONSE)
            endpoints = [
                _endpoint("model-a", client),
                _endpoint("numind/NuExtract3", client, extraction_api="nuextract"),
            ]
            semaphore = asyncio.Semaphore(1)

            with patch(
                "generate_ground_truth.nuextract_vision_extract_toc_entries",
                new=AsyncMock(return_value=[_entry("Einleitung", 9), _entry("Schluss", 40)]),
            ) as mock_nuextract:
                key, passed, reason = await _run_book(
                    "book13", pdf_path, endpoints, semaphore, cache_directory, 0.90, sleep=AsyncMock(),
                )

            self.assertTrue(passed)
            mock_nuextract.assert_awaited_once()
            self.assertTrue(mock_nuextract.call_args.kwargs["use_instructions"])

    async def test_extraction_api_nuextract_with_text_kind_dispatches_to_nuextract_text_function(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp) / "corpus"):
            tmp_path = Path(tmp)
            cache_directory = tmp_path / "cache"
            pdf_path = _make_pdf(tmp_path / "book.pdf")
            client = _fake_vision_client(_VISION_RESPONSE)
            endpoints = [
                _endpoint("model-a", client),
                _endpoint(
                    "acme/finetuned-nuextract2", client, kind="text",
                    extraction_api="nuextract", extraction_instructions=False,
                ),
            ]
            semaphore = asyncio.Semaphore(1)

            with patch(
                "generate_ground_truth.nuextract_text_extract_toc_entries",
                new=AsyncMock(return_value=[_entry("Einleitung", 9), _entry("Schluss", 40)]),
            ) as mock_nuextract_text:
                key, passed, reason = await _run_book(
                    "book14", pdf_path, endpoints, semaphore, cache_directory, 0.90, sleep=AsyncMock(),
                )

            self.assertTrue(passed)
            mock_nuextract_text.assert_awaited_once()
            self.assertFalse(mock_nuextract_text.call_args.kwargs["use_instructions"])

    async def test_empty_extraction_api_still_dispatches_to_the_ordinary_vision_function(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp) / "corpus"):
            tmp_path = Path(tmp)
            cache_directory = tmp_path / "cache"
            pdf_path = _make_pdf(tmp_path / "book.pdf")
            client = _fake_vision_client(_VISION_RESPONSE)
            endpoints = [_endpoint("model-a", client), _endpoint("model-b", client)]
            semaphore = asyncio.Semaphore(1)

            with patch("generate_ground_truth.nuextract_vision_extract_toc_entries") as mock_nuextract:
                key, passed, reason = await _run_book(
                    "book15", pdf_path, endpoints, semaphore, cache_directory, 0.90, sleep=AsyncMock(),
                )

            self.assertTrue(passed)
            mock_nuextract.assert_not_called()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_generate_ground_truth.py -v -k nuextract`
Expected: FAIL -- `AttributeError: <module 'generate_ground_truth' from ...> does not have the attribute 'nuextract_vision_extract_toc_entries'` (raised by `unittest.mock.patch` on a target that doesn't exist yet, since `cli/generate_ground_truth.py` hasn't imported it).

- [ ] **Step 3: Implement**

In `cli/generate_ground_truth.py`, add this import right after the existing `from dnb_toc_ground_truth.matching import ...` line (currently line 44):

```python
from dnb_toc_ground_truth.nuextract import nuextract_text_extract_toc_entries, nuextract_vision_extract_toc_entries
```

Then modify `_run_book`'s inner `_call` function (currently at lines 222-226):

```python
                async def _call(ep=endpoint):
                    async with semaphore:
                        if ep.extraction_api == "nuextract":
                            fn = nuextract_text_extract_toc_entries if ep.kind == "text" else nuextract_vision_extract_toc_entries
                            return await fn(pdf_path, ep.model_id, ep.client, use_instructions=ep.extraction_instructions)
                        if ep.kind == "text":
                            return await text_extract_toc_entries(pdf_path, ep.model_id, ep.client)
                        return await vision_extract_toc_entries(pdf_path, ep.model_id, ep.client)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_generate_ground_truth.py -v`
Expected: PASS -- the full file, including the three new tests and every pre-existing test (the `_endpoint` helper change is backward-compatible via defaults).

- [ ] **Step 5: Commit**

```bash
git add cli/generate_ground_truth.py tests/test_generate_ground_truth.py
git commit -m "feat: dispatch to nuextract template-mode functions on extraction_api"
```

---

### Task 6: Documentation -- `.endpoints.dist` and `docs/llm-inference-providers.md`

**Files:**
- Modify: `.endpoints.dist`
- Modify: `docs/llm-inference-providers.md`

- [ ] **Step 1: Add a NuExtract3 example entry to `.endpoints.dist`**

The current file (`cat .endpoints.dist`) is a JSON array of two example objects (Pixtral, Qwen3-Omni). Append a third entry to the array, so the full file reads:

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
  },
  {
    "framework": "vLLM",
    "framework_args": "--model=numind/NuExtract3 --trust-remote-code --chat-template-content-format openai --generation-config vllm --max-model-len 131072",
    "host": "10.179.7.236:24148",
    "key": "REPLACE_WITH_REAL_API_KEY",
    "status": "Running",
    "url": "https://your-inference-provider.example/session-c",
    "extraction_api": "nuextract",
    "extraction_instructions": "true"
  }
]
```

(The `extraction_api`/`extraction_instructions` fields are shown explicitly here even though `numind/NuExtract3` gets them by convenience default when absent -- this is the documentation file, so spelling them out is clearer for a reader who hasn't read `inference.py`'s source.)

- [ ] **Step 2: Document the new fields in `docs/llm-inference-providers.md`**

Append this new section to the end of the file (after the existing "Picking a different 'Framework image reference'" section, which currently ends the file):

```markdown

## `.endpoints` fields controlling extraction API style

Two optional fields, read by `inference.py`'s endpoint-file parser
(both the JSON-array and plain-text pasted-session-table formats):

- `extraction_api` -- `"nuextract"` routes this endpoint through
  `nuextract.py`'s template-mode extraction (an explicit JSON schema
  plus optional prose instructions, sent via
  `extra_body={"chat_template_kwargs": {...}}`) instead of the ordinary
  free-text-prompt path every other model uses. Empty/absent (the
  default) means the ordinary path, unchanged.
- `extraction_instructions` -- `"false"`/`"0"`/`"no"` omits the
  `instructions` field from the template-mode request (needed for a
  finetuned NuExtract2-family checkpoint, whose own chat template
  doesn't accept a separate instructions parameter the way NuExtract3's
  does). Defaults to `true`; only meaningful when
  `extraction_api == "nuextract"`.

**`numind/NuExtract3` gets `extraction_api: "nuextract"` and
`extraction_instructions: true` by convenience default when neither
field is set explicitly** -- a one-off carve-out for that exact model id
only, so the endpoint already documented above works without editing
`.endpoints`. Any other NuExtract-family checkpoint (e.g. a finetuned
NuExtract2) must declare both fields explicitly, since a finetuned
checkpoint can be renamed to anything.

A free-text prompt sent as ordinary chat content does NOT work against
`numind/NuExtract3` -- confirmed empirically: the model either ignores
it (falling back to its own trained document-to-markdown default) or
returns a degenerate single-entry response. Template mode with an
explicit JSON schema is the only reliable path. See design spec
`docs/superpowers/specs/2026-08-22-nuextract-template-mode-integration-design.md`
for the full empirical comparison (vision vs. text input, with vs.
without the `instructions` field).
```

- [ ] **Step 3: Commit**

```bash
git add .endpoints.dist docs/llm-inference-providers.md
git commit -m "docs: document extraction_api/extraction_instructions .endpoints fields"
```

---

### Task 7: Full test suite verification

**Files:** None (verification only)

- [ ] **Step 1: Run the entire test suite**

Run: `uv run pytest tests/ -v`
Expected: PASS -- every test in the project, including all tests added in Tasks 1-5. No skips, no failures, no errors.

- [ ] **Step 2: If everything passes, this task needs no commit** (nothing changed). If any pre-existing test broke, fix the regression, re-run, and commit the fix with a message describing exactly what broke and why (e.g. `fix: <specific regression> introduced by <task>`).
