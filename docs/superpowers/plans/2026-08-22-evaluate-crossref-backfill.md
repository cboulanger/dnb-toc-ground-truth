# Evaluate-Crossref Backfill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `cli/evaluate_crossref.py --model <model> --backfill` extracts and caches every Crossref-evaluation-corpus book missing a llm-cache entry for `<model>` before scoring, so a model's raw extraction quality can be measured without running the full two-model bulk-gate pipeline (which deliberately excludes eval-tier books).

**Architecture:** A new library function `backfill_model_cache` in `crossref_evaluation.py` reuses `evaluate_model_corpus`'s own `no_cache` list to know which books need backfilling, dispatches each through `vision_extract_toc_entries` or `nuextract_vision_extract_toc_entries` (based on the resolved `ModelEndpoint`'s `extraction_api`), and writes results via `vision.write_cached_llm_entries`. `cli/evaluate_crossref.py` gains `--backfill`/`--endpoints-file` flags and a small testable `_run_backfill(args, config)` helper (mirroring `generate_ground_truth.py`'s `_resolve_endpoints` pattern) that resolves each `--model` against `--endpoints-file` and calls it.

**Tech Stack:** Python 3.12, pytest + pytest-asyncio, `openai` AsyncOpenAI client, `unittest.mock` for test doubles -- no real network calls in any test.

**Reference:** Design spec `docs/superpowers/specs/2026-08-22-evaluate-crossref-backfill-design.md` -- read it first for the full rationale.

---

### Task 1: `backfill_model_cache` in `crossref_evaluation.py`

**Files:**
- Modify: `src/dnb_toc_ground_truth/crossref_evaluation.py`
- Test: `tests/test_crossref_evaluation.py`

- [ ] **Step 1: Write the failing tests**

Add these imports to the top of `tests/test_crossref_evaluation.py` (extend the existing import block, don't replace it):

```python
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from pypdf import PdfWriter

from dnb_toc_ground_truth.inference import ModelEndpoint
```

(`patch` is already imported in this file from `unittest.mock` -- change that line to `from unittest.mock import AsyncMock, MagicMock, patch` instead of adding a duplicate import line.)

Add this helper function near the top of the file, right after the existing `_write_llm_cache_entry` helper:

```python
def _make_pdf(path: Path) -> Path:
    writer = PdfWriter()
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
```

Change the import line pulling from `dnb_toc_ground_truth.crossref_evaluation` to also bring in the new function:

```python
from dnb_toc_ground_truth.crossref_evaluation import (
    BookMetrics,
    backfill_model_cache,
    discover_cached_models,
    evaluate_book,
    evaluate_corpus,
    evaluate_model_corpus,
    _load_entries,
)
```

Then append this new test class at the end of the file (before the `if __name__ == "__main__":` line):

```python
class TestBackfillModelCache(unittest.TestCase):
    def test_backfills_a_missing_book_and_writes_its_cache_entry(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            manifest_path = corpus.manifest_path()
            manifest_path.write_text(
                json.dumps({"toc_only": True, "books": [{"filename": "9783899718188.pdf", "doi": "10.1/x"}]}),
                encoding="utf-8",
            )
            corpus.pdf_dir().mkdir(parents=True, exist_ok=True)
            _make_pdf(corpus.pdf_path("9783899718188"))
            _write_evaluation_json("9783899718188", [
                {"title": "Introduction", "authors": [], "printed_page_number": "1", "skip": False},
            ])
            client = _fake_client(
                '[{"title": "1. Introduction", "authors": [], "printed_page_number": "1", "skip": false}]'
            )
            endpoint = ModelEndpoint(label="some/model", model_id="some/model", kind="vision", client=client)

            succeeded, failed = asyncio.run(backfill_model_cache("some/model", endpoint, corpus.llm_cache_dir()))

            self.assertEqual(succeeded, ["9783899718188"])
            self.assertEqual(failed, [])
            cached = vision.load_cached_llm_entries(corpus.llm_cache_dir(), "9783899718188", "some/model")
            self.assertIsNotNone(cached)
            self.assertEqual(cached[0].title, "1. Introduction")

    def test_a_book_already_cached_is_not_re_extracted(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            manifest_path = corpus.manifest_path()
            manifest_path.write_text(
                json.dumps({"toc_only": True, "books": [{"filename": "9783899718188.pdf", "doi": "10.1/x"}]}),
                encoding="utf-8",
            )
            _write_evaluation_json("9783899718188", [
                {"title": "Introduction", "authors": [], "printed_page_number": "1", "skip": False},
            ])
            _write_llm_cache_entry("9783899718188", "some/model", [
                TocEntry(title="Introduction", printed_page_number="1", source_page_index=-1, skip=False),
            ])
            client = _fake_client("[]")
            endpoint = ModelEndpoint(label="some/model", model_id="some/model", kind="vision", client=client)

            succeeded, failed = asyncio.run(backfill_model_cache("some/model", endpoint, corpus.llm_cache_dir()))

            self.assertEqual(succeeded, [])
            self.assertEqual(failed, [])
            client.chat.completions.create.assert_not_called()

    def test_a_missing_pdf_is_reported_as_failed_without_raising(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            manifest_path = corpus.manifest_path()
            manifest_path.write_text(
                json.dumps({"toc_only": True, "books": [{"filename": "9783899718188.pdf", "doi": "10.1/x"}]}),
                encoding="utf-8",
            )
            _write_evaluation_json("9783899718188", [
                {"title": "Introduction", "authors": [], "printed_page_number": "1", "skip": False},
            ])
            # No PDF written at corpus.pdf_path(...) -- must be reported, not crash.
            client = _fake_client("[]")
            endpoint = ModelEndpoint(label="some/model", model_id="some/model", kind="vision", client=client)

            succeeded, failed = asyncio.run(backfill_model_cache("some/model", endpoint, corpus.llm_cache_dir()))

            self.assertEqual(succeeded, [])
            self.assertEqual(failed, ["9783899718188"])
            client.chat.completions.create.assert_not_called()

    def test_an_extraction_failure_is_reported_and_does_not_abort_remaining_books(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            manifest_path = corpus.manifest_path()
            manifest_path.write_text(json.dumps({"toc_only": True, "books": [
                {"filename": "9783899718188.pdf", "doi": "10.1/x"},
                {"filename": "9781234567897.pdf", "doi": "10.1/y"},
            ]}), encoding="utf-8")
            corpus.pdf_dir().mkdir(parents=True, exist_ok=True)
            _make_pdf(corpus.pdf_path("9783899718188"))
            _make_pdf(corpus.pdf_path("9781234567897"))
            _write_evaluation_json("9783899718188", [
                {"title": "Introduction", "authors": [], "printed_page_number": "1", "skip": False},
            ])
            _write_evaluation_json("9781234567897", [
                {"title": "Introduction", "authors": [], "printed_page_number": "1", "skip": False},
            ])
            client = MagicMock()
            good_response = _fake_response(
                '[{"title": "1. Introduction", "authors": [], "printed_page_number": "1", "skip": false}]'
            )
            client.chat.completions.create = AsyncMock(side_effect=[RuntimeError("boom"), good_response])
            endpoint = ModelEndpoint(label="some/model", model_id="some/model", kind="vision", client=client)

            succeeded, failed = asyncio.run(backfill_model_cache("some/model", endpoint, corpus.llm_cache_dir()))

            self.assertEqual(len(succeeded), 1)
            self.assertEqual(len(failed), 1)

    def test_routes_through_nuextract_when_endpoint_declares_extraction_api(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            manifest_path = corpus.manifest_path()
            manifest_path.write_text(
                json.dumps({"toc_only": True, "books": [{"filename": "9783899718188.pdf", "doi": "10.1/x"}]}),
                encoding="utf-8",
            )
            corpus.pdf_dir().mkdir(parents=True, exist_ok=True)
            _make_pdf(corpus.pdf_path("9783899718188"))
            _write_evaluation_json("9783899718188", [
                {"title": "Introduction", "authors": [], "printed_page_number": "1", "skip": False},
            ])
            endpoint = ModelEndpoint(
                label="numind/NuExtract3", model_id="numind/NuExtract3", kind="vision", client=MagicMock(),
                extraction_api="nuextract", extraction_instructions=False,
            )

            with patch(
                "dnb_toc_ground_truth.crossref_evaluation.nuextract_vision_extract_toc_entries",
                new=AsyncMock(return_value=[
                    TocEntry(title="1. Introduction", printed_page_number="1", source_page_index=-1),
                ]),
            ) as mock_nuextract:
                succeeded, failed = asyncio.run(
                    backfill_model_cache("numind/NuExtract3", endpoint, corpus.llm_cache_dir())
                )

            self.assertEqual(succeeded, ["9783899718188"])
            mock_nuextract.assert_awaited_once()
            self.assertFalse(mock_nuextract.call_args.kwargs["use_instructions"])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_crossref_evaluation.py::TestBackfillModelCache -v`
Expected: FAIL -- `ImportError: cannot import name 'backfill_model_cache' from 'dnb_toc_ground_truth.crossref_evaluation'`.

- [ ] **Step 3: Implement `backfill_model_cache`**

In `src/dnb_toc_ground_truth/crossref_evaluation.py`, add these imports (extend the existing `from dnb_toc_ground_truth import corpus, crossref, matching, vision` line's neighbors, don't replace it):

```python
from dnb_toc_ground_truth.inference import ModelEndpoint
from dnb_toc_ground_truth.nuextract import nuextract_vision_extract_toc_entries
from dnb_toc_ground_truth.vision import vision_extract_toc_entries
```

Then append this function at the end of the file, right after `evaluate_model_corpus`:

```python
async def backfill_model_cache(
    model: str, endpoint: ModelEndpoint, cache_directory: Path,
) -> tuple[list[str], list[str]]:
    """For every Crossref-evaluation-corpus book missing a llm-cache
    entry for `model` (reuses evaluate_model_corpus's own no_cache list
    -- the exact same book-selection logic scoring already trusts),
    extracts once via `endpoint` and writes the cache entry. Returns
    (succeeded_keys, failed_keys) -- a failure (missing PDF, network
    error, unparseable response, empty result) is printed and skipped,
    not retried; this is a manual one-off utility run, not
    generate_ground_truth.py's unattended batch job. `model` and
    `endpoint.model_id` are expected to match (the caller resolved
    `endpoint` FOR this `model`); kept as two separate parameters rather
    than reading `endpoint.model_id` directly so the cache is written
    under exactly the model id the caller/CLI asked to backfill, not
    whatever string the endpoints file happened to resolve it to."""
    _, missing_keys = evaluate_model_corpus(model)
    succeeded, failed = [], []
    for key in missing_keys:
        pdf_path = corpus.pdf_path(key)
        if not pdf_path.exists():
            print(f"[backfill] {key}: skipped, no PDF at {pdf_path}")
            failed.append(key)
            continue
        try:
            if endpoint.extraction_api == "nuextract":
                entries = await nuextract_vision_extract_toc_entries(
                    pdf_path, endpoint.model_id, endpoint.client,
                    use_instructions=endpoint.extraction_instructions,
                )
            else:
                entries = await vision_extract_toc_entries(pdf_path, endpoint.model_id, endpoint.client)
        except Exception as exc:  # noqa: BLE001 -- one book's failure must not abort the whole backfill
            print(f"[backfill] {key}: failed -- {type(exc).__name__}: {exc}")
            failed.append(key)
            continue
        if entries:
            vision.write_cached_llm_entries(cache_directory, key, model, entries, kind="vision")
            succeeded.append(key)
        else:
            print(f"[backfill] {key}: extraction returned no entries, not cached")
            failed.append(key)
    return succeeded, failed
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_crossref_evaluation.py -v`
Expected: PASS -- every test in the file, including all 5 new `TestBackfillModelCache` tests.

- [ ] **Step 5: Commit**

```bash
git add src/dnb_toc_ground_truth/crossref_evaluation.py tests/test_crossref_evaluation.py
git commit -m "feat: add backfill_model_cache for populating missing eval-corpus llm-cache entries"
```

---

### Task 2: CLI wiring in `cli/evaluate_crossref.py`

**Files:**
- Modify: `cli/evaluate_crossref.py`
- Create: `tests/test_evaluate_crossref.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_evaluate_crossref.py`:

```python
"""Unit tests for cli/evaluate_crossref.py's --backfill CLI wiring. The
actual backfill logic (extraction, cache writing) is tested at the
library level in tests/test_crossref_evaluation.py -- this only tests
that _run_backfill resolves --model against --endpoints-file correctly
and fails loudly when it can't, mirroring
tests/test_generate_ground_truth.py's TestResolveEndpoints convention of
testing the internal helper directly rather than main() via sys.argv."""

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cli"))

from evaluate_crossref import _run_backfill


def _args(**overrides) -> argparse.Namespace:
    defaults = dict(model=None, endpoints_file=None)
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestRunBackfill(unittest.TestCase):
    def test_raises_without_a_matching_endpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            endpoints_path = Path(tmp) / ".endpoints"
            endpoints_path.write_text(json.dumps([
                {"url": "https://x.invalid/a", "key": "k", "model": "some-other-model"},
            ]), encoding="utf-8")
            args = _args(model=["nonexistent/model"], endpoints_file=endpoints_path)
            with self.assertRaises(ValueError):
                _run_backfill(args, {})

    def test_requires_at_least_one_model(self):
        args = _args(model=None)
        with self.assertRaises(SystemExit):
            _run_backfill(args, {})


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_evaluate_crossref.py -v`
Expected: FAIL -- `ImportError: cannot import name '_run_backfill' from 'evaluate_crossref'`.

- [ ] **Step 3: Implement**

In `cli/evaluate_crossref.py`, change the import block at the top of the file from:

```python
import argparse
from pathlib import Path

from dnb_toc_ground_truth import corpus, inference
from dnb_toc_ground_truth.crossref_evaluation import (
    BookMetrics,
    discover_cached_models,
    evaluate_corpus,
    evaluate_model_corpus,
)
```

to:

```python
import argparse
import asyncio
from pathlib import Path

from dnb_toc_ground_truth import corpus, inference
from dnb_toc_ground_truth.crossref_evaluation import (
    BookMetrics,
    backfill_model_cache,
    discover_cached_models,
    evaluate_corpus,
    evaluate_model_corpus,
)
```

Add this new function right before `def main() -> int:`:

```python
def _run_backfill(args: argparse.Namespace, config: dict) -> None:
    """Resolves each --model against --endpoints-file and backfills its
    missing Crossref-evaluation-corpus llm-cache entries -- see design
    spec docs/superpowers/specs/2026-08-22-evaluate-crossref-backfill-
    design.md. Raises SystemExit if --backfill was given with no --model
    at all, and lets resolve_model_endpoints' own ValueError (naming the
    unmatched model id) propagate uncaught -- same fail-loud-and-early
    convention as generate_ground_truth.py's own endpoint resolution."""
    if not args.model:
        raise SystemExit("--backfill requires at least one --model")
    endpoints_file = Path(args.endpoints_file or config.get("endpoints_file", inference.DEFAULT_ENDPOINTS_FILENAME))
    entries = inference.load_endpoint_entries(endpoints_file)
    for model in args.model:
        endpoint = inference.resolve_model_endpoints([model], "vision", entries)[0]
        succeeded, failed = asyncio.run(backfill_model_cache(model, endpoint, corpus.llm_cache_dir()))
        print(f"[backfill] {model}: {len(succeeded)} written, {len(failed)} failed")
```

In `main()`, add these two new `parser.add_argument` calls right after the existing `--all-models` argument (currently the block reads `--full`, `--min-f1`, `--model`, `--all-models`, `--corpus`, `--config-file` in that order -- insert the two new ones between `--all-models` and `--corpus`):

```python
    parser.add_argument(
        "--backfill", action="store_true",
        help="Before scoring, extract and cache any --model book missing a llm-cache entry, "
             "resolved against --endpoints-file (vision-input models only)",
    )
    parser.add_argument(
        "--endpoints-file", type=Path, default=None,
        help=f"Path to the endpoints file, only used with --backfill (default: "
             f"{inference.DEFAULT_ENDPOINTS_FILENAME}, or config file's \"endpoints_file\")",
    )
```

Then, in `main()`, right after the existing `corpus.set_corpus(args.corpus or config.get("corpus") or corpus.DEFAULT_CORPUS_NAME)` line and before `results, no_coverage = evaluate_corpus()`, add:

```python
    if args.backfill:
        _run_backfill(args, config)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_evaluate_crossref.py -v`
Expected: PASS -- both tests.

Then run: `uv run pytest tests/ -q`
Expected: PASS -- the full suite (327 pre-existing + 5 from Task 1 + 2 from Task 2 = 334), no regressions.

- [ ] **Step 5: Commit**

```bash
git add cli/evaluate_crossref.py tests/test_evaluate_crossref.py
git commit -m "feat: add --backfill/--endpoints-file flags to evaluate_crossref.py"
```

---

### Task 3: Manual smoke check against the real corpus (not a unit test)

**Files:** None (verification only, no code change)

This task is deliberately NOT automated -- it exercises a real network endpoint and is exactly the operation this whole feature exists to support, so it's the actual acceptance check for the feature, not something to mock away.

- [ ] **Step 1: Confirm `numind/NuExtract3` is live in `.endpoints`**

Run: `cat .endpoints | python3 -m json.tool | grep -A2 '"model": "numind/NuExtract3"'` (or open `.endpoints` directly). If the entry is missing or the MPCDF session has expired, STOP and report back rather than guessing at a replacement -- this step requires the user's live MPCDF session.

- [ ] **Step 2: Run the backfill against the real 54-book evaluation corpus**

Run: `uv run python cli/evaluate_crossref.py --model numind/NuExtract3 --backfill --full`

Expected: prints `[backfill] numind/NuExtract3: N written, M failed` (N should be close to 54, allowing for any books with a genuinely missing PDF), followed by the normal per-book and aggregate precision/recall/F1 output for `numind/NuExtract3` against the Crossref evaluation corpus.

- [ ] **Step 3: Report the results**

Report the aggregate mean precision/recall/F1 for `numind/NuExtract3` and how many books succeeded/failed, so the user can compare it against the other models' already-cached scores (`Qwen/Qwen3-Omni-30B-A3B-Instruct`, `mistralai/Mistral-Small-3.2-24B-Instruct-2506`, etc. -- run `uv run python cli/evaluate_crossref.py --all-models --full` to see all of them side by side).
