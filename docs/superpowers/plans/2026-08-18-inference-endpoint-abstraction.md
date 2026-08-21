# Pluggable inference endpoints Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `generate_dnb_toc_ground_truth.py` and `refresh_llm_cache.py` target any OpenAI-compatible inference endpoint (KISSKI, an MPCDF LLM Inference Service session, or any other) via a repeatable `--endpoint ALIAS` CLI flag, per `docs/superpowers/specs/2026-08-18-inference-endpoint-abstraction-design.md`.

**Architecture:** One new module, `evaluation/inference_endpoints.py`, holds a `ModelEndpoint` dataclass (`label`, `model_id`, `client`) and `resolve_endpoint_from_env(alias)`, which reads `<ALIAS>_BASE_URL`/`<ALIAS>_API_KEY`/`<ALIAS>_MODEL` from the environment. Both scripts' KISSKI-specific discovery/selection code (`evaluation/kisski.py`) is untouched and remains the default when `--endpoint` is omitted; `--endpoint` bypasses it entirely.

**Tech Stack:** Python 3.12, `openai` (`AsyncOpenAI`), `pytest`/`unittest`.

---

### Task 1: `evaluation/inference_endpoints.py` -- `ModelEndpoint` + `resolve_endpoint_from_env`

**Files:**
- Create: `evaluation/inference_endpoints.py`
- Test: `tests/test_inference_endpoints.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_inference_endpoints.py`:

```python
"""Unit tests for evaluation/inference_endpoints.py."""

import os
import unittest
from unittest.mock import patch

from evaluation.inference_endpoints import ModelEndpoint, resolve_endpoint_from_env


class TestResolveEndpointFromEnv(unittest.TestCase):
    def test_builds_endpoint_from_all_three_vars(self):
        env = {
            "MPCDF_A_BASE_URL": "https://example.invalid/v1",
            "MPCDF_A_API_KEY": "secret-key",
            "MPCDF_A_MODEL": "Qwen/Qwen3-VL-30B-A3B-Instruct",
        }
        with patch.dict(os.environ, env, clear=False):
            endpoint = resolve_endpoint_from_env("MPCDF_A")

        self.assertIsInstance(endpoint, ModelEndpoint)
        self.assertEqual(endpoint.label, "MPCDF_A")
        self.assertEqual(endpoint.model_id, "Qwen/Qwen3-VL-30B-A3B-Instruct")
        self.assertIn("https://example.invalid/v1", str(endpoint.client.base_url))
        self.assertEqual(endpoint.client.api_key, "secret-key")

    def test_missing_base_url_raises_naming_that_var(self):
        env = {"MPCDF_B_API_KEY": "k", "MPCDF_B_MODEL": "m"}
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaises(ValueError) as ctx:
                resolve_endpoint_from_env("MPCDF_B")
        self.assertIn("MPCDF_B_BASE_URL", str(ctx.exception))

    def test_missing_api_key_raises_naming_that_var(self):
        env = {"MPCDF_C_BASE_URL": "https://example.invalid/v1", "MPCDF_C_MODEL": "m"}
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaises(ValueError) as ctx:
                resolve_endpoint_from_env("MPCDF_C")
        self.assertIn("MPCDF_C_API_KEY", str(ctx.exception))

    def test_missing_model_raises_naming_that_var(self):
        env = {"MPCDF_D_BASE_URL": "https://example.invalid/v1", "MPCDF_D_API_KEY": "k"}
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaises(ValueError) as ctx:
                resolve_endpoint_from_env("MPCDF_D")
        self.assertIn("MPCDF_D_MODEL", str(ctx.exception))

    def test_all_missing_names_all_three(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError) as ctx:
                resolve_endpoint_from_env("MPCDF_E")
        message = str(ctx.exception)
        self.assertIn("MPCDF_E_BASE_URL", message)
        self.assertIn("MPCDF_E_API_KEY", message)
        self.assertIn("MPCDF_E_MODEL", message)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_inference_endpoints.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'evaluation.inference_endpoints'`

- [ ] **Step 3: Write the implementation**

Create `evaluation/inference_endpoints.py`:

```python
"""Provider-agnostic OpenAI-compatible inference endpoints -- lets
generate_dnb_toc_ground_truth.py and refresh_llm_cache.py target KISSKI,
an MPCDF LLM Inference Service session (https://llm.mpcdf.mpg.de), or any
other OpenAI-compatible chat completions endpoint, selected per invocation
via a repeatable --endpoint ALIAS CLI flag. See design spec
docs/superpowers/specs/2026-08-18-inference-endpoint-abstraction-design.md.

Deliberately NOT a Provider class hierarchy: KISSKI's real discovery/
demand-aware model selection (evaluation/kisski.py) has no equivalent for
a self-deployed MPCDF session -- no shared pool, no discovery endpoint,
you pick and deploy exactly one model yourself. This module stays a flat
(base_url, api_key, model_id) resolver; evaluation/kisski.py is untouched
and remains the default path when no --endpoint is given.
"""

import os
from dataclasses import dataclass

from openai import AsyncOpenAI

DEFAULT_TIMEOUT = 90.0


@dataclass(frozen=True)
class ModelEndpoint:
    """One ready-to-call (client, model_id) pair. `label` is the alias
    (or "kisski" for the auto-selected default path) -- used only for
    log/print output, never to branch behavior."""

    label: str
    model_id: str
    client: AsyncOpenAI


def resolve_endpoint_from_env(alias: str, *, timeout: float = DEFAULT_TIMEOUT) -> ModelEndpoint:
    """Builds a ModelEndpoint from `<alias>_BASE_URL`, `<alias>_API_KEY`,
    `<alias>_MODEL` environment variables. Raises ValueError naming
    exactly which variable(s) are missing -- this is meant to be diagnosed
    by a human setting up an MPCDF (or other) session, not by reading a
    bare KeyError traceback."""
    var_names = (f"{alias}_BASE_URL", f"{alias}_API_KEY", f"{alias}_MODEL")
    missing = [var for var in var_names if var not in os.environ]
    if missing:
        raise ValueError(
            f"--endpoint {alias} requires environment variables "
            f"{alias}_BASE_URL, {alias}_API_KEY, {alias}_MODEL to be set -- "
            f"missing: {', '.join(missing)}"
        )
    base_url, api_key, model_id = (os.environ[var] for var in var_names)
    client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
    return ModelEndpoint(label=alias, model_id=model_id, client=client)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_inference_endpoints.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add evaluation/inference_endpoints.py tests/test_inference_endpoints.py
git commit -m "feat(evaluation): add provider-agnostic ModelEndpoint/resolve_endpoint_from_env"
```

---

### Task 2: `generate_dnb_toc_ground_truth.py` -- `_run_book`/`_run_all` take `ModelEndpoint` tuples

**Files:**
- Modify: `evaluation/scripts/generate_dnb_toc_ground_truth.py:185-233,318-326`
- Modify: `tests/test_generate_dnb_toc_ground_truth.py:1-29,215-362`

- [ ] **Step 1: Update the existing tests to the new signature (still red for now)**

In `tests/test_generate_dnb_toc_ground_truth.py`, change the import block (lines 17-29) to add `ModelEndpoint`:

```python
from chapter_segmentation.segmentation import TocEntry
from evaluation.dnb_toc_vision import load_cached_llm_entries, write_cached_llm_entries
from evaluation.inference_endpoints import ModelEndpoint
from evaluation.kisski import KisskiModel
from evaluation.scripts.generate_dnb_toc_ground_truth import (
    _binding_rate_limit_window,
    _call_with_retry,
    _is_stale_bulk_gate_entry,
    _retry_after_seconds,
    _run_book,
    _run_book_entries,
    _select_best_models,
    _still_needs_a_decision,
)
```

Replace the whole block from `def _fake_vision_client` (line 215) through the end of `TestRunBook` (line 362) with:

```python
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


def _endpoint(model_id: str, client) -> ModelEndpoint:
    return ModelEndpoint(label="test", model_id=model_id, client=client)


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
            endpoints = (_endpoint("model-a", client), _endpoint("model-b", client))
            semaphore = asyncio.Semaphore(1)

            key, passed, reason = await _run_book(
                "book1", pdf_path, endpoints, semaphore, corpus_directory, cache_directory,
                sleep=AsyncMock(),
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
            write_cached_llm_entries(cache_directory, "book2", "model-a", entries)
            write_cached_llm_entries(cache_directory, "book2", "model-b", entries)
            client = _fake_vision_client(_VISION_RESPONSE)
            endpoints = (_endpoint("model-a", client), _endpoint("model-b", client))
            semaphore = asyncio.Semaphore(1)

            key, passed, reason = await _run_book(
                "book2", pdf_path, endpoints, semaphore, corpus_directory, cache_directory,
                sleep=AsyncMock(),
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
            endpoints = (_endpoint("model-a", client), _endpoint("model-b", client))
            semaphore = asyncio.Semaphore(1)

            key, passed, reason = await _run_book(
                "book3", bad_pdf, endpoints, semaphore, corpus_directory, cache_directory,
                sleep=AsyncMock(),
            )

            self.assertFalse(passed)
            self.assertTrue(reason.startswith("error:"))

    async def test_one_model_failing_preserves_the_others_cache_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            corpus_directory = tmp_path / "corpus"
            cache_directory = tmp_path / "cache"
            corpus_directory.mkdir()
            pdf_path = _make_pdf(tmp_path / "book.pdf")
            client = MagicMock()
            good_message = MagicMock()
            good_message.content = _VISION_RESPONSE
            good_choice = MagicMock()
            good_choice.message = good_message
            good_response = MagicMock()
            good_response.choices = [good_choice]
            client.chat.completions.create = AsyncMock(
                side_effect=[good_response, RuntimeError("boom"), RuntimeError("boom"), RuntimeError("boom")]
            )
            endpoints = (_endpoint("model-a", client), _endpoint("model-b", client))
            semaphore = asyncio.Semaphore(1)

            key, passed, reason = await _run_book(
                "book4", pdf_path, endpoints, semaphore, corpus_directory, cache_directory,
                sleep=AsyncMock(),
            )

            self.assertFalse(passed)
            self.assertTrue(reason.startswith("error:"))
            self.assertIsNotNone(load_cached_llm_entries(cache_directory, "book4", "model-a"))
            self.assertIsNone(load_cached_llm_entries(cache_directory, "book4", "model-b"))

    async def test_semaphore_is_released_during_backoff_sleep(self):
        # Regression test for a real 2026-08-17 batch stall: the semaphore
        # used to wrap the whole retry sequence, so a backoff sleep held a
        # concurrency slot hostage -- if enough books hit RateLimitError
        # around the same time, every slot ended up asleep simultaneously
        # and the batch stalled with zero throughput even though nothing
        # had crashed. It must be released before each sleep so other
        # books can make progress while this one backs off.
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            corpus_directory = tmp_path / "corpus"
            cache_directory = tmp_path / "cache"
            corpus_directory.mkdir()
            pdf_path = _make_pdf(tmp_path / "book.pdf")
            client = MagicMock()
            client.chat.completions.create = AsyncMock(side_effect=RuntimeError("boom"))
            endpoints = (_endpoint("model-a", client), _endpoint("model-b", client))
            semaphore = asyncio.Semaphore(1)
            observed_lock_state_during_sleep = []

            async def spying_sleep(_delay):
                observed_lock_state_during_sleep.append(semaphore.locked())

            await _run_book(
                "book5", pdf_path, endpoints, semaphore, corpus_directory, cache_directory,
                sleep=spying_sleep,
            )

            self.assertTrue(observed_lock_state_during_sleep, "sleep (backoff) was never invoked")
            self.assertTrue(
                all(not locked for locked in observed_lock_state_during_sleep),
                "semaphore was still held during a backoff sleep",
            )

    async def test_two_independent_endpoints_each_get_their_own_client_called(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            corpus_directory = tmp_path / "corpus"
            cache_directory = tmp_path / "cache"
            corpus_directory.mkdir()
            pdf_path = _make_pdf(tmp_path / "book.pdf")
            client_a = _fake_vision_client(_VISION_RESPONSE)
            client_b = _fake_vision_client(_VISION_RESPONSE)
            endpoints = (_endpoint("model-a", client_a), _endpoint("model-b", client_b))
            semaphore = asyncio.Semaphore(1)

            key, passed, reason = await _run_book(
                "book6", pdf_path, endpoints, semaphore, corpus_directory, cache_directory,
                sleep=AsyncMock(),
            )

            self.assertTrue(passed)
            client_a.chat.completions.create.assert_awaited_once()
            client_b.chat.completions.create.assert_awaited_once()
            self.assertEqual(client_a.chat.completions.create.await_args.kwargs["model"], "model-a")
            self.assertEqual(client_b.chat.completions.create.await_args.kwargs["model"], "model-b")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_generate_dnb_toc_ground_truth.py -v`
Expected: FAIL -- `ModuleNotFoundError: No module named 'evaluation.inference_endpoints'` isn't the error here (Task 1 already created it); instead expect `TypeError: _run_book() takes ... positional arguments but ... were given` (old signature still has separate `models`/`client` params) or an `AttributeError`/mismatched-call error, since `_run_book` hasn't been updated yet.

- [ ] **Step 3: Update `_run_book`/`_run_all` in the implementation**

In `evaluation/scripts/generate_dnb_toc_ground_truth.py`, add to the imports (near the top, alongside the existing `from evaluation.kisski import ...` line):

```python
from evaluation.inference_endpoints import ModelEndpoint, resolve_endpoint_from_env
```

Replace `_run_book` (lines 185-232) with:

```python
async def _run_book(
    key: str, pdf_path: Path, endpoints: tuple[ModelEndpoint, ModelEndpoint], semaphore: asyncio.Semaphore,
    corpus_directory: Path, cache_directory: Path, sleep=asyncio.sleep,
) -> tuple[str, bool, str]:
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

    `semaphore` is acquired only around each individual API call attempt
    (inside the closure passed to _call_with_retry), NOT around the whole
    retry sequence -- found the hard way (2026-08-17 batch run) that
    holding it for the full sequence lets a backoff sleep occupy a
    concurrency slot for up to minutes, and if enough books hit
    RateLimitError around the same time, every slot ends up asleep at once
    and the entire batch stalls with zero throughput even though nothing
    actually crashed. Releasing it between attempts lets other books make
    progress while one book backs off."""
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
    except Exception as exc:  # noqa: BLE001 -- must never let one book crash the whole batch
        print(f"[error] {key}: {exc}{_rate_limit_headers_suffix(exc)}")
        return key, False, f"error: {type(exc).__name__}"
```

Replace `_run_all` (lines 318-326) with:

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

`_generate` still calls `_run_all(candidates, models, client, args.concurrency, cdir, ...)` with the old shape at this point -- that's fixed in Task 3, so `_generate`/`main()` will not run correctly until Task 3 lands. This task is scoped to `_run_book`/`_run_all` and their tests only; don't run `main()` end-to-end yet.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_generate_dnb_toc_ground_truth.py -v`
Expected: PASS (all `TestRunBook` tests, including the new `test_two_independent_endpoints_each_get_their_own_client_called`). Other test classes in this file (`TestCallWithRetry`, `TestStillNeedsADecision`, etc.) are unaffected and should still pass.

- [ ] **Step 5: Commit**

```bash
git add evaluation/scripts/generate_dnb_toc_ground_truth.py tests/test_generate_dnb_toc_ground_truth.py
git commit -m "refactor(dnb-toc-gate): _run_book/_run_all take per-model ModelEndpoints"
```

---

### Task 3: `generate_dnb_toc_ground_truth.py` -- `_resolve_vision_endpoints` + `--endpoint` CLI

**Files:**
- Modify: `evaluation/scripts/generate_dnb_toc_ground_truth.py:314-316,374-429`
- Modify: `tests/test_generate_dnb_toc_ground_truth.py` (imports + new test class)

- [ ] **Step 1: Write the failing tests**

In `tests/test_generate_dnb_toc_ground_truth.py`, add `os` and `patch` to imports (top of file):

```python
import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
```

Add `_resolve_vision_endpoints` to the import from the module under test:

```python
from evaluation.scripts.generate_dnb_toc_ground_truth import (
    _binding_rate_limit_window,
    _call_with_retry,
    _is_stale_bulk_gate_entry,
    _resolve_vision_endpoints,
    _retry_after_seconds,
    _run_book,
    _run_book_entries,
    _select_best_models,
    _still_needs_a_decision,
)
```

Append a new test class at the end of the file:

```python
class TestResolveVisionEndpoints(unittest.TestCase):
    def test_no_aliases_falls_back_to_kisski_auto_select(self):
        env = {"KISSKI_API_KEY": "kisski-key"}
        with patch.dict(os.environ, env, clear=False), patch(
            "evaluation.scripts.generate_dnb_toc_ground_truth._pick_models",
            return_value=["model-x", "model-y"],
        ) as mock_pick:
            endpoints = _resolve_vision_endpoints(None)

        mock_pick.assert_called_once()
        self.assertEqual(endpoints[0].label, "kisski")
        self.assertEqual(endpoints[0].model_id, "model-x")
        self.assertEqual(endpoints[1].label, "kisski")
        self.assertEqual(endpoints[1].model_id, "model-y")
        self.assertIs(endpoints[0].client, endpoints[1].client)

    def test_exactly_two_aliases_resolved_independently(self):
        env = {
            "MPCDF_A_BASE_URL": "https://a.invalid/v1", "MPCDF_A_API_KEY": "ka", "MPCDF_A_MODEL": "model-a",
            "MPCDF_B_BASE_URL": "https://b.invalid/v1", "MPCDF_B_API_KEY": "kb", "MPCDF_B_MODEL": "model-b",
        }
        with patch.dict(os.environ, env, clear=False):
            endpoints = _resolve_vision_endpoints(["MPCDF_A", "MPCDF_B"])

        self.assertEqual(endpoints[0].label, "MPCDF_A")
        self.assertEqual(endpoints[0].model_id, "model-a")
        self.assertEqual(endpoints[1].label, "MPCDF_B")
        self.assertEqual(endpoints[1].model_id, "model-b")
        self.assertIsNot(endpoints[0].client, endpoints[1].client)

    def test_one_alias_is_a_user_error(self):
        with self.assertRaises(SystemExit):
            _resolve_vision_endpoints(["MPCDF_A"])

    def test_three_aliases_is_a_user_error(self):
        with self.assertRaises(SystemExit):
            _resolve_vision_endpoints(["MPCDF_A", "MPCDF_B", "MPCDF_C"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_generate_dnb_toc_ground_truth.py::TestResolveVisionEndpoints -v`
Expected: FAIL with `ImportError: cannot import name '_resolve_vision_endpoints'`

- [ ] **Step 3: Implement `_resolve_vision_endpoints` and wire the CLI**

In `evaluation/scripts/generate_dnb_toc_ground_truth.py`, replace `_pick_models` (lines 314-316) with `_pick_models` unchanged plus a new function right after it:

```python
def _pick_models(base_url: str, api_key: str) -> list[str]:
    return _select_best_models(fetch_kisski_models(base_url, api_key))


def _resolve_vision_endpoints(endpoint_aliases: Optional[list[str]]) -> tuple[ModelEndpoint, ModelEndpoint]:
    """Resolves the two ModelEndpoints the two-independent-vision-model
    gate calls. No --endpoint given -> today's default: KISSKI discovery
    picks two distinct vision-capable models, sharing one client (both
    live behind the same KISSKI base URL, unchanged from before this
    endpoint abstraction existed). Exactly two --endpoint aliases -> each
    resolved independently via resolve_endpoint_from_env, letting the two
    reads come from different endpoints/providers (e.g. two MPCDF
    sessions, or one MPCDF session + one manually-picked KISSKI model).
    Any other alias count is a user error -- the gate's independence
    guarantee requires exactly two reads."""
    if not endpoint_aliases:
        api_key = os.environ["KISSKI_API_KEY"]
        model_ids = tuple(_pick_models(DEFAULT_KISSKI_BASE_URL, api_key))
        # Explicit per-request timeout -- the openai SDK's own default
        # (600s read timeout) let one slow/hung KISSKI response occupy a
        # concurrency slot for up to 10 minutes PER ATTEMPT, times up to 6
        # retry attempts (_call_with_retry's default), a worst case over
        # an hour for a single book (found live, 2026-08-17: a batch
        # stalled with 4 connections to KISSKI stuck ESTABLISHED for 20+
        # minutes, well past this script's typical successful per-call
        # latency). 90s is generous for a 1-4 page TOC scan's vision call
        # while still bounding the worst case.
        client = AsyncOpenAI(base_url=DEFAULT_KISSKI_BASE_URL, api_key=api_key, timeout=90.0)
        return (
            ModelEndpoint(label="kisski", model_id=model_ids[0], client=client),
            ModelEndpoint(label="kisski", model_id=model_ids[1], client=client),
        )
    if len(endpoint_aliases) != 2:
        raise SystemExit(
            f"--endpoint requires exactly 2 aliases for the two-independent-model gate, "
            f"got {len(endpoint_aliases)}: {endpoint_aliases}"
        )
    return tuple(resolve_endpoint_from_env(alias) for alias in endpoint_aliases)
```

Replace `_generate` (lines 374-415) with:

```python
def _generate(args: argparse.Namespace) -> int:
    cdir = corpus_dir(_CORPUS_NAME)
    eval_tier_path = cdir / "eval_tier_ids.json"
    eval_tier_ids = set(json.loads(eval_tier_path.read_text(encoding="utf-8"))) if eval_tier_path.exists() else set()
    rejected_path = cdir / "arbitration-rejected.json"
    rejected_ids = (
        {entry["key"] for entry in json.loads(rejected_path.read_text(encoding="utf-8"))["rejected"]}
        if rejected_path.exists() else set()
    )

    books = load_manifest_books(_CORPUS_NAME)
    eligible = [b for b in books if _still_needs_a_decision(b, cdir, eval_tier_ids, rejected_ids)]
    if args.limit is not None:
        eligible = eligible[: args.limit]
    candidates = [(manifest_key(b), cdir / b["filename"]) for b in eligible if (cdir / b["filename"]).exists()]
    missing_pdf_count = len(eligible) - len(candidates)

    endpoints = _resolve_vision_endpoints(args.endpoint)

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
    print(f"{len(passed)}/{len(results)} books passed the gate and got .expected.json written.")
    for reason, count in sorted(by_reason.items()):
        print(f"  {count} skipped: {reason}")
    if missing_pdf_count:
        print(f"  {missing_pdf_count} skipped: missing_pdf (not downloaded locally)")
    return 0
```

Replace `main()` (lines 418-429) with:

```python
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--limit", type=int, default=None, help="Process at most this many books (smoke-test convenience)")
    parser.add_argument("--concurrency", type=int, default=4, help="How many books to process concurrently (default: 4)")
    parser.add_argument(
        "--spot-check", type=int, default=None, metavar="N",
        help="Instead of generating, sample N passing bulk-tier books and walk through a visual Accept/Reject check",
    )
    parser.add_argument(
        "--endpoint", action="append", default=None, metavar="ALIAS",
        help="Use an explicit OpenAI-compatible endpoint instead of KISSKI auto-discovery -- pass exactly twice "
             "(the gate needs two independent reads), e.g. --endpoint MPCDF_A --endpoint MPCDF_B. Each ALIAS must "
             "have <ALIAS>_BASE_URL, <ALIAS>_API_KEY, <ALIAS>_MODEL set in the environment.",
    )
    args = parser.parse_args()
    if args.spot_check is not None:
        return _spot_check(corpus_dir(_CORPUS_NAME), args.spot_check)
    return _generate(args)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_generate_dnb_toc_ground_truth.py -v`
Expected: PASS, all tests in the file (including the new `TestResolveVisionEndpoints` class).

- [ ] **Step 5: Manual CLI smoke check (no network calls)**

Run: `uv run python evaluation/scripts/generate_dnb_toc_ground_truth.py --help`
Expected: help text shows the new `--endpoint ALIAS` option alongside `--limit`/`--concurrency`/`--spot-check`.

- [ ] **Step 6: Commit**

```bash
git add evaluation/scripts/generate_dnb_toc_ground_truth.py tests/test_generate_dnb_toc_ground_truth.py
git commit -m "feat(dnb-toc-gate): add --endpoint flag to target non-KISSKI inference endpoints"
```

---

### Task 4: `refresh_llm_cache.py` -- `_OpenAICompatibleLLMClient` wraps a given client

**Files:**
- Modify: `evaluation/refresh_llm_cache.py:72-109`
- Test: `tests/test_refresh_llm_cache.py`

- [ ] **Step 1: Write the failing test**

In `tests/test_refresh_llm_cache.py`, replace the existing import block (lines 13-21) with:

```python
from evaluation.refresh_llm_cache import (
    _OpenAICompatibleLLMClient,
    _all_cached_model_ids,
    _call_with_retry,
    _fully_covered_model_ids,
    _has_cached_entry,
    _process_model,
    _run_book_for_model,
    _upsert_cache,
)
```

Add a new test class (anywhere after the imports, e.g. right before `class TestFullyCoveredModelIds`):

```python
class TestOpenAICompatibleLLMClient(unittest.IsolatedAsyncioTestCase):
    async def test_uses_the_given_client_not_a_new_one(self):
        fake_client = unittest.mock.MagicMock()
        message = unittest.mock.MagicMock()
        message.content = "hello"
        choice = unittest.mock.MagicMock()
        choice.message = message
        response = unittest.mock.MagicMock()
        response.choices = [choice]
        fake_client.chat.completions.create = unittest.mock.AsyncMock(return_value=response)

        llm_client = _OpenAICompatibleLLMClient(model="model-x", client=fake_client)
        result = await llm_client.generate("prompt", max_tokens=10, temperature=0.0)

        self.assertEqual(result, "hello")
        fake_client.chat.completions.create.assert_awaited_once_with(
            model="model-x", messages=[{"role": "user", "content": "prompt"}], max_tokens=10, temperature=0.0,
        )
```

(This file already does `import unittest.mock` at the top -- confirm that import is present; if the file uses `from unittest.mock import ...` instead, adjust `unittest.mock.MagicMock`/`AsyncMock` to the imported names accordingly.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_refresh_llm_cache.py::TestOpenAICompatibleLLMClient -v`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'client'` (current constructor takes `base_url`/`api_key`, not `client`).

- [ ] **Step 3: Update the implementation**

In `evaluation/refresh_llm_cache.py`, add `AsyncOpenAI` to the top-level imports (this task only needs this one addition -- `ModelEndpoint`/`resolve_endpoint_from_env`/`KisskiModel` aren't used until Task 5, which adds them then so this task's diff doesn't carry unused imports). Change:

```python
from chapter_segmentation.segmentation import analyze_attachment_llm_only
from evaluation.harness import available_public_books, list_corpora, llm_cache_dir, public_pages_for
from evaluation.kisski import DEFAULT_KISSKI_BASE_URL, fetch_kisski_models, select_full_regen, select_gap_fill, select_top5
```

to:

```python
from openai import AsyncOpenAI

from chapter_segmentation.segmentation import analyze_attachment_llm_only
from evaluation.harness import available_public_books, list_corpora, llm_cache_dir, public_pages_for
from evaluation.kisski import DEFAULT_KISSKI_BASE_URL, fetch_kisski_models, select_full_regen, select_gap_fill, select_top5
```

Replace the `_OpenAICompatibleLLMClient` class with:

```python
class _OpenAICompatibleLLMClient:
    """Minimal LLMClient (see chapter_segmentation.llm.LLMClient) wrapping
    an already-built AsyncOpenAI client -- callers construct the client
    themselves (KISSKI's own base_url/api_key, or a ModelEndpoint's
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
```

The one existing call site inside `_main` currently reads:

```python
        llm_client = _OpenAICompatibleLLMClient(model=model.id, base_url=base_url, api_key=api_key)
```

Update it to build the client explicitly:

```python
        llm_client = _OpenAICompatibleLLMClient(model=model.id, client=AsyncOpenAI(base_url=base_url, api_key=api_key))
```

(This one-line edit keeps `_main` runnable after this task's constructor-signature change. Task 5 replaces the rest of `_main`'s body around it.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_refresh_llm_cache.py -v`
Expected: PASS, every test in the file (the new `TestOpenAICompatibleLLMClient` plus all pre-existing tests, unaffected).

- [ ] **Step 5: Commit**

```bash
git add evaluation/refresh_llm_cache.py tests/test_refresh_llm_cache.py
git commit -m "refactor(refresh-llm-cache): _OpenAICompatibleLLMClient wraps a given AsyncOpenAI client"
```

---

### Task 5: `refresh_llm_cache.py` -- `--endpoint` CLI + `_main` branch

**Files:**
- Modify: `evaluation/refresh_llm_cache.py:181-330` (`_run_book_for_model` through the bottom `if __name__ == "__main__":` block)
- Test: `tests/test_refresh_llm_cache.py`

- [ ] **Step 1: Write the failing test**

In `tests/test_refresh_llm_cache.py`, add an `evaluation.inference_endpoints` import and extend the `evaluation.refresh_llm_cache` import block (as left by Task 4's Step 1) with `_model_and_client_for_endpoint`:

```python
from evaluation.inference_endpoints import ModelEndpoint
from evaluation.refresh_llm_cache import (
    _OpenAICompatibleLLMClient,
    _all_cached_model_ids,
    _call_with_retry,
    _fully_covered_model_ids,
    _has_cached_entry,
    _model_and_client_for_endpoint,
    _process_model,
    _run_book_for_model,
    _upsert_cache,
)
```

Add a new test class:

```python
class TestModelAndClientForEndpoint(unittest.TestCase):
    def test_wraps_endpoint_with_demand_zero_and_the_endpoints_own_client(self):
        fake_client = unittest.mock.MagicMock()
        endpoint = ModelEndpoint(label="MPCDF_A", model_id="Qwen/Qwen3-VL-30B", client=fake_client)

        model, llm_client = _model_and_client_for_endpoint(endpoint)

        self.assertEqual(model.id, "Qwen/Qwen3-VL-30B")
        self.assertEqual(model.demand, 0)
        self.assertIsInstance(llm_client, _OpenAICompatibleLLMClient)
        self.assertIs(llm_client._client, fake_client)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_refresh_llm_cache.py::TestModelAndClientForEndpoint -v`
Expected: FAIL with `ImportError: cannot import name '_model_and_client_for_endpoint'`

- [ ] **Step 3: Implement `_model_and_client_for_endpoint` and wire `_main`/CLI**

In `evaluation/refresh_llm_cache.py`, add `ModelEndpoint`/`resolve_endpoint_from_env` and `KisskiModel` to the imports. Change:

```python
from evaluation.harness import available_public_books, list_corpora, llm_cache_dir, public_pages_for
from evaluation.kisski import DEFAULT_KISSKI_BASE_URL, fetch_kisski_models, select_full_regen, select_gap_fill, select_top5
```

to:

```python
from evaluation.harness import available_public_books, list_corpora, llm_cache_dir, public_pages_for
from evaluation.inference_endpoints import ModelEndpoint, resolve_endpoint_from_env
from evaluation.kisski import (
    DEFAULT_KISSKI_BASE_URL, KisskiModel, fetch_kisski_models, select_full_regen, select_gap_fill, select_top5,
)
```

Add this function right before `async def _main(...)`:

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

Replace `async def _main(...)` through the end of the file with:

```python
async def _main(
    mode: Optional[str], endpoint_aliases: Optional[list[str]], base_url: str, limit: int,
    corpus: Optional[str], clear: bool, concurrency: int,
) -> int:
    corpora = [corpus] if corpus else list_corpora()
    # (corpus, manifest_key, cache_dir) for every scorable book across every in-scope corpus.
    book_entries: list[tuple[str, str, Path]] = [
        (c, manifest_key, llm_cache_dir(c))
        for c in corpora
        for manifest_key, _expected_path, _book in available_public_books(c)
    ]
    if not book_entries:
        print("No public-cache evaluation books present.")
        return 1
    book_specs = [(cache_dir, manifest_key) for _corpus, manifest_key, cache_dir in book_entries]

    if clear:
        cleared = 0
        for _corpus, manifest_key, cache_dir in book_entries:
            cache_path = cache_dir / f"{manifest_key}.json"
            if cache_path.exists():
                cache_path.unlink()
                cleared += 1
        print(f"--clear: removed {cleared} cache file(s) across {len(corpora)} corpus/corpora before regenerating.")

    if endpoint_aliases:
        print(f"Selected endpoints: {endpoint_aliases}")
        for alias in endpoint_aliases:
            endpoint = resolve_endpoint_from_env(alias)
            model, llm_client = _model_and_client_for_endpoint(endpoint)
            worker = functools.partial(_run_book_for_model, model=model, mode="endpoint", llm_client=llm_client)
            await _process_model(book_entries, concurrency, worker)
        return 0

    api_key = os.environ["KISSKI_API_KEY"]
    all_models = fetch_kisski_models(base_url, api_key)
    if mode == "top5":
        selected = select_top5(all_models, limit=limit)
    elif mode == "fill-gaps":
        selected = select_gap_fill(all_models, _fully_covered_model_ids(book_specs), limit=limit)
    else:
        cached_ids = _all_cached_model_ids(book_specs)
        selected = select_full_regen(all_models, cached_ids)
        retired = sorted(cached_ids - {m.id for m in all_models})
        if retired:
            print(f"Skipping cached models no longer offered by KISSKI: {retired}")

    if not selected:
        if mode == "fill-gaps":
            print("No models to run (fill-gaps: every non-busy model already fully covered).")
        elif mode == "full":
            print("No models to run (full: no models are cached yet).")
        else:
            print("No models to run.")
        return 0

    print(f"Selected models: {[m.id for m in selected]}")
    for model in selected:
        llm_client = _OpenAICompatibleLLMClient(model=model.id, client=AsyncOpenAI(base_url=base_url, api_key=api_key))
        worker = functools.partial(_run_book_for_model, model=model, mode=mode, llm_client=llm_client)
        await _process_model(book_entries, concurrency, worker)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--mode", choices=["top5", "fill-gaps", "full"], default=None)
    mode_group.add_argument(
        "--endpoint", action="append", default=None, metavar="ALIAS",
        help="Use one or more explicit OpenAI-compatible endpoints instead of KISSKI discovery -- repeatable, "
             "e.g. --endpoint MPCDF_A --endpoint MPCDF_B. Each ALIAS must have <ALIAS>_BASE_URL, "
             "<ALIAS>_API_KEY, <ALIAS>_MODEL set in the environment. Runs every given endpoint once over the "
             "corpus (unconditionally, like --mode full); --mode's top5/fill-gaps/full sweep-a-shared-pool "
             "semantics don't apply since there's no discovery involved -- you already know exactly which "
             "model(s) you deployed.",
    )
    parser.add_argument("--base-url", default=DEFAULT_KISSKI_BASE_URL)
    parser.add_argument(
        "--limit", type=int, default=5,
        help="Max models to run this invocation (top5/fill-gaps only; full is always uncapped). Default 5.",
    )
    parser.add_argument("--corpus", help="Only refresh this corpus (default: every corpus under evaluation/corpus/)")
    parser.add_argument(
        "--clear", action="store_true",
        help="Delete every cache file in scope before regenerating (use when the underlying public-cache text changed).",
    )
    parser.add_argument(
        "--concurrency", type=int, default=4,
        help="Max concurrent book requests per model. KISSKI publishes no documented rate limit, "
             "so this is a conservative default -- raise it if you don't observe 429s. Default 4.",
    )
    args = parser.parse_args()
    if args.mode is None and not args.endpoint:
        args.mode = "top5"
    raise SystemExit(asyncio.run(_main(
        mode=args.mode, endpoint_aliases=args.endpoint, base_url=args.base_url, limit=args.limit,
        corpus=args.corpus, clear=args.clear, concurrency=args.concurrency,
    )))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_refresh_llm_cache.py -v`
Expected: PASS, every test in the file.

- [ ] **Step 5: Manual CLI smoke check (no network calls)**

Run: `uv run python evaluation/refresh_llm_cache.py --help`
Expected: help text shows `--mode` and `--endpoint` as mutually exclusive options, both still listed with their descriptions.

Run: `uv run python evaluation/refresh_llm_cache.py --mode top5 --endpoint MPCDF_A`
Expected: argparse error, `argument --endpoint: not allowed with argument --mode`, exit code 2.

- [ ] **Step 6: Commit**

```bash
git add evaluation/refresh_llm_cache.py tests/test_refresh_llm_cache.py
git commit -m "feat(refresh-llm-cache): add --endpoint flag to target non-KISSKI inference endpoints"
```

---

### Task 6: Documentation

**Files:**
- Modify: `evaluation/scripts/README.md` (the `## \`generate_dnb_toc_ground_truth.py\`` section, currently lines 358-401)
- Modify: `evaluation/refresh_llm_cache.py:1-70` (module docstring)

- [ ] **Step 1: Regenerate the `--help` dump for `generate_dnb_toc_ground_truth.py`**

Run: `uv run python evaluation/scripts/generate_dnb_toc_ground_truth.py --help`

Copy the real output and replace the fenced block in `evaluation/scripts/README.md`'s `## \`generate_dnb_toc_ground_truth.py\`` section (the ` ``` ` block currently spanning lines 365-401) with it verbatim, keeping the one-line prose summary above the block unchanged (it already correctly describes the script; only the flag list needs the `--endpoint` addition, which the regenerated `--help` output supplies automatically).

- [ ] **Step 2: Update `refresh_llm_cache.py`'s module docstring**

In `evaluation/refresh_llm_cache.py`, insert a new paragraph into the module docstring (top of file, after the existing `--mode full: ...` paragraph, before the closing `"""`):

```
--endpoint ALIAS (repeatable): bypasses KISSKI discovery entirely and
runs the corpus against the given OpenAI-compatible endpoint(s) instead
-- e.g. an MPCDF LLM Inference Service session
(https://llm.mpcdf.mpg.de). Each ALIAS must have <ALIAS>_BASE_URL,
<ALIAS>_API_KEY, <ALIAS>_MODEL set in the environment (see
evaluation/inference_endpoints.py). Mutually exclusive with --mode --
there's no discovery/demand concept for a model you deployed yourself,
so top5/fill-gaps/full's sweep-a-shared-pool semantics don't apply; every
given endpoint just runs once, unconditionally, over the corpus.
```

- [ ] **Step 3: Verify the docstring change doesn't break anything**

Run: `uv run python evaluation/refresh_llm_cache.py --help`
Expected: the new paragraph appears in the printed help text (the script uses `formatter_class=argparse.RawDescriptionHelpFormatter`, so the whole docstring prints as-is).

- [ ] **Step 4: Commit**

```bash
git add evaluation/scripts/README.md evaluation/refresh_llm_cache.py
git commit -m "docs: document --endpoint for generate_dnb_toc_ground_truth.py and refresh_llm_cache.py"
```

---

### Task 7: Full verification pass

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite**

Run: `uv run pytest -q`
Expected: all tests pass, no new failures or errors anywhere in the suite (this exercises `tests/test_generate_dnb_toc_ground_truth.py`, `tests/test_refresh_llm_cache.py`, `tests/test_inference_endpoints.py`, and every other existing test file, confirming Tasks 1-6 didn't regress anything outside their own files -- e.g. `tests/test_dnb_toc_vision.py`, which imports `vision_extract_toc_entries` and is unaffected since that function's own signature never changed).

- [ ] **Step 2: Confirm both scripts still run their default (no-`--endpoint`) path structurally**

Run: `uv run python -c "import evaluation.scripts.generate_dnb_toc_ground_truth; import evaluation.refresh_llm_cache; print('imports OK')"`
Expected: `imports OK`, no `ImportError`/`SyntaxError` (a lightweight sanity check that every edit across Tasks 2-5 left both modules importable, without spending real KISSKI/MPCDF API budget).

- [ ] **Step 3: Report status**

No commit for this task (verification only). If Step 1 or Step 2 fails, stop and fix before considering the plan complete.
