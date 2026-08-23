"""Unit tests for cli/generate_ground_truth.py's pure logic. The real
PDF-reading/vision-LLM-calling main() is exercised manually against the
real corpus with a real .endpoints file -- see design spec
docs/superpowers/specs/2026-08-21-dnb-toc-ground-truth-extraction-design.md."""

import argparse
import asyncio
import dataclasses
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from openai import RateLimitError
from pypdf import PdfWriter

from dnb_toc_ground_truth import corpus
from dnb_toc_ground_truth.inference import ModelEndpoint, load_endpoint_entries, resolve_model_endpoints
from dnb_toc_ground_truth.toc_entry import TocEntry
from dnb_toc_ground_truth.vision import load_cached_kind, load_cached_llm_entries, write_cached_llm_entries

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cli"))

from generate_ground_truth import (
    _acquire_lock,
    _binding_rate_limit_window,
    _call_with_retry,
    _is_stale_bulk_gate_entry,
    _lock_path,
    _release_lock,
    _resolve_endpoints,
    _retry_after_seconds,
    _run_book,
    _run_book_entries,
    _still_needs_a_decision,
)


def _rate_limit_error(headers: dict) -> RateLimitError:
    return RateLimitError(
        "rate limited",
        response=httpx.Response(
            429, headers=headers, request=httpx.Request("POST", "https://example.com"),
        ),
        body=None,
    )


def _entry(title: str, page: int, authors: tuple[str, ...] = ()) -> TocEntry:
    return TocEntry(title=title, printed_page_number=page, source_page_index=-1, authors=authors)


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

    async def test_rate_limit_error_gets_the_longer_linear_backoff(self):
        rate_limit_error = RateLimitError(
            "rate limited", response=httpx.Response(429, request=httpx.Request("POST", "https://example.com")), body=None,
        )
        coro_fn = AsyncMock(side_effect=[rate_limit_error, rate_limit_error, "ok"])
        sleep = AsyncMock()

        result = await _call_with_retry(coro_fn, attempts=3, base_delay=2.0, rate_limit_delay=20.0, sleep=sleep)

        self.assertEqual(result, "ok")
        sleep.assert_any_call(20.0)
        sleep.assert_any_call(40.0)

    async def test_non_rate_limit_error_keeps_the_short_exponential_backoff(self):
        coro_fn = AsyncMock(side_effect=[RuntimeError("boom"), "ok"])
        sleep = AsyncMock()

        result = await _call_with_retry(coro_fn, attempts=2, base_delay=2.0, rate_limit_delay=20.0, sleep=sleep)

        self.assertEqual(result, "ok")
        sleep.assert_called_once_with(2.0)

    async def test_rate_limit_error_with_retry_after_header_sleeps_that_exact_value(self):
        # A "minute" window is inline-retryable -- must use the server's own
        # retry-after value, not the blind rate_limit_delay*attempt formula.
        error = _rate_limit_error({"retry-after": "37", "x-ratelimit-remaining-minute": "0"})
        coro_fn = AsyncMock(side_effect=[error, "ok"])
        sleep = AsyncMock()

        result = await _call_with_retry(coro_fn, attempts=3, base_delay=2.0, rate_limit_delay=20.0, sleep=sleep)

        self.assertEqual(result, "ok")
        sleep.assert_called_once_with(37.0)

    async def test_rate_limit_error_with_hour_window_also_retries_inline(self):
        error = _rate_limit_error({"retry-after": "900", "x-ratelimit-remaining-hour": "0"})
        coro_fn = AsyncMock(side_effect=[error, "ok"])
        sleep = AsyncMock()

        result = await _call_with_retry(coro_fn, attempts=3, sleep=sleep)

        self.assertEqual(result, "ok")
        sleep.assert_called_once_with(900.0)

    async def test_rate_limit_error_bound_by_day_window_gives_up_immediately(self):
        # The whole point: a day-scale quota won't reset within this run,
        # so no further attempts should even be made, regardless of
        # `attempts` -- see _call_with_retry's docstring.
        error = _rate_limit_error({"retry-after": "54179", "x-ratelimit-remaining-day": "0"})
        coro_fn = AsyncMock(side_effect=error)
        sleep = AsyncMock()

        with self.assertRaises(RateLimitError):
            await _call_with_retry(coro_fn, attempts=6, sleep=sleep)

        coro_fn.assert_awaited_once()
        sleep.assert_not_awaited()

    async def test_rate_limit_error_with_no_headers_falls_back_to_linear_backoff(self):
        error = _rate_limit_error({})
        coro_fn = AsyncMock(side_effect=[error, "ok"])
        sleep = AsyncMock()

        result = await _call_with_retry(coro_fn, attempts=3, base_delay=2.0, rate_limit_delay=20.0, sleep=sleep)

        self.assertEqual(result, "ok")
        sleep.assert_called_once_with(20.0)

    async def test_a_call_that_never_returns_is_killed_by_the_hard_wall_clock_timeout(self):
        # A request that hangs forever (e.g. a proxy trickling keep-alive
        # bytes so the HTTP client's own per-read timeout never fires)
        # must still be treated as a retryable failure, not block forever.
        # A real coroutine function (not an AsyncMock side_effect list) --
        # AsyncMock doesn't await a coroutine OBJECT handed to it as a
        # side_effect item, it just returns it unawaited, so it can't
        # simulate a genuinely long-running call.
        calls = 0

        async def coro_fn():
            nonlocal calls
            calls += 1
            if calls == 1:
                await asyncio.sleep(3600)
            return "ok"

        sleep = AsyncMock()

        result = await _call_with_retry(coro_fn, attempts=2, call_timeout=0.01, sleep=sleep)

        self.assertEqual(result, "ok")
        self.assertEqual(calls, 2)
        sleep.assert_called_once_with(2.0)

    async def test_hard_timeout_exhausting_all_attempts_raises_a_descriptive_timeout_error(self):
        async def coro_fn():
            await asyncio.sleep(3600)

        sleep = AsyncMock()

        with self.assertRaises(TimeoutError) as ctx:
            await _call_with_retry(coro_fn, attempts=2, call_timeout=0.01, sleep=sleep)

        self.assertIn("0s", str(ctx.exception))


class TestBindingRateLimitWindow(unittest.TestCase):
    def test_none_when_no_headers(self):
        self.assertIsNone(_binding_rate_limit_window({}))
        self.assertIsNone(_binding_rate_limit_window(None))

    def test_none_when_no_remaining_header_is_zero(self):
        headers = {"x-ratelimit-remaining-day": "5", "x-ratelimit-remaining-minute": "1"}
        self.assertIsNone(_binding_rate_limit_window(headers))

    def test_picks_the_single_zeroed_window(self):
        self.assertEqual(_binding_rate_limit_window({"x-ratelimit-remaining-minute": "0"}), "minute")
        self.assertEqual(_binding_rate_limit_window({"x-ratelimit-remaining-hour": "0"}), "hour")
        self.assertEqual(_binding_rate_limit_window({"x-ratelimit-remaining-day": "0"}), "day")

    def test_prefers_the_longest_window_when_several_are_zeroed(self):
        headers = {
            "x-ratelimit-remaining-minute": "0",
            "x-ratelimit-remaining-hour": "0",
            "x-ratelimit-remaining-day": "0",
        }
        self.assertEqual(_binding_rate_limit_window(headers), "day")


class TestRetryAfterSeconds(unittest.TestCase):
    def test_none_when_absent(self):
        self.assertIsNone(_retry_after_seconds({}))
        self.assertIsNone(_retry_after_seconds(None))

    def test_parses_a_numeric_value(self):
        self.assertEqual(_retry_after_seconds({"retry-after": "54179"}), 54179.0)

    def test_none_for_an_unparseable_value(self):
        self.assertIsNone(_retry_after_seconds({"retry-after": "not-a-number"}))


class TestRunBookEntries(unittest.TestCase):
    def test_passing_book_writes_expected_json(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            a = [_entry("Einleitung", 9), _entry("Schluss", 40)]
            b = [_entry("Einleitung", 9), _entry("Schluss", 40)]

            key, passed, reason = _run_book_entries("book1", [a, b], 0.90)

            self.assertEqual(key, "book1")
            self.assertTrue(passed)
            self.assertEqual(reason, "ok")
            gt_path = corpus.expected_json_path("book1")
            self.assertTrue(gt_path.exists())
            data = json.loads(gt_path.read_text(encoding="utf-8"))
            self.assertFalse(data["verified"])
            self.assertEqual(data["source"], "bulk_gate")
            self.assertEqual(len(data["entries"]), 2)
            self.assertIn("skip", data["entries"][0])

    def test_below_threshold_book_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            a = [_entry("Einleitung", 9), _entry("A", 20), _entry("B", 30), _entry("C", 40)]
            b = [_entry("Einleitung", 9)]

            key, passed, reason = _run_book_entries("book2", [a, b], 0.90)

            self.assertFalse(passed)
            self.assertEqual(reason, "below_threshold")
            self.assertFalse(corpus.expected_json_path("book2").exists())

    def test_no_entries_from_either_side_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            key, passed, reason = _run_book_entries("book3", [[], []], 0.90)

            self.assertFalse(passed)
            self.assertEqual(reason, "no_entries")
            self.assertFalse(corpus.expected_json_path("book3").exists())

    def test_a_single_requested_endpoint_is_reported_as_single_reading_only(self):
        # A one-element entries_by_endpoint is a legitimate call shape --
        # e.g. `--use-vision numind/NuExtract3` alone, backfilling one
        # more model's reading onto books that already have others
        # cached under different endpoints. gate_books itself raises
        # ValueError below 2 lists; this must be turned into an ordinary
        # skip reason before it ever reaches gate_books, not surface as
        # an uncaught exception.
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            a = [_entry("Einleitung", 9), _entry("Schluss", 40)]

            key, passed, reason = _run_book_entries("book4", [a], 0.90)

            self.assertFalse(passed)
            self.assertEqual(reason, "single_reading_only")
            self.assertFalse(corpus.expected_json_path("book4").exists())


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


def _endpoint(
    model_id: str, client, kind: str = "vision", extraction_api: str = "", extraction_instructions: bool = True,
) -> ModelEndpoint:
    return ModelEndpoint(
        label="test", model_id=model_id, kind=kind, client=client,
        extraction_api=extraction_api, extraction_instructions=extraction_instructions,
    )


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
    async def test_records_a_positive_duration_seconds_in_the_cache_file(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp) / "corpus"):
            tmp_path = Path(tmp)
            cache_directory = tmp_path / "cache"
            pdf_path = _make_pdf(tmp_path / "book.pdf")
            client = _fake_vision_client(_VISION_RESPONSE)
            endpoints = [_endpoint("model-a", client), _endpoint("model-b", client)]
            semaphore = asyncio.Semaphore(1)

            await _run_book("book_dur", pdf_path, endpoints, semaphore, cache_directory, 0.90, sleep=AsyncMock())

            from dnb_toc_ground_truth.vision import cache_path
            data = json.loads(cache_path(cache_directory, "book_dur", "model-a").read_text(encoding="utf-8"))
            self.assertIsInstance(data["duration_seconds"], float)
            self.assertGreaterEqual(data["duration_seconds"], 0.0)

    async def test_calls_each_model_once_and_writes_on_agreement(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp) / "corpus"):
            tmp_path = Path(tmp)
            cache_directory = tmp_path / "cache"
            pdf_path = _make_pdf(tmp_path / "book.pdf")
            client = _fake_vision_client(_VISION_RESPONSE)
            endpoints = [_endpoint("model-a", client), _endpoint("model-b", client)]
            semaphore = asyncio.Semaphore(1)

            key, passed, reason = await _run_book(
                "book1", pdf_path, endpoints, semaphore, cache_directory, 0.90, sleep=AsyncMock(),
            )

            self.assertTrue(passed)
            self.assertEqual(reason, "ok")
            self.assertEqual(client.chat.completions.create.await_count, 2)
            self.assertTrue(corpus.expected_json_path("book1").exists())

    async def test_cached_model_entries_are_reused_without_a_new_call(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp) / "corpus"):
            tmp_path = Path(tmp)
            cache_directory = tmp_path / "cache"
            pdf_path = _make_pdf(tmp_path / "book.pdf")
            entries = [_entry("Einleitung", 9), _entry("Schluss", 40)]
            write_cached_llm_entries(cache_directory, "book2", "model-a", entries)
            write_cached_llm_entries(cache_directory, "book2", "model-b", entries)
            client = _fake_vision_client(_VISION_RESPONSE)
            endpoints = [_endpoint("model-a", client), _endpoint("model-b", client)]
            semaphore = asyncio.Semaphore(1)

            key, passed, reason = await _run_book(
                "book2", pdf_path, endpoints, semaphore, cache_directory, 0.90, sleep=AsyncMock(),
            )

            self.assertTrue(passed)
            client.chat.completions.create.assert_not_called()

    async def test_a_corrupt_pdf_is_reported_not_raised(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp) / "corpus"):
            tmp_path = Path(tmp)
            cache_directory = tmp_path / "cache"
            bad_pdf = tmp_path / "not-a-pdf.pdf"
            bad_pdf.write_text("this is not a pdf")
            client = _fake_vision_client(_VISION_RESPONSE)
            endpoints = [_endpoint("model-a", client), _endpoint("model-b", client)]
            semaphore = asyncio.Semaphore(1)

            key, passed, reason = await _run_book(
                "book3", bad_pdf, endpoints, semaphore, cache_directory, 0.90, sleep=AsyncMock(),
            )

            self.assertFalse(passed)
            self.assertTrue(reason.startswith("error:"))

    async def test_one_model_failing_preserves_the_others_cache_entry(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp) / "corpus"):
            tmp_path = Path(tmp)
            cache_directory = tmp_path / "cache"
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
            endpoints = [_endpoint("model-a", client), _endpoint("model-b", client)]
            semaphore = asyncio.Semaphore(1)

            key, passed, reason = await _run_book(
                "book4", pdf_path, endpoints, semaphore, cache_directory, 0.90, sleep=AsyncMock(),
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
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp) / "corpus"):
            tmp_path = Path(tmp)
            cache_directory = tmp_path / "cache"
            pdf_path = _make_pdf(tmp_path / "book.pdf")
            client = MagicMock()
            client.chat.completions.create = AsyncMock(side_effect=RuntimeError("boom"))
            endpoints = [_endpoint("model-a", client), _endpoint("model-b", client)]
            semaphore = asyncio.Semaphore(1)
            observed_lock_state_during_sleep = []

            async def spying_sleep(_delay):
                observed_lock_state_during_sleep.append(semaphore.locked())

            await _run_book(
                "book5", pdf_path, endpoints, semaphore, cache_directory, 0.90, sleep=spying_sleep,
            )

            self.assertTrue(observed_lock_state_during_sleep, "sleep (backoff) was never invoked")
            self.assertTrue(
                all(not locked for locked in observed_lock_state_during_sleep),
                "semaphore was still held during a backoff sleep",
            )

    async def test_two_independent_endpoints_each_get_their_own_client_called(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp) / "corpus"):
            tmp_path = Path(tmp)
            cache_directory = tmp_path / "cache"
            pdf_path = _make_pdf(tmp_path / "book.pdf")
            client_a = _fake_vision_client(_VISION_RESPONSE)
            client_b = _fake_vision_client(_VISION_RESPONSE)
            endpoints = [_endpoint("model-a", client_a), _endpoint("model-b", client_b)]
            semaphore = asyncio.Semaphore(1)

            key, passed, reason = await _run_book(
                "book6", pdf_path, endpoints, semaphore, cache_directory, 0.90, sleep=AsyncMock(),
            )

            self.assertTrue(passed)
            client_a.chat.completions.create.assert_awaited_once()
            client_b.chat.completions.create.assert_awaited_once()
            self.assertEqual(client_a.chat.completions.create.await_args.kwargs["model"], "model-a")
            self.assertEqual(client_b.chat.completions.create.await_args.kwargs["model"], "model-b")

    async def test_a_book_whose_lock_is_already_held_is_skipped_without_calling_any_model(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp) / "corpus"):
            tmp_path = Path(tmp)
            cache_directory = tmp_path / "cache"
            pdf_path = _make_pdf(tmp_path / "book.pdf")
            client = _fake_vision_client(_VISION_RESPONSE)
            endpoints = [_endpoint("model-a", client), _endpoint("model-b", client)]
            semaphore = asyncio.Semaphore(1)
            _acquire_lock("book7")  # simulate another process already working on it

            key, passed, reason = await _run_book(
                "book7", pdf_path, endpoints, semaphore, cache_directory, 0.90, sleep=AsyncMock(),
            )

            self.assertFalse(passed)
            self.assertEqual(reason, "locked_by_another_process")
            client.chat.completions.create.assert_not_called()

    async def test_the_lock_is_released_after_a_book_finishes_successfully(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp) / "corpus"):
            tmp_path = Path(tmp)
            cache_directory = tmp_path / "cache"
            pdf_path = _make_pdf(tmp_path / "book.pdf")
            client = _fake_vision_client(_VISION_RESPONSE)
            endpoints = [_endpoint("model-a", client), _endpoint("model-b", client)]
            semaphore = asyncio.Semaphore(1)

            await _run_book(
                "book8", pdf_path, endpoints, semaphore, cache_directory, 0.90, sleep=AsyncMock(),
            )

            self.assertFalse(_lock_path("book8").exists())

    async def test_the_lock_is_released_even_when_the_book_errors(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp) / "corpus"):
            tmp_path = Path(tmp)
            cache_directory = tmp_path / "cache"
            bad_pdf = tmp_path / "not-a-pdf.pdf"
            bad_pdf.write_text("this is not a pdf")
            client = _fake_vision_client(_VISION_RESPONSE)
            endpoints = [_endpoint("model-a", client), _endpoint("model-b", client)]
            semaphore = asyncio.Semaphore(1)

            await _run_book(
                "book9", bad_pdf, endpoints, semaphore, cache_directory, 0.90, sleep=AsyncMock(),
            )

            self.assertFalse(_lock_path("book9").exists())

    async def test_second_endpoint_kind_text_dispatches_to_text_extract_toc_entries(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp) / "corpus"):
            tmp_path = Path(tmp)
            cache_directory = tmp_path / "cache"
            pdf_path = _make_pdf(tmp_path / "book.pdf")
            client = _fake_vision_client(_VISION_RESPONSE)
            endpoints = [_endpoint("vision-model", client, kind="vision"), _endpoint("text-model", client, kind="text")]
            semaphore = asyncio.Semaphore(1)

            with patch(
                "generate_ground_truth.text_extract_toc_entries",
                new=AsyncMock(return_value=[_entry("Einleitung", 9), _entry("Schluss", 40)]),
            ) as mock_text_extract:
                key, passed, reason = await _run_book(
                    "book10", pdf_path, endpoints, semaphore, cache_directory, 0.90, sleep=AsyncMock(),
                )

            self.assertTrue(passed)
            mock_text_extract.assert_awaited_once()
            self.assertEqual(client.chat.completions.create.await_count, 1)
            self.assertEqual(load_cached_llm_entries(cache_directory, "book10", "text-model")[0].title, "Einleitung")
            self.assertEqual(load_cached_kind(cache_directory, "book10", "vision-model"), "vision")
            self.assertEqual(load_cached_kind(cache_directory, "book10", "text-model"), "text")

    async def test_all_vision_endpoints_are_cached_as_vision(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp) / "corpus"):
            tmp_path = Path(tmp)
            cache_directory = tmp_path / "cache"
            pdf_path = _make_pdf(tmp_path / "book.pdf")
            client = _fake_vision_client(_VISION_RESPONSE)
            endpoints = [_endpoint("model-a", client), _endpoint("model-b", client)]
            semaphore = asyncio.Semaphore(1)

            key, passed, reason = await _run_book(
                "book11", pdf_path, endpoints, semaphore, cache_directory, 0.90, sleep=AsyncMock(),
            )

            self.assertTrue(passed)
            self.assertEqual(load_cached_kind(cache_directory, "book11", "model-a"), "vision")
            self.assertEqual(load_cached_kind(cache_directory, "book11", "model-b"), "vision")

    async def test_a_cache_entry_written_under_a_different_kind_is_not_trusted(self):
        # Regression test: cache_path keys purely on (key, model_id), not
        # kind -- if the same model id was previously cached as a vision
        # endpoint's result and is now being requested as the text
        # endpoint (or vice versa), the stale wrong-kind entries must NOT
        # be silently reused. A mismatch must be treated exactly like a
        # cache miss: re-extract, and overwrite the cache with the
        # correct kind.
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp) / "corpus"):
            tmp_path = Path(tmp)
            cache_directory = tmp_path / "cache"
            pdf_path = _make_pdf(tmp_path / "book.pdf")
            stale_entries = [_entry("Stale vision-cached title", 1)]
            write_cached_llm_entries(cache_directory, "book12", "shared-model", stale_entries, kind="vision")
            client = _fake_vision_client(_VISION_RESPONSE)
            endpoints = [_endpoint("model-a", client), _endpoint("shared-model", client, kind="text")]
            semaphore = asyncio.Semaphore(1)

            with patch(
                "generate_ground_truth.text_extract_toc_entries",
                new=AsyncMock(return_value=[_entry("Einleitung", 9), _entry("Schluss", 40)]),
            ) as mock_text_extract:
                key, passed, reason = await _run_book(
                    "book12", pdf_path, endpoints, semaphore, cache_directory, 0.90, sleep=AsyncMock(),
                )

            self.assertTrue(passed)
            mock_text_extract.assert_awaited_once()
            self.assertEqual(load_cached_kind(cache_directory, "book12", "shared-model"), "text")
            self.assertEqual(load_cached_llm_entries(cache_directory, "book12", "shared-model")[0].title, "Einleitung")

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

    async def test_a_real_endpoints_file_entry_correctly_drives_nuextract_dispatch(self):
        # Join-point test: every layer above is well-covered in isolation
        # (test_inference.py covers .endpoints -> ModelEndpoint fields;
        # the other TestRunBook tests cover ModelEndpoint -> dispatch using
        # the hand-built _endpoint() helper), but nothing drives the full
        # chain -- a real .endpoints file, through the real
        # load_endpoint_entries/resolve_model_endpoints, into _run_book's
        # actual dispatch decision. This closes that gap.
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp) / "corpus"):
            tmp_path = Path(tmp)
            cache_directory = tmp_path / "cache"
            pdf_path = _make_pdf(tmp_path / "book.pdf")
            endpoints_path = tmp_path / ".endpoints"
            endpoints_path.write_text(json.dumps([
                {"url": "https://x.invalid/a", "key": "k1", "model": "model-a"},
                {
                    "url": "https://x.invalid/b", "key": "k2", "model": "acme/finetuned-nuextract2",
                    "extraction_api": "nuextract", "extraction_instructions": "false",
                },
            ]))

            entries = load_endpoint_entries(endpoints_path)
            resolved = resolve_model_endpoints(["model-a", "acme/finetuned-nuextract2"], "vision", entries)
            # resolve_model_endpoints builds a real AsyncOpenAI client per
            # entry (from the .endpoints file's url/key) -- swap in a fake
            # so _run_book's actual model calls don't hit the network,
            # while every other field (model_id, kind, extraction_api,
            # extraction_instructions) stays exactly what real endpoint-file
            # parsing and resolution produced.
            client = _fake_vision_client(_VISION_RESPONSE)
            endpoints = [dataclasses.replace(ep, client=client) for ep in resolved]
            semaphore = asyncio.Semaphore(1)

            with patch(
                "generate_ground_truth.nuextract_vision_extract_toc_entries",
                new=AsyncMock(return_value=[_entry("Einleitung", 9), _entry("Schluss", 40)]),
            ) as mock_nuextract:
                key, passed, reason = await _run_book(
                    "book16", pdf_path, endpoints, semaphore, cache_directory, 0.90, sleep=AsyncMock(),
                )

            self.assertTrue(passed)
            mock_nuextract.assert_awaited_once()
            self.assertFalse(mock_nuextract.call_args.kwargs["use_instructions"])


class TestAcquireLock(unittest.TestCase):
    def test_first_acquire_succeeds_and_creates_the_lock_file(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            self.assertTrue(_acquire_lock("book1"))
            self.assertTrue(_lock_path("book1").exists())

    def test_second_acquire_of_a_fresh_lock_fails(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            _acquire_lock("book1")
            self.assertFalse(_acquire_lock("book1"))

    def test_acquire_succeeds_again_after_release(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            _acquire_lock("book1")
            _release_lock("book1")
            self.assertTrue(_acquire_lock("book1"))

    def test_a_stale_lock_is_reclaimed(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            _acquire_lock("book1", stale_after=10.0)
            old = 1_000_000_000  # arbitrary, far enough in the past to be stale under any stale_after
            os.utime(_lock_path("book1"), (old, old))
            self.assertTrue(_acquire_lock("book1", stale_after=10.0))

    def test_release_of_a_lock_that_was_never_acquired_does_not_raise(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            _release_lock("book1")  # must not raise


class TestStillNeedsADecision(unittest.TestCase):
    def test_true_for_a_fresh_book(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            book = {"filename": "book1.pdf"}
            self.assertTrue(_still_needs_a_decision(book, set(), set()))

    def test_false_when_held_out_for_eval_tier(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            book = {"filename": "book1.pdf"}
            self.assertFalse(_still_needs_a_decision(book, {"book1"}, set()))

    def test_false_when_permanently_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            book = {"filename": "book1.pdf"}
            self.assertFalse(_still_needs_a_decision(book, set(), {"book1"}))

    def test_false_when_expected_json_already_exists(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            corpus.ground_truth_dir().mkdir(parents=True)
            corpus.expected_json_path("book1").write_text("{}", encoding="utf-8")
            book = {"filename": "book1.pdf"}
            self.assertFalse(_still_needs_a_decision(book, set(), set()))

    def test_true_for_a_stale_pre_skip_field_bulk_gate_file(self):
        # A bulk_gate file written before the 2026-08-17 extraction-standard
        # change has entries with no "skip" key at all -- it's missing
        # whatever lines the old prompt told the model to omit outright, so
        # it counts as undecided again rather than staying stuck forever.
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            corpus.ground_truth_dir().mkdir(parents=True)
            corpus.expected_json_path("book1").write_text(
                json.dumps({"entries": [{"title": "Einleitung", "authors": [], "printed_page_number": "9"}],
                            "verified": False, "source": "bulk_gate"}),
                encoding="utf-8",
            )
            book = {"filename": "book1.pdf"}
            self.assertTrue(_still_needs_a_decision(book, set(), set()))

    def test_false_for_a_current_schema_bulk_gate_file(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            corpus.ground_truth_dir().mkdir(parents=True)
            corpus.expected_json_path("book1").write_text(
                json.dumps({"entries": [{"title": "Einleitung", "authors": [], "printed_page_number": "9",
                                          "skip": False}],
                            "verified": False, "source": "bulk_gate"}),
                encoding="utf-8",
            )
            book = {"filename": "book1.pdf"}
            self.assertFalse(_still_needs_a_decision(book, set(), set()))

    def test_false_for_a_stale_arbitration_file_never_auto_reprocessed(self):
        # Unlike a stale bulk_gate file, an agent_arbitration file went
        # through direct human/AI-agent review -- it must never be silently
        # overwritten by an automated, unreviewed re-run just because it
        # also predates the "skip" key. Retrofitting it is a deliberate
        # manual task, not this function's job.
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            corpus.ground_truth_dir().mkdir(parents=True)
            corpus.expected_json_path("book1").write_text(
                json.dumps({"entries": [{"title": "Einleitung", "authors": [], "printed_page_number": "9"}],
                            "verified": True, "source": "agent_arbitration"}),
                encoding="utf-8",
            )
            book = {"filename": "book1.pdf"}
            self.assertFalse(_still_needs_a_decision(book, set(), set()))


class TestIsStaleBulkGateEntry(unittest.TestCase):
    def test_true_when_bulk_gate_entries_lack_skip_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "book1.expected.json"
            path.write_text(
                json.dumps({"entries": [{"title": "X", "authors": [], "printed_page_number": "1"}],
                            "verified": False, "source": "bulk_gate"}),
                encoding="utf-8",
            )
            self.assertTrue(_is_stale_bulk_gate_entry(path))

    def test_false_when_skip_key_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "book1.expected.json"
            path.write_text(
                json.dumps({"entries": [{"title": "X", "authors": [], "printed_page_number": "1", "skip": False}],
                            "verified": False, "source": "bulk_gate"}),
                encoding="utf-8",
            )
            self.assertFalse(_is_stale_bulk_gate_entry(path))

    def test_false_for_non_bulk_gate_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "book1.expected.json"
            path.write_text(
                json.dumps({"entries": [{"title": "X", "authors": [], "printed_page_number": "1"}],
                            "verified": True, "source": "agent_arbitration"}),
                encoding="utf-8",
            )
            self.assertFalse(_is_stale_bulk_gate_entry(path))

    def test_false_for_empty_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "book1.expected.json"
            path.write_text(
                json.dumps({"entries": [], "verified": False, "source": "bulk_gate"}), encoding="utf-8",
            )
            self.assertFalse(_is_stale_bulk_gate_entry(path))


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
