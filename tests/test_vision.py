"""Unit/integration tests for vision.py -- vision-LLM
TOC extraction for dnb-toc-only, see design spec
docs/superpowers/specs/2026-08-16-dnb-toc-uniform-ocr-design.md section 3.
render_pages_to_images is integration-tested against a real (synthetic,
blank) PDF via the real pdftoppm binary -- poppler is already a documented
project dependency (README.md). vision_extract_toc_entries is
tested with a mocked OpenAI-shaped client, no real network call."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from pypdf import PdfWriter

from dnb_toc_ground_truth.toc_entry import TocEntry
from dnb_toc_ground_truth.vision import (
    _MAX_VISION_PAGES,
    cache_path,
    load_cached_kind,
    load_cached_llm_entries,
    render_pages_to_images,
    versioned_cache_dir,
    vision_extract_toc_entries,
    write_cached_llm_entries,
)


def _make_pdf(path: Path, page_count: int) -> Path:
    writer = PdfWriter()
    for _ in range(page_count):
        writer.add_blank_page(width=200, height=200)
    with open(path, "wb") as f:
        writer.write(f)
    return path


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

    def test_records_duration_seconds_in_the_cache_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            entries = [TocEntry(title="Einleitung", printed_page_number=9, source_page_index=0)]
            write_cached_llm_entries(cache_dir, "book_dur", "model-a", entries, duration_seconds=4.5)
            data = json.loads(cache_path(cache_dir, "book_dur", "model-a").read_text(encoding="utf-8"))
            self.assertEqual(data["duration_seconds"], 4.5)
            # "duration_seconds" must sit right after "generated_at" -- insertion
            # order is preserved by json.dumps, and a caller reading the raw file
            # by eye relies on it being next to the other per-run metadata.
            keys = list(data.keys())
            self.assertEqual(keys.index("duration_seconds"), keys.index("generated_at") + 1)

    def test_duration_seconds_defaults_to_null_when_not_given(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            entries = [TocEntry(title="Einleitung", printed_page_number=9, source_page_index=0)]
            write_cached_llm_entries(cache_dir, "book_no_dur", "model-a", entries)
            data = json.loads(cache_path(cache_dir, "book_no_dur", "model-a").read_text(encoding="utf-8"))
            self.assertIsNone(data["duration_seconds"])

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

    def test_round_trip_preserves_skip(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            entries = [
                TocEntry(title="Einleitung", printed_page_number=9, source_page_index=0, skip=False),
                TocEntry(title="Bibliographie", printed_page_number=200, source_page_index=0, skip=True),
            ]
            write_cached_llm_entries(cache_dir, "book4", "model-a", entries)
            loaded = load_cached_llm_entries(cache_dir, "book4", "model-a")
            self.assertEqual(loaded, entries)
            self.assertFalse(loaded[0].skip)
            self.assertTrue(loaded[1].skip)

    def test_cache_files_live_under_the_versioned_subdirectory(self):
        # Regression test for the 2026-08-17 extraction-standard change:
        # cache_path/write_cached_llm_entries must write inside
        # versioned_cache_dir(cache_directory), not cache_directory itself,
        # so a schema bump can never be mistaken for the previous
        # version's (now-incomplete) cached results.
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            entries = [TocEntry(title="Einleitung", printed_page_number=9, source_page_index=0)]
            write_cached_llm_entries(cache_dir, "book5", "model-a", entries)
            self.assertTrue(cache_path(cache_dir, "book5", "model-a").exists())
            self.assertEqual(cache_path(cache_dir, "book5", "model-a").parent, versioned_cache_dir(cache_dir))
            self.assertFalse((cache_dir / "book5.model-a.json").exists())

    def test_an_older_schema_versions_leftover_file_is_never_read(self):
        # Simulates a pre-2026-08-17 cache file sitting directly in
        # cache_directory (the old, unversioned layout) -- it must be a
        # cache MISS under the current code, not silently trusted.
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            cache_dir.mkdir(exist_ok=True)
            (cache_dir / "book6.model-a.json").write_text(
                '{"generated_at": 0, "entries": [{"title": "Old", "printed_page_number": 1, '
                '"source_page_index": 0, "authors": [], "printed_roman": false}]}',
                encoding="utf-8",
            )
            self.assertIsNone(load_cached_llm_entries(cache_dir, "book6", "model-a"))

    def test_cache_path_sanitizes_a_model_id_containing_a_slash(self):
        # Regression test: an MPCDF-style model id is a real HuggingFace
        # repo id (e.g. "Qwen/Qwen2.5-VL-7B-Instruct"), which contains a
        # "/". cache_path must not let that "/" become a new path
        # separator -- the result must still be a single flat filename
        # living directly inside versioned_cache_dir, never a nested
        # subdirectory one level deeper.
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            path = cache_path(cache_dir, "book7", "Qwen/Qwen2.5-VL-7B-Instruct")
            self.assertEqual(path.parent, versioned_cache_dir(cache_dir))

    def test_write_and_load_round_trip_with_a_slash_containing_model_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            entries = [TocEntry(title="Einleitung", printed_page_number=9, source_page_index=0)]
            write_cached_llm_entries(cache_dir, "book8", "Qwen/Qwen2.5-VL-7B-Instruct", entries)
            path = cache_path(cache_dir, "book8", "Qwen/Qwen2.5-VL-7B-Instruct")
            self.assertTrue(path.is_file())
            self.assertEqual(path.parent, versioned_cache_dir(cache_dir))
            loaded = load_cached_llm_entries(cache_dir, "book8", "Qwen/Qwen2.5-VL-7B-Instruct")
            self.assertEqual(loaded, entries)


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


def _fake_response(response_text: str):
    message = MagicMock()
    message.content = response_text
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


def _fake_vision_client(response_text: str):
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=_fake_response(response_text))
    return client


def _fake_vision_client_sequence(*response_texts: str):
    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=[_fake_response(t) for t in response_texts])
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
            self.assertEqual(entries[0].printed_page_number, "9")
            self.assertEqual(entries[1].authors, ("Jane Author",))

    async def test_parses_skip_flag_per_entry(self):
        response = (
            '[{"title": "Einleitung", "authors": [], "printed_page_number": "9", "skip": false}, '
            '{"title": "Bibliographie", "authors": [], "printed_page_number": "200", "skip": true}]'
        )
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = _make_pdf(Path(tmp) / "book.pdf", 1)
            client = _fake_vision_client(response)
            entries = await vision_extract_toc_entries(pdf_path, "some-model", client)
            self.assertFalse(entries[0].skip)
            self.assertTrue(entries[1].skip)

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

    async def test_escalates_max_tokens_and_recovers_from_a_truncated_first_response(self):
        # A truncated JSON array (no closing "]") reliably fails
        # parse_json_array regardless of cause -- vision_extract_toc_entries
        # escalates max_tokens once and retries the SAME images/prompt
        # before giving up, mirroring segmentation.py's _extract_with_retry.
        truncated = '[{"title": "Einleitung", "authors": [], "printed_page_number": "9"}'
        complete = '[{"title": "Einleitung", "authors": [], "printed_page_number": "9"}]'
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = _make_pdf(Path(tmp) / "book.pdf", 1)
            client = _fake_vision_client_sequence(truncated, complete)
            entries = await vision_extract_toc_entries(pdf_path, "some-model", client)
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0].title, "Einleitung")
            self.assertEqual(client.chat.completions.create.await_count, 2)
            first_call_kwargs = client.chat.completions.create.call_args_list[0].kwargs
            second_call_kwargs = client.chat.completions.create.call_args_list[1].kwargs
            self.assertEqual(first_call_kwargs["max_tokens"], 4096)
            self.assertEqual(second_call_kwargs["max_tokens"], 8192)


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
