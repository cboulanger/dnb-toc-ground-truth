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
from unittest.mock import Mock, patch

import httpx

from dnb_toc_ground_truth.crossref import CrossrefBookData, fetch_crossref_book, normalize_isbn, write_evaluation_entry
from dnb_toc_ground_truth.toc_entry import TocEntry


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

    def test_strips_page_number_glued_onto_title(self):
        # Real-world artifact found in some publishers' Crossref
        # registrations (e.g. isbn:9783111702681): the printed page
        # number is glued directly onto the front of the title text with
        # no separator.
        client = Mock()
        client.get.return_value = _json_response({
            "message": {"items": [
                {"type": "book-chapter", "DOI": "10.1/x", "title": ["49Strategies for Responding"], "author": [], "page": "49-68"},
            ]}
        })
        with tempfile.TemporaryDirectory() as tmp:
            data = fetch_crossref_book("9783899718188", client, None, Path(tmp))
        self.assertEqual(data.chapters[0].title, "Strategies for Responding")
        self.assertEqual(data.chapters[0].printed_page_number, "49")

    def test_does_not_strip_a_number_that_is_not_the_page_prefix(self):
        client = Mock()
        client.get.return_value = _json_response({
            "message": {"items": [
                {"type": "book-chapter", "DOI": "10.1/x", "title": ["1984: A Retrospective"], "author": [], "page": "49-68"},
            ]}
        })
        with tempfile.TemporaryDirectory() as tmp:
            data = fetch_crossref_book("9783899718188", client, None, Path(tmp))
        self.assertEqual(data.chapters[0].title, "1984: A Retrospective")

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

    def test_confirmed_empty_response_is_cached(self):
        client = Mock()
        client.get.return_value = _json_response({"message": {"items": []}})
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            data = fetch_crossref_book("9783899718188", client, None, cache_dir)

            self.assertIsNone(data.doi)
            self.assertEqual(data.chapters, ())
            cache_path = cache_dir / "9783899718188.crossref.json"
            self.assertTrue(cache_path.exists())
            cached_payload = json.loads(cache_path.read_text(encoding="utf-8"))
            self.assertTrue(cached_payload["fetched_at"])

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

    def test_cache_write_failure_still_returns_correct_data(self):
        # "never raises" must hold for cache-write failures too (a full
        # disk, a permissions problem, a bad --crossref-cache-dir), not
        # just network/HTTP/JSON failures -- see _save_cache's docstring.
        client = Mock()
        client.get.return_value = _json_response(_MIXED_TYPE_RESPONSE)
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            with patch.object(Path, "write_text", side_effect=OSError("disk full")):
                data = fetch_crossref_book("9783899718188", client, None, cache_dir)
        self.assertEqual(data.doi, "10.1515/book-doi")
        self.assertEqual(len(data.chapters), 2)
        self.assertFalse((cache_dir / "9783899718188.crossref.json").exists())


class TestWriteEvaluationEntry(unittest.TestCase):
    def _data(self, chapters):
        return CrossrefBookData(isbn="9783899718188", doi="10.1/x", chapters=tuple(chapters), fetched_at="2026-08-22T00:00:00")

    def test_writes_file_when_enough_paged_chapters(self):
        chapters = [
            TocEntry(title="A", printed_page_number="1", source_page_index=-1, skip=False),
            TocEntry(title="B", printed_page_number="10", source_page_index=-1, skip=False),
            TocEntry(title="C", printed_page_number="20", source_page_index=-1, skip=False),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            eval_dir = Path(tmp)
            written = write_evaluation_entry("9783899718188", self._data(chapters), eval_dir, min_chapters=3)

            self.assertTrue(written)
            path = eval_dir / "9783899718188.expected.json"
            self.assertTrue(path.exists())
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["source"], "crossref")
            self.assertEqual(len(payload["entries"]), 3)
            self.assertEqual(payload["entries"][0], {"title": "A", "authors": [], "printed_page_number": "1", "skip": False})

    def test_skips_page_less_chapters_when_counting_and_writing(self):
        chapters = [
            TocEntry(title="A", printed_page_number="1", source_page_index=-1, skip=False),
            TocEntry(title="B", printed_page_number="10", source_page_index=-1, skip=False),
            TocEntry(title="C", printed_page_number="20", source_page_index=-1, skip=False),
            TocEntry(title="Index", printed_page_number=None, source_page_index=-1, skip=False),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            eval_dir = Path(tmp)
            write_evaluation_entry("9783899718188", self._data(chapters), eval_dir, min_chapters=3)
            payload = json.loads((eval_dir / "9783899718188.expected.json").read_text(encoding="utf-8"))
        self.assertEqual({e["title"] for e in payload["entries"]}, {"A", "B", "C"})

    def test_does_not_write_below_min_chapters(self):
        chapters = [
            TocEntry(title="A", printed_page_number="1", source_page_index=-1, skip=False),
            TocEntry(title="B", printed_page_number="10", source_page_index=-1, skip=False),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            eval_dir = Path(tmp)
            written = write_evaluation_entry("9783899718188", self._data(chapters), eval_dir, min_chapters=3)
        self.assertFalse(written)
        self.assertFalse((eval_dir / "9783899718188.expected.json").exists())

    def test_min_chapters_is_configurable(self):
        chapters = [
            TocEntry(title="A", printed_page_number="1", source_page_index=-1, skip=False),
            TocEntry(title="B", printed_page_number="10", source_page_index=-1, skip=False),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            eval_dir = Path(tmp)
            written = write_evaluation_entry("9783899718188", self._data(chapters), eval_dir, min_chapters=2)
        self.assertTrue(written)

    def test_write_failure_returns_false_without_raising(self):
        chapters = [
            TocEntry(title="A", printed_page_number="1", source_page_index=-1, skip=False),
            TocEntry(title="B", printed_page_number="10", source_page_index=-1, skip=False),
            TocEntry(title="C", printed_page_number="20", source_page_index=-1, skip=False),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            eval_dir = Path(tmp)
            with patch.object(Path, "write_text", side_effect=OSError("disk full")):
                written = write_evaluation_entry("9783899718188", self._data(chapters), eval_dir, min_chapters=3)
        self.assertFalse(written)
