"""Unit tests for cli/arbitrate.py's pure logic --
see design spec docs/superpowers/specs/2026-08-16-dnb-toc-arbitration-design.md.
No real corpus paths, no network -- all fixtures live in tempdirs, with
dnb_toc_ground_truth.corpus.CORPUS_DIR patched per test so the module's
path helpers (which read the module-level constant fresh at call time)
resolve into isolated per-test directories."""

import json
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

from dnb_toc_ground_truth import corpus
from dnb_toc_ground_truth.toc_entry import TocEntry
from dnb_toc_ground_truth.vision import write_cached_llm_entries

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cli"))

from arbitrate import (
    _cached_kinds_for_book,
    _cached_models_for_book,
    books_needing_arbitration,
    format_book_report,
    reject_book,
)


def _entry(title: str, page: int) -> TocEntry:
    return TocEntry(title=title, printed_page_number=page, source_page_index=0)


class TestBooksNeedingArbitration(unittest.TestCase):
    def test_includes_a_book_with_cache_and_no_expected_json(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp) / "corpus"):
            cache_directory = Path(tmp) / "cache"
            corpus.corpus_dir().mkdir(parents=True)
            write_cached_llm_entries(cache_directory, "book1", "model-a", [_entry("X", 1)])

            self.assertEqual(books_needing_arbitration(cache_directory), ["book1"])

    def test_excludes_a_book_that_already_passed_the_gate(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp) / "corpus"):
            cache_directory = Path(tmp) / "cache"
            corpus.ground_truth_dir().mkdir(parents=True)
            write_cached_llm_entries(cache_directory, "book2", "model-a", [_entry("X", 1)])
            corpus.expected_json_path("book2").write_text('{"entries": [], "verified": false}', encoding="utf-8")

            self.assertEqual(books_needing_arbitration(cache_directory), [])

    def test_excludes_an_already_rejected_book(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp) / "corpus"):
            cache_directory = Path(tmp) / "cache"
            corpus.corpus_dir().mkdir(parents=True)
            write_cached_llm_entries(cache_directory, "book3", "model-a", [_entry("X", 1)])
            corpus.arbitration_rejected_path().write_text(
                json.dumps({"rejected": [{"key": "book3", "reason": "unrecoverable", "rejected_at": "2026-08-16"}]}),
                encoding="utf-8",
            )

            self.assertEqual(books_needing_arbitration(cache_directory), [])


class TestCachedModelsForBook(unittest.TestCase):
    def test_loads_every_cached_model_for_a_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_directory = Path(tmp)
            write_cached_llm_entries(cache_directory, "book1", "model-a", [_entry("X", 1)])
            write_cached_llm_entries(cache_directory, "book1", "model-b", [_entry("Y", 2)])
            write_cached_llm_entries(cache_directory, "book2", "model-a", [_entry("Z", 3)])

            result = _cached_models_for_book(cache_directory, "book1")

            self.assertEqual(set(result), {"model-a", "model-b"})
            self.assertEqual(result["model-a"][0].title, "X")

    def test_handles_a_model_id_that_itself_contains_a_dot(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_directory = Path(tmp)
            write_cached_llm_entries(cache_directory, "book1", "qwen3.6-27b", [_entry("X", 1)])

            result = _cached_models_for_book(cache_directory, "book1")

            self.assertEqual(set(result), {"qwen3.6-27b"})


class TestFormatBookReport(unittest.TestCase):
    def test_two_model_disagreement_lists_unmatched_entries_from_each_side(self):
        report = format_book_report(
            "book1", "Some Title", Path("/tmp/book1.pdf"),
            {
                "model-a": [_entry("Einleitung", 9), _entry("Only in A", 20)],
                "model-b": [_entry("Einleitung", 9), _entry("Only in B", 30)],
            },
        )
        self.assertIn("Only in A", report)
        self.assertIn("Only in B", report)
        self.assertIn("model-a", report)
        self.assertIn("model-b", report)
        self.assertIn("Some Title", report)

    def test_single_surviving_model_lists_its_entries_with_a_note(self):
        report = format_book_report(
            "book2", "Some Title", Path("/tmp/book2.pdf"),
            {"model-a": [_entry("Einleitung", 9)]},
        )
        self.assertIn("Only vision: model-a returned usable output", report)
        self.assertIn("Einleitung", report)

    def test_more_than_two_models_falls_back_to_a_plain_per_model_listing(self):
        report = format_book_report(
            "book3", "Some Title", Path("/tmp/book3.pdf"),
            {
                "model-a": [_entry("From A", 1)],
                "model-b": [_entry("From B", 2)],
                "model-c": [_entry("From C", 3)],
            },
        )
        self.assertIn("found 3", report)
        self.assertIn("model-a", report)
        self.assertIn("model-b", report)
        self.assertIn("model-c", report)
        self.assertIn("From A", report)
        self.assertIn("From B", report)
        self.assertIn("From C", report)

    def test_unknown_page_number_renders_as_question_mark(self):
        report = format_book_report(
            "book4", "Some Title", Path("/tmp/book4.pdf"),
            {"model-a": [TocEntry(title="Mystery", printed_page_number=None, source_page_index=0)]},
        )
        self.assertIn("p.   ?", report)


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

    def test_an_unrecognized_kind_surfaces_as_itself_not_silently_as_text(self):
        report = format_book_report(
            "book7", "Some Title", Path("/tmp/book7.pdf"),
            {"model-a": [_entry("Einleitung", 9)]},
            {"model-a": "something-unexpected"},
        )
        self.assertIn("Only something-unexpected: model-a returned usable output", report)
        self.assertNotIn("text (OCR'd)", report)


class TestCachedKindsForBook(unittest.TestCase):
    def test_reads_each_models_own_kind(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_directory = Path(tmp)
            write_cached_llm_entries(cache_directory, "book1", "model-a", [_entry("X", 1)], kind="vision")
            write_cached_llm_entries(cache_directory, "book1", "model-b", [_entry("Y", 2)], kind="text")

            result = _cached_kinds_for_book(cache_directory, "book1")

            self.assertEqual(result, {"model-a": "vision", "model-b": "text"})


class TestRejectBook(unittest.TestCase):
    def test_creates_the_file_on_first_rejection(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            result = reject_book("book1", "unrecoverable", today=lambda: date(2026, 8, 16))
            self.assertEqual(result, 0)
            data = json.loads(corpus.arbitration_rejected_path().read_text(encoding="utf-8"))
            self.assertEqual(data["rejected"], [{"key": "book1", "reason": "unrecoverable", "rejected_at": "2026-08-16"}])

    def test_appends_to_an_existing_rejection_list(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            reject_book("book1", "reason one", today=lambda: date(2026, 8, 16))
            reject_book("book2", "reason two", today=lambda: date(2026, 8, 16))
            data = json.loads(corpus.arbitration_rejected_path().read_text(encoding="utf-8"))
            self.assertEqual([entry["key"] for entry in data["rejected"]], ["book1", "book2"])

    def test_errors_on_a_duplicate_key_without_overwriting(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            reject_book("book1", "original reason", today=lambda: date(2026, 8, 16))
            result = reject_book("book1", "different reason", today=lambda: date(2026, 8, 17))
            self.assertEqual(result, 1)
            data = json.loads(corpus.arbitration_rejected_path().read_text(encoding="utf-8"))
            self.assertEqual(len(data["rejected"]), 1)
            self.assertEqual(data["rejected"][0]["reason"], "original reason")
