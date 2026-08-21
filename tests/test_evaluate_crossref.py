"""Unit tests for cli/evaluate_crossref.py -- measures agreement between
this corpus's own ground truth and cached Crossref chapter data, reusing
matching.diff_toc_entries unmodified. See design spec
docs/superpowers/specs/2026-08-21-crossref-cross-validation-design.md."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dnb_toc_ground_truth import corpus
from dnb_toc_ground_truth.crossref import CrossrefBookData
from dnb_toc_ground_truth.toc_entry import TocEntry

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cli"))

from evaluate_crossref import BookAgreement, evaluate_book, evaluate_corpus, _load_crossref_data, _load_gt_entries


def _write_expected_json(key: str, entries: list[dict]) -> None:
    corpus.ground_truth_dir().mkdir(parents=True, exist_ok=True)
    corpus.expected_json_path(key).write_text(
        json.dumps({"entries": entries, "verified": True, "source": "agent_arbitration"}), encoding="utf-8",
    )


class TestLoadGtEntries(unittest.TestCase):
    def test_builds_toc_entries_from_expected_json(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            _write_expected_json("9783899718188", [
                {"title": "Preface", "authors": [], "printed_page_number": "vii", "skip": True},
                {"title": "1. Introduction", "authors": ["Jane Author"], "printed_page_number": "1", "skip": False},
            ])
            entries = _load_gt_entries("9783899718188")
            self.assertEqual(len(entries), 2)
            self.assertTrue(entries[0].skip)
            self.assertFalse(entries[1].skip)
            self.assertEqual(entries[1].authors, ("Jane Author",))


class TestEvaluateBook(unittest.TestCase):
    def test_filters_skip_entries_before_comparing(self):
        gt_entries = (
            TocEntry(title="Part I", printed_page_number="9", source_page_index=0, skip=True),
            TocEntry(title="1. Introduction", printed_page_number="11", source_page_index=0, skip=False),
        )
        crossref_data = CrossrefBookData(
            isbn="9783899718188", doi="10.1/x", fetched_at="",
            chapters=(TocEntry(title="Introduction", printed_page_number="11", source_page_index=-1, skip=False),),
        )
        result = evaluate_book("9783899718188", gt_entries, crossref_data)
        self.assertEqual(result.matched, 1)
        self.assertEqual(result.only_in_gt, 0)
        self.assertEqual(result.only_in_crossref, 0)
        self.assertEqual(result.agreement_rate, 1.0)

    def test_reports_disagreement(self):
        gt_entries = (
            TocEntry(title="1. Introduction", printed_page_number="11", source_page_index=0, skip=False),
            TocEntry(title="2. Methods", printed_page_number="30", source_page_index=0, skip=False),
        )
        crossref_data = CrossrefBookData(
            isbn="9783899718188", doi="10.1/x", fetched_at="",
            chapters=(TocEntry(title="Introduction", printed_page_number="11", source_page_index=-1, skip=False),),
        )
        result = evaluate_book("9783899718188", gt_entries, crossref_data)
        self.assertEqual(result.matched, 1)
        self.assertEqual(result.only_in_gt, 1)
        self.assertEqual(result.only_in_crossref, 0)
        self.assertEqual(result.agreement_rate, 0.5)

    def test_matches_correctly_when_crossref_chapters_are_not_in_page_order(self):
        # Crossref's /works response order is registration order, not
        # printed page order -- unlike two independent TOC-page reads,
        # which diff_toc_entries' greedy alignment assumes are already
        # in the same (page) order. Found empirically (2026-08-21,
        # isbn:9783111702681): an out-of-order chapter list took a
        # 20-chapter book's real match count from 20 down to 5 before
        # evaluate_book started sorting both sides by page first.
        gt_entries = (
            TocEntry(title="Introduction", printed_page_number="1", source_page_index=0, skip=False),
            TocEntry(title="Methods", printed_page_number="20", source_page_index=0, skip=False),
            TocEntry(title="Results", printed_page_number="40", source_page_index=0, skip=False),
        )
        crossref_data = CrossrefBookData(
            isbn="9783899718188", doi="10.1/x", fetched_at="",
            chapters=(
                TocEntry(title="Results", printed_page_number="40", source_page_index=-1, skip=False),
                TocEntry(title="Introduction", printed_page_number="1", source_page_index=-1, skip=False),
                TocEntry(title="Methods", printed_page_number="20", source_page_index=-1, skip=False),
            ),
        )
        result = evaluate_book("9783899718188", gt_entries, crossref_data)
        self.assertEqual(result.matched, 3)
        self.assertEqual(result.only_in_gt, 0)
        self.assertEqual(result.only_in_crossref, 0)
        self.assertEqual(result.agreement_rate, 1.0)

    def test_all_gt_entries_skipped_degrades_to_zero_agreement_without_crashing(self):
        gt_entries = (
            TocEntry(title="Part I", printed_page_number="9", source_page_index=0, skip=True),
            TocEntry(title="Part II", printed_page_number="50", source_page_index=0, skip=True),
        )
        crossref_data = CrossrefBookData(
            isbn="9783899718188", doi="10.1/x", fetched_at="",
            chapters=(TocEntry(title="Introduction", printed_page_number="11", source_page_index=-1, skip=False),),
        )
        result = evaluate_book("9783899718188", gt_entries, crossref_data)
        self.assertEqual(result.matched, 0)
        self.assertEqual(result.only_in_gt, 0)
        self.assertEqual(result.only_in_crossref, 1)
        self.assertEqual(result.agreement_rate, 0.0)


class TestEvaluateCorpus(unittest.TestCase):
    def test_skips_books_with_no_cached_crossref_data(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            manifest_path = corpus.manifest_path()
            manifest_path.write_text(
                json.dumps({"toc_only": True, "books": [{"filename": "9783899718188.pdf", "doi": None}]}),
                encoding="utf-8",
            )
            _write_expected_json("9783899718188", [
                {"title": "1. Introduction", "authors": [], "printed_page_number": "1", "skip": False},
            ])
            results, no_coverage = evaluate_corpus()
            self.assertEqual(results, [])
            self.assertEqual(no_coverage, ["9783899718188"])

    def test_evaluates_book_with_cached_crossref_data(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            manifest_path = corpus.manifest_path()
            manifest_path.write_text(
                json.dumps({"toc_only": True, "books": [{"filename": "9783899718188.pdf", "doi": "10.1/x"}]}),
                encoding="utf-8",
            )
            _write_expected_json("9783899718188", [
                {"title": "1. Introduction", "authors": [], "printed_page_number": "1", "skip": False},
            ])
            corpus.crossref_cache_dir().mkdir(parents=True, exist_ok=True)
            (corpus.crossref_cache_dir() / "9783899718188.crossref.json").write_text(
                json.dumps({
                    "isbn": "9783899718188", "doi": "10.1/x", "fetched_at": "",
                    "chapters": [{"title": "Introduction", "authors": [], "printed_page_number": "1"}],
                }),
                encoding="utf-8",
            )
            results, no_coverage = evaluate_corpus()
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].key, "9783899718188")
            self.assertEqual(results[0].agreement_rate, 1.0)
            self.assertEqual(no_coverage, [])

    def test_finds_cached_crossref_data_when_lookup_key_is_a_hyphenated_isbn_variant(self):
        # fetch_crossref_book() always writes the cache file keyed by the
        # NORMALIZED isbn (crossref.normalize_isbn), but a manifest filename
        # stem -- the raw lookup key -- could in principle be a hyphenated
        # or differently-cased variant of the same ISBN. _load_crossref_data
        # must normalize before looking up the cache, or it silently misses
        # already-cached data and misreports the book as having no coverage.
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            corpus.crossref_cache_dir().mkdir(parents=True, exist_ok=True)
            (corpus.crossref_cache_dir() / "9783899718188.crossref.json").write_text(
                json.dumps({
                    "isbn": "9783899718188", "doi": "10.1/x", "fetched_at": "",
                    "chapters": [{"title": "Introduction", "authors": [], "printed_page_number": "1"}],
                }),
                encoding="utf-8",
            )
            data = _load_crossref_data("978-3-89971-818-8")
            self.assertIsNotNone(data)
            self.assertEqual(len(data.chapters), 1)

    def test_evaluates_book_whose_manifest_key_is_a_hyphenated_isbn_variant(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            manifest_path = corpus.manifest_path()
            manifest_path.write_text(
                json.dumps({
                    "toc_only": True,
                    "books": [{"filename": "978-3-89971-818-8.pdf", "doi": "10.1/x"}],
                }),
                encoding="utf-8",
            )
            _write_expected_json("978-3-89971-818-8", [
                {"title": "1. Introduction", "authors": [], "printed_page_number": "1", "skip": False},
            ])
            corpus.crossref_cache_dir().mkdir(parents=True, exist_ok=True)
            (corpus.crossref_cache_dir() / "9783899718188.crossref.json").write_text(
                json.dumps({
                    "isbn": "9783899718188", "doi": "10.1/x", "fetched_at": "",
                    "chapters": [{"title": "Introduction", "authors": [], "printed_page_number": "1"}],
                }),
                encoding="utf-8",
            )
            results, no_coverage = evaluate_corpus()
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].key, "978-3-89971-818-8")
            self.assertEqual(results[0].agreement_rate, 1.0)
            self.assertEqual(no_coverage, [])
