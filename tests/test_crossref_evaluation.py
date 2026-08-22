"""Unit tests for dnb_toc_ground_truth.crossref_evaluation -- measures
precision/recall/F1 between this corpus's own ground truth (or an
individual llm-cache model's raw extraction) and its committed Crossref
evaluation corpus (data/corpus/pilot/evaluation/*.expected.json), reusing
matching.diff_toc_entries unmodified. See design spec
docs/superpowers/specs/2026-08-22-crossref-evaluation-corpus-design.md."""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from pypdf import PdfWriter

from dnb_toc_ground_truth import corpus, vision
from dnb_toc_ground_truth.crossref_evaluation import (
    BookMetrics,
    backfill_model_cache,
    discover_cached_models,
    evaluate_book,
    evaluate_corpus,
    evaluate_model_corpus,
    _load_entries,
)
from dnb_toc_ground_truth.inference import ModelEndpoint
from dnb_toc_ground_truth.toc_entry import TocEntry


def _write_expected_json(key: str, entries: list[dict]) -> None:
    corpus.ground_truth_dir().mkdir(parents=True, exist_ok=True)
    corpus.expected_json_path(key).write_text(
        json.dumps({"entries": entries, "verified": True, "source": "agent_arbitration"}), encoding="utf-8",
    )


def _write_evaluation_json(key: str, entries: list[dict]) -> None:
    corpus.evaluation_dir().mkdir(parents=True, exist_ok=True)
    corpus.evaluation_json_path(key).write_text(
        json.dumps({"entries": entries, "source": "crossref", "fetched_at": ""}), encoding="utf-8",
    )


def _write_llm_cache_entry(key: str, model: str, entries: list[TocEntry]) -> None:
    vision.write_cached_llm_entries(corpus.llm_cache_dir(), key, model, entries)


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


class TestLoadEntries(unittest.TestCase):
    def test_builds_toc_entries_from_expected_json_shaped_file(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            _write_expected_json("9783899718188", [
                {"title": "Preface", "authors": [], "printed_page_number": "vii", "skip": True},
                {"title": "1. Introduction", "authors": ["Jane Author"], "printed_page_number": "1", "skip": False},
            ])
            entries = _load_entries(corpus.expected_json_path("9783899718188"))
            self.assertEqual(len(entries), 2)
            self.assertTrue(entries[0].skip)
            self.assertFalse(entries[1].skip)
            self.assertEqual(entries[1].authors, ("Jane Author",))


class TestEvaluateBook(unittest.TestCase):
    def test_perfect_match_scores_1_0_across_all_three_metrics(self):
        gt_entries = (
            TocEntry(title="Part I", printed_page_number="9", source_page_index=0, skip=True),
            TocEntry(title="1. Introduction", printed_page_number="11", source_page_index=0, skip=False),
        )
        crossref_entries = (TocEntry(title="Introduction", printed_page_number="11", source_page_index=-1, skip=False),)
        result = evaluate_book("9783899718188", gt_entries, crossref_entries)
        self.assertEqual(result.tp, 1)
        self.assertEqual(result.fp, 0)
        self.assertEqual(result.fn, 0)
        self.assertEqual(result.precision, 1.0)
        self.assertEqual(result.recall, 1.0)
        self.assertEqual(result.f1, 1.0)

    def test_a_gt_chapter_crossref_missed_lowers_recall_not_precision(self):
        gt_entries = (
            TocEntry(title="1. Introduction", printed_page_number="11", source_page_index=0, skip=False),
            TocEntry(title="2. Methods", printed_page_number="30", source_page_index=0, skip=False),
        )
        crossref_entries = (TocEntry(title="Introduction", printed_page_number="11", source_page_index=-1, skip=False),)
        result = evaluate_book("9783899718188", gt_entries, crossref_entries)
        self.assertEqual(result.tp, 1)
        self.assertEqual(result.fn, 1)
        self.assertEqual(result.fp, 0)
        self.assertEqual(result.precision, 1.0)
        self.assertEqual(result.recall, 0.5)
        self.assertAlmostEqual(result.f1, 2 / 3)

    def test_a_crossref_chapter_with_no_gt_match_lowers_precision_not_recall(self):
        gt_entries = (TocEntry(title="1. Introduction", printed_page_number="11", source_page_index=0, skip=False),)
        crossref_entries = (
            TocEntry(title="Introduction", printed_page_number="11", source_page_index=-1, skip=False),
            TocEntry(title="Unrelated Extra Chapter", printed_page_number="99", source_page_index=-1, skip=False),
        )
        result = evaluate_book("9783899718188", gt_entries, crossref_entries)
        self.assertEqual(result.tp, 1)
        self.assertEqual(result.fp, 1)
        self.assertEqual(result.fn, 0)
        self.assertEqual(result.precision, 0.5)
        self.assertEqual(result.recall, 1.0)

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
        crossref_entries = (
            TocEntry(title="Results", printed_page_number="40", source_page_index=-1, skip=False),
            TocEntry(title="Introduction", printed_page_number="1", source_page_index=-1, skip=False),
            TocEntry(title="Methods", printed_page_number="20", source_page_index=-1, skip=False),
        )
        result = evaluate_book("9783899718188", gt_entries, crossref_entries)
        self.assertEqual(result.tp, 3)
        self.assertEqual(result.precision, 1.0)
        self.assertEqual(result.recall, 1.0)

    def test_all_gt_entries_skipped_degrades_to_zero_without_crashing(self):
        gt_entries = (
            TocEntry(title="Part I", printed_page_number="9", source_page_index=0, skip=True),
            TocEntry(title="Part II", printed_page_number="50", source_page_index=0, skip=True),
        )
        crossref_entries = (TocEntry(title="Introduction", printed_page_number="11", source_page_index=-1, skip=False),)
        result = evaluate_book("9783899718188", gt_entries, crossref_entries)
        self.assertEqual(result.tp, 0)
        self.assertEqual(result.precision, 0.0)
        self.assertEqual(result.recall, 0.0)
        self.assertEqual(result.f1, 0.0)


class TestEvaluateCorpus(unittest.TestCase):
    def test_skips_books_with_no_evaluation_corpus_entry(self):
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

    def test_evaluates_book_with_matching_evaluation_corpus_entry(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            manifest_path = corpus.manifest_path()
            manifest_path.write_text(
                json.dumps({"toc_only": True, "books": [{"filename": "9783899718188.pdf", "doi": "10.1/x"}]}),
                encoding="utf-8",
            )
            _write_expected_json("9783899718188", [
                {"title": "1. Introduction", "authors": [], "printed_page_number": "1", "skip": False},
            ])
            _write_evaluation_json("9783899718188", [
                {"title": "Introduction", "authors": [], "printed_page_number": "1", "skip": False},
            ])
            results, no_coverage = evaluate_corpus()
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].key, "9783899718188")
            self.assertEqual(results[0].f1, 1.0)
            self.assertEqual(no_coverage, [])

    def test_ignores_ground_truth_only_book_without_evaluation_entry(self):
        # No evaluation/<key>.expected.json at all -- the book must not
        # appear in results, only in no_coverage.
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            manifest_path = corpus.manifest_path()
            manifest_path.write_text(
                json.dumps({"toc_only": True, "books": [
                    {"filename": "9783899718188.pdf", "doi": None},
                    {"filename": "9781234567897.pdf", "doi": None},
                ]}),
                encoding="utf-8",
            )
            _write_expected_json("9783899718188", [
                {"title": "1. Introduction", "authors": [], "printed_page_number": "1", "skip": False},
            ])
            # Second book has no ground truth at all -- must be skipped
            # entirely (not even counted as no_coverage), same as before.
            results, no_coverage = evaluate_corpus()
            self.assertEqual(results, [])
            self.assertEqual(no_coverage, ["9783899718188"])

    def test_evaluates_book_whose_manifest_key_is_a_hyphenated_isbn_variant(self):
        # write_evaluation_entry always writes under the NORMALIZED isbn
        # (both backfill_crossref.py and fetch_corpus.py normalize before
        # calling it), but a manifest filename stem -- the raw lookup key
        # -- could in principle be a hyphenated or differently-cased
        # variant of the same ISBN. evaluate_corpus must normalize before
        # looking up the evaluation file, or it silently misses
        # already-written data and misreports the book as uncovered.
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
            _write_evaluation_json("9783899718188", [
                {"title": "Introduction", "authors": [], "printed_page_number": "1", "skip": False},
            ])
            results, no_coverage = evaluate_corpus()
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].key, "978-3-89971-818-8")
            self.assertEqual(results[0].f1, 1.0)
            self.assertEqual(no_coverage, [])


class TestEvaluateModelCorpus(unittest.TestCase):
    def test_scores_model_cache_entries_against_crossref_sample(self):
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
                TocEntry(title="1. Introduction", printed_page_number="1", source_page_index=-1, skip=False),
            ])
            results, no_cache = evaluate_model_corpus("some/model")
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].key, "9783899718188")
            self.assertEqual(results[0].f1, 1.0)
            self.assertEqual(no_cache, [])

    def test_book_with_no_cache_entry_for_model_is_reported_not_dropped(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            manifest_path = corpus.manifest_path()
            manifest_path.write_text(
                json.dumps({"toc_only": True, "books": [{"filename": "9783899718188.pdf", "doi": "10.1/x"}]}),
                encoding="utf-8",
            )
            _write_evaluation_json("9783899718188", [
                {"title": "Introduction", "authors": [], "printed_page_number": "1", "skip": False},
            ])
            results, no_cache = evaluate_model_corpus("some/model")
            self.assertEqual(results, [])
            self.assertEqual(no_cache, ["9783899718188"])

    def test_ignores_books_without_a_crossref_evaluation_entry(self):
        # A cache entry alone (no evaluation-corpus file) isn't a
        # meaningful comparison -- must not appear in results OR
        # no_cache, mirroring evaluate_corpus's own ground-truth-only-book
        # handling.
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            manifest_path = corpus.manifest_path()
            manifest_path.write_text(
                json.dumps({"toc_only": True, "books": [{"filename": "9783899718188.pdf", "doi": None}]}),
                encoding="utf-8",
            )
            _write_llm_cache_entry("9783899718188", "some/model", [
                TocEntry(title="1. Introduction", printed_page_number="1", source_page_index=-1, skip=False),
            ])
            results, no_cache = evaluate_model_corpus("some/model")
            self.assertEqual(results, [])
            self.assertEqual(no_cache, [])


class TestDiscoverCachedModels(unittest.TestCase):
    def test_finds_only_models_cached_for_crossref_sample_books(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            manifest_path = corpus.manifest_path()
            manifest_path.write_text(
                json.dumps({"toc_only": True, "books": [
                    {"filename": "9783899718188.pdf", "doi": "10.1/x"},
                    {"filename": "9781234567897.pdf", "doi": None},
                ]}),
                encoding="utf-8",
            )
            _write_evaluation_json("9783899718188", [
                {"title": "Introduction", "authors": [], "printed_page_number": "1", "skip": False},
            ])
            _write_llm_cache_entry("9783899718188", "model/a", [
                TocEntry(title="Introduction", printed_page_number="1", source_page_index=-1, skip=False),
            ])
            _write_llm_cache_entry("9783899718188", "model/b", [
                TocEntry(title="Introduction", printed_page_number="1", source_page_index=-1, skip=False),
            ])
            # Cached for a book with no crossref evaluation entry -- must
            # not be discovered.
            _write_llm_cache_entry("9781234567897", "model/only-uncovered-book", [
                TocEntry(title="Introduction", printed_page_number="1", source_page_index=-1, skip=False),
            ])
            models = discover_cached_models()
            self.assertEqual(models, ["model__a", "model__b"])


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

    def test_an_empty_extraction_result_is_reported_as_failed_and_not_cached(self):
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
            client = _fake_client("[]")
            endpoint = ModelEndpoint(label="some/model", model_id="some/model", kind="vision", client=client)

            succeeded, failed = asyncio.run(backfill_model_cache("some/model", endpoint, corpus.llm_cache_dir()))

            self.assertEqual(succeeded, [])
            self.assertEqual(failed, ["9783899718188"])
            cached = vision.load_cached_llm_entries(corpus.llm_cache_dir(), "9783899718188", "some/model")
            self.assertIsNone(cached)

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


if __name__ == "__main__":
    unittest.main()
