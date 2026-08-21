"""Unit tests for cli/backfill_crossref.py -- backfills Crossref DOI +
chapter data for existing manifest entries that already have
.expected.json but no doi yet. See design spec
docs/superpowers/specs/2026-08-21-crossref-cross-validation-design.md."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from dnb_toc_ground_truth import corpus

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cli"))

from backfill_crossref import _needs_backfill, backfill


def _json_response(payload: dict) -> Mock:
    response = Mock()
    response.status_code = 200
    response.json.return_value = payload
    response.raise_for_status = Mock()
    response.headers = {}
    return response


class TestNeedsBackfill(unittest.TestCase):
    def test_true_when_expected_json_exists_and_no_doi(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            corpus.ground_truth_dir().mkdir(parents=True)
            corpus.expected_json_path("9783899718188").write_text("{}", encoding="utf-8")
            self.assertTrue(_needs_backfill({"filename": "9783899718188.pdf", "doi": None}))

    def test_false_when_doi_already_present(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            corpus.ground_truth_dir().mkdir(parents=True)
            corpus.expected_json_path("9783899718188").write_text("{}", encoding="utf-8")
            self.assertFalse(_needs_backfill({"filename": "9783899718188.pdf", "doi": "10.1/x"}))

    def test_false_when_no_expected_json(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            corpus.ground_truth_dir().mkdir(parents=True)
            self.assertFalse(_needs_backfill({"filename": "9783899718188.pdf", "doi": None}))


class TestBackfill(unittest.TestCase):
    def test_writes_doi_for_eligible_book_and_caches_chapters(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            corpus.ground_truth_dir().mkdir(parents=True)
            corpus.expected_json_path("9783899718188").write_text('{"entries": []}', encoding="utf-8")
            manifest_path = corpus.manifest_path()
            manifest_path.write_text(
                json.dumps({"toc_only": True, "books": [
                    {"filename": "9783899718188.pdf", "doi": None},
                ]}),
                encoding="utf-8",
            )
            client = Mock()
            client.get.return_value = _json_response({
                "message": {"items": [
                    {"type": "book", "DOI": "10.1515/found", "title": ["X"]},
                    {"type": "book-chapter", "DOI": "10.1515/ch1", "title": ["A Chapter"], "author": [], "page": "1-10"},
                ]}
            })

            checked, found, cached = backfill(manifest_path, client, None, corpus.crossref_cache_dir(), force=False)

            self.assertEqual(checked, 1)
            self.assertEqual(found, 1)
            self.assertEqual(cached, 1)
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(data["books"][0]["doi"], "10.1515/found")

    def test_skips_book_without_expected_json(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            corpus.ground_truth_dir().mkdir(parents=True)
            manifest_path = corpus.manifest_path()
            manifest_path.write_text(
                json.dumps({"toc_only": True, "books": [{"filename": "9783899718188.pdf", "doi": None}]}),
                encoding="utf-8",
            )
            client = Mock()

            checked, found, cached = backfill(manifest_path, client, None, corpus.crossref_cache_dir(), force=False)

            self.assertEqual(checked, 0)
            client.get.assert_not_called()

    def test_skips_book_that_already_has_doi(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            corpus.ground_truth_dir().mkdir(parents=True)
            corpus.expected_json_path("9783899718188").write_text('{"entries": []}', encoding="utf-8")
            manifest_path = corpus.manifest_path()
            manifest_path.write_text(
                json.dumps({"toc_only": True, "books": [
                    {"filename": "9783899718188.pdf", "doi": "10.1/already"},
                ]}),
                encoding="utf-8",
            )
            client = Mock()

            checked, found, cached = backfill(manifest_path, client, None, corpus.crossref_cache_dir(), force=False)

            self.assertEqual(checked, 0)
            client.get.assert_not_called()

    def test_manifest_untouched_when_no_doi_found(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            corpus.ground_truth_dir().mkdir(parents=True)
            corpus.expected_json_path("9783899718188").write_text('{"entries": []}', encoding="utf-8")
            manifest_path = corpus.manifest_path()
            original = json.dumps({"toc_only": True, "books": [
                {"filename": "9783899718188.pdf", "doi": None},
            ]})
            manifest_path.write_text(original, encoding="utf-8")
            client = Mock()
            client.get.return_value = _json_response({"message": {"items": []}})

            checked, found, cached = backfill(manifest_path, client, None, corpus.crossref_cache_dir(), force=False)

            self.assertEqual(found, 0)
            self.assertEqual(cached, 0)
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertIsNone(data["books"][0]["doi"])
