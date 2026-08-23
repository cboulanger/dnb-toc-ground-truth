"""Unit tests for cli/skip_list.py's pure logic. No real corpus paths --
all fixtures live in tempdirs, with dnb_toc_ground_truth.corpus.CORPUS_DIR
patched per test so the module's path helpers (which read the
module-level constant fresh at call time) resolve into isolated per-test
directories."""

import json
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

from dnb_toc_ground_truth import corpus

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cli"))

from skip_list import add_skip, load_skip_set, remove_skip


class TestAddSkip(unittest.TestCase):
    def test_creates_the_file_on_first_skip(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            result = add_skip("book1", "model-a", "hangs", today=lambda: date(2026, 8, 23))
            self.assertEqual(result, 0)
            data = json.loads(corpus.model_skip_list_path().read_text(encoding="utf-8"))
            self.assertEqual(
                data["skipped"],
                [{"key": "book1", "model": "model-a", "reason": "hangs", "skipped_at": "2026-08-23"}],
            )

    def test_appends_to_an_existing_skip_list(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            add_skip("book1", "model-a", "reason one", today=lambda: date(2026, 8, 23))
            add_skip("book2", "model-b", "reason two", today=lambda: date(2026, 8, 23))
            data = json.loads(corpus.model_skip_list_path().read_text(encoding="utf-8"))
            self.assertEqual([(e["key"], e["model"]) for e in data["skipped"]], [("book1", "model-a"), ("book2", "model-b")])

    def test_errors_on_a_duplicate_pair_without_overwriting(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            add_skip("book1", "model-a", "original reason", today=lambda: date(2026, 8, 23))
            result = add_skip("book1", "model-a", "different reason", today=lambda: date(2026, 8, 24))
            self.assertEqual(result, 1)
            data = json.loads(corpus.model_skip_list_path().read_text(encoding="utf-8"))
            self.assertEqual(len(data["skipped"]), 1)
            self.assertEqual(data["skipped"][0]["reason"], "original reason")

    def test_the_same_key_with_a_different_model_is_a_distinct_pair(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            add_skip("book1", "model-a", "reason a", today=lambda: date(2026, 8, 23))
            result = add_skip("book1", "model-b", "reason b", today=lambda: date(2026, 8, 23))
            self.assertEqual(result, 0)
            data = json.loads(corpus.model_skip_list_path().read_text(encoding="utf-8"))
            self.assertEqual(len(data["skipped"]), 2)


class TestRemoveSkip(unittest.TestCase):
    def test_removes_a_matching_pair(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            add_skip("book1", "model-a", "reason", today=lambda: date(2026, 8, 23))
            result = remove_skip("book1", "model-a")
            self.assertEqual(result, 0)
            data = json.loads(corpus.model_skip_list_path().read_text(encoding="utf-8"))
            self.assertEqual(data["skipped"], [])

    def test_errors_when_the_pair_is_not_present(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            result = remove_skip("book1", "model-a")
            self.assertEqual(result, 1)

    def test_leaves_other_pairs_for_the_same_key_untouched(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            add_skip("book1", "model-a", "reason a", today=lambda: date(2026, 8, 23))
            add_skip("book1", "model-b", "reason b", today=lambda: date(2026, 8, 23))
            remove_skip("book1", "model-a")
            data = json.loads(corpus.model_skip_list_path().read_text(encoding="utf-8"))
            self.assertEqual([(e["key"], e["model"]) for e in data["skipped"]], [("book1", "model-b")])


class TestLoadSkipSet(unittest.TestCase):
    def test_empty_when_no_file_exists(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            self.assertEqual(load_skip_set(), set())

    def test_returns_every_pair_as_a_tuple(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            add_skip("book1", "model-a", "reason", today=lambda: date(2026, 8, 23))
            add_skip("book2", "model-b", "reason", today=lambda: date(2026, 8, 23))
            self.assertEqual(load_skip_set(), {("book1", "model-a"), ("book2", "model-b")})


if __name__ == "__main__":
    unittest.main()
