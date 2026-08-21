"""Unit tests for cli/select_eval_sample.py's pure
stratification logic (design spec
docs/superpowers/specs/2026-08-15-dnb-toc-ground-truth-generation-design.md
section 5). The real file-walking main() is exercised manually against
the real corpus."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cli"))

from dnb_toc_ground_truth.corpus import manifest_key
from select_eval_sample import _decade, stratify_sample


class TestManifestKey(unittest.TestCase):
    def test_strips_pdf_extension(self):
        self.assertEqual(manifest_key({"filename": "9783899718188.pdf"}), "9783899718188")


class TestDecade(unittest.TestCase):
    def test_parses_start_date(self):
        self.assertEqual(_decade({"publication": [{"startDate": "2002"}]}), "2000s")

    def test_falls_back_to_unknown_when_absent(self):
        self.assertEqual(_decade({}), "unknown")

    def test_falls_back_to_unknown_when_unparseable(self):
        self.assertEqual(_decade({"publication": [{"startDate": "n.d."}]}), "unknown")


class TestStratifySample(unittest.TestCase):
    def _books(self, n: int, language: str = "de", prefix: str = "book") -> list[dict]:
        return [{"filename": f"{prefix}{i}.pdf", "language": language} for i in range(n)]

    def test_returns_requested_size_when_pool_is_large_enough(self):
        books = self._books(100)
        records = {f"book{i}": {"publication": [{"startDate": "2010"}]} for i in range(100)}
        selected = stratify_sample(books, records, sample_size=20)
        self.assertEqual(len(selected), 20)
        self.assertEqual(len(set(selected)), 20)

    def test_covers_multiple_strata_proportionally(self):
        de_books = self._books(80, "de", prefix="de_book")
        en_books = self._books(20, "en", prefix="en_book")
        books = de_books + en_books
        records = {}
        for b in de_books:
            records[manifest_key(b)] = {"publication": [{"startDate": "2010"}]}
        for b in en_books:
            records[manifest_key(b)] = {"publication": [{"startDate": "1990"}]}
        selected = stratify_sample(books, records, sample_size=50)
        selected_langs = {b["language"] for b in books if manifest_key(b) in selected}
        self.assertEqual(selected_langs, {"de", "en"})

    def test_deterministic_for_fixed_seed(self):
        books = self._books(50)
        records = {f"book{i}": {"publication": [{"startDate": "2010"}]} for i in range(50)}
        first = stratify_sample(books, records, sample_size=10, seed=42)
        second = stratify_sample(books, records, sample_size=10, seed=42)
        self.assertEqual(first, second)

    def test_never_exceeds_available_books(self):
        books = self._books(5)
        records = {f"book{i}": {} for i in range(5)}
        selected = stratify_sample(books, records, sample_size=50)
        self.assertEqual(len(selected), 5)
