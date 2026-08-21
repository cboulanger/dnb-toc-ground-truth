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
from unittest.mock import Mock

import httpx

from dnb_toc_ground_truth.crossref import normalize_isbn


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
