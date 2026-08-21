"""Unit tests for toc_entry.py -- vendored from
chapter-segmentation's tests/test_segmentation.py's TocEntry/
_parse_toc_page_number/_toc_items_to_entries coverage and
tests/test_llm_json.py's parse_json_array coverage, trimmed to what this
repo actually exercises (page-number parsing, item-to-entry conversion,
JSON-array extraction)."""

import unittest

from dnb_toc_ground_truth.toc_entry import TocEntry, _parse_toc_page_number, _toc_items_to_entries, parse_json_array


class TestParseTocPageNumber(unittest.TestCase):
    def test_parses_digit_string(self):
        self.assertEqual(_parse_toc_page_number("42"), 42)

    def test_parses_lowercase_roman(self):
        self.assertEqual(_parse_toc_page_number("vii"), 7)

    def test_rejects_implausibly_large_roman(self):
        self.assertIsNone(_parse_toc_page_number("mmmmm"))

    def test_rejects_non_roman_word(self):
        self.assertIsNone(_parse_toc_page_number("civil"))


class TestTocEntryPageNormalization(unittest.TestCase):
    def test_legacy_negative_one_sentinel_becomes_none(self):
        entry = TocEntry(title="X", printed_page_number=-1, source_page_index=0)
        self.assertIsNone(entry.printed_page_number)

    def test_string_is_stripped(self):
        entry = TocEntry(title="X", printed_page_number=" 12 ", source_page_index=0)
        self.assertEqual(entry.printed_page_number, "12")


class TestTocItemsToEntries(unittest.TestCase):
    def test_converts_well_formed_item(self):
        entries = _toc_items_to_entries([
            {"title": "Introduction", "authors": ["Jane Doe"], "printed_page_number": "12", "skip": False},
        ])
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].title, "Introduction")
        self.assertEqual(entries[0].authors, ("Jane Doe",))
        self.assertFalse(entries[0].skip)

    def test_skips_items_with_too_short_title(self):
        entries = _toc_items_to_entries([{"title": "AB", "printed_page_number": "1"}])
        self.assertEqual(entries, [])

    def test_tolerates_string_authors_field(self):
        entries = _toc_items_to_entries([{"title": "Chapter One", "authors": "Jane Doe", "printed_page_number": "1"}])
        self.assertEqual(entries[0].authors, ())

    def test_defaults_skip_to_false_when_absent(self):
        entries = _toc_items_to_entries([{"title": "Chapter One", "printed_page_number": "1"}])
        self.assertFalse(entries[0].skip)


class TestParseJsonArray(unittest.TestCase):
    def test_extracts_bare_array(self):
        self.assertEqual(parse_json_array('[{"a": 1}]'), [{"a": 1}])

    def test_strips_markdown_code_fence(self):
        self.assertEqual(parse_json_array('```json\n[{"a": 1}]\n```'), [{"a": 1}])

    def test_raises_on_no_array_found(self):
        with self.assertRaises(ValueError):
            parse_json_array("no array here")


if __name__ == "__main__":
    unittest.main()
