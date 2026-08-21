"""Unit tests for evaluation/dnb_toc_matching.py -- the whole-book
agreement gate that decides which dnb-toc-only books' extracted entries
are trustworthy enough for the bulk ground-truth tier (design spec
docs/superpowers/specs/2026-08-15-dnb-toc-ground-truth-generation-design.md
section 4). No PDFs, no network -- pure functions over synthetic
TocEntry lists."""

import unittest
from dataclasses import replace

from dnb_toc_ground_truth.toc_entry import TocEntry
from dnb_toc_ground_truth.matching import (
    _title_near_identical,
    align_toc_entries,
    diff_toc_entries,
    gate_book,
    toc_entry_to_gt_dict,
)


def _entry(title: str, page: str | int | None, authors: tuple[str, ...] = ()) -> TocEntry:
    return TocEntry(title=title, printed_page_number=page, source_page_index=0, authors=authors)


class TestAlignTocEntries(unittest.TestCase):
    def test_matches_same_page_and_similar_title(self):
        a = [_entry("Einleitung", 9)]
        b = [_entry("Einleitung", 9)]
        self.assertEqual(align_toc_entries(a, b), [(0, 0)])

    def test_no_match_on_page_mismatch(self):
        a = [_entry("Einleitung", 9)]
        b = [_entry("Einleitung", 11)]
        self.assertEqual(align_toc_entries(a, b), [])

    def test_no_match_when_either_page_unknown(self):
        a = [_entry("Einleitung", -1)]
        b = [_entry("Einleitung", 9)]
        self.assertEqual(align_toc_entries(a, b), [])

    def test_no_match_on_dissimilar_title_same_page(self):
        a = [_entry("Einleitung", 9)]
        b = [_entry("Bibliographie", 9)]
        self.assertEqual(align_toc_entries(a, b), [])

    def test_order_preserving_scan_misses_out_of_order_match(self):
        # Same "TOC order is book order" monotonicity tradeoff
        # src/chapter_segmentation/evidence/fusion.py's _align already
        # makes: a[0] ("Erster Teil", page 9) can only match b at or after
        # b-index 0. b[0] ("Zweiter Teil", page 40) doesn't match, so the
        # scan advances to b[1] ("Erster Teil", page 9), which does --
        # consuming last_j=1. a[1] ("Zweiter Teil", page 40) then has no
        # b-index left to scan (>= 2), even though a real match existed
        # earlier in b. This is expected behavior, not a bug.
        a = [_entry("Erster Teil", 9), _entry("Zweiter Teil", 40)]
        b = [_entry("Zweiter Teil", 40), _entry("Erster Teil", 9)]
        self.assertEqual(align_toc_entries(a, b), [(0, 1)])

    def test_empty_lists(self):
        self.assertEqual(align_toc_entries([], []), [])
        self.assertEqual(align_toc_entries([_entry("X", 1)], []), [])
        self.assertEqual(align_toc_entries([], [_entry("X", 1)]), [])

    def test_matches_via_title_variant_when_primary_title_is_a_wrapped_fragment(self):
        # Real case found in a 2026-08-15 smoke test (book 9783899718188):
        # the heuristic's regex only captures a wrapped title's last
        # line as .title, with the full title in title_variants. The
        # LLM reads the title whole. Without checking title_variants,
        # this real match was scored below the alignment threshold.
        a = [TocEntry(
            title="Systemtheorie für Recht und Rechtswissenschaft",
            printed_page_number=33, source_page_index=0,
            title_variants=(
                "Niklas Luhmann und das Recht - Über die Nutzlosigkeit der "
                "Systemtheorie für Recht und Rechtswissenschaft",
            ),
        )]
        b = [_entry(
            "Niklas Luhmann und das Recht - Über die Nutzlosigkeit der "
            "Systemtheorie für Recht und Rechtswissenschaft",
            33,
        )]
        self.assertEqual(align_toc_entries(a, b), [(0, 0)])

    def test_matches_despite_trailing_ocr_noise_via_partial_ratio(self):
        # Real garbled-dot-leader-OCR cases measured in the investigation
        # behind docs/superpowers/specs/2026-08-16-dnb-toc-uniform-ocr-design.md
        # section 1d: token_sort_ratio alone scores these well below the
        # 70.0 threshold (a handful of garbage tokens dominates a short
        # real title's token multiset) even though one title is exactly
        # the other's real content plus a trailing noise run.
        a = [_entry("Ein Interview ss m onen een ee eee eee ees", 81)]
        b = [_entry("Ein Interview", 81)]
        self.assertEqual(align_toc_entries(a, b), [(0, 0)])

    def test_still_rejects_genuinely_different_titles_on_the_same_page(self):
        # Negative control: partial_ratio must not become so permissive
        # that two different real entries sharing a page number align.
        a = [_entry("Die Einheit der Vernunft in der Vielfalt ihrer Stimmen", 117)]
        b = [_entry("Metaphysik nach Kant", 117)]
        self.assertEqual(align_toc_entries(a, b), [])

    def test_matches_two_null_page_entries_with_identical_titles(self):
        # Real case found via evaluation/experiments/dnb-toc-ground-truth.md's
        # bulk-gate spot-check (2026-08-19, book 9783495485019): both models
        # independently read a page-number-less "Anhang:" divider
        # identically, but the old blanket "either side None -> never
        # match" rule meant two IDENTICAL readings could never align,
        # silently duplicating the entry downstream in gate_book.
        a = [_entry("Anhang:", None)]
        b = [_entry("Anhang:", None)]
        self.assertEqual(align_toc_entries(a, b), [(0, 0)])

    def test_does_not_match_two_null_page_entries_with_different_titles(self):
        # Without a page number to confirm the pair, a looser bar would
        # risk merging two genuinely different unrelated dividers.
        a = [_entry("Anhang:", None)]
        b = [_entry("Vorwort", None)]
        self.assertEqual(align_toc_entries(a, b), [])

    def test_does_not_match_null_page_against_known_page(self):
        a = [_entry("Anhang:", None)]
        b = [_entry("Anhang:", 331)]
        self.assertEqual(align_toc_entries(a, b), [])

    def test_null_page_match_requires_near_exact_title_not_just_fuzzy(self):
        # A truncated/split title still scores ~100 via partial_ratio
        # (it's a strict prefix), which is exactly what the null-page
        # path must NOT accept without a page number to confirm the pair
        # -- token_sort_ratio (~42 for this real pair) is what actually
        # distinguishes it from a genuine near-duplicate.
        a = [_entry("Einleitung:", None)]
        b = [_entry("Einleitung: Endlichkeit und Verantwortung", None)]
        self.assertEqual(align_toc_entries(a, b), [])

    def test_matches_on_prefixed_page_marker(self):
        # The real reported bug this whole change exists for: two
        # independent extractions that both correctly read "R42" for the
        # same line must be able to align -- the old int-with--1-sentinel
        # representation collapsed both to the same "unknown" value and
        # skipped them before matching was even attempted.
        a = [_entry("Appendix", "R42")]
        b = [_entry("Appendix", "R42")]
        self.assertEqual(align_toc_entries(a, b), [(0, 0)])

    def test_matches_prefixed_marker_case_insensitively(self):
        a = [_entry("Appendix", "R42")]
        b = [_entry("Appendix", "r42")]
        self.assertEqual(align_toc_entries(a, b), [(0, 0)])

    def test_matches_roman_numeral_case_difference(self):
        a = [_entry("Foreword", "VII")]
        b = [_entry("Foreword", "vii")]
        self.assertEqual(align_toc_entries(a, b), [(0, 0)])

    def test_matches_arabic_page_with_leading_zero(self):
        a = [_entry("Einleitung", "07")]
        b = [_entry("Einleitung", "7")]
        self.assertEqual(align_toc_entries(a, b), [(0, 0)])


class TestDiffTocEntries(unittest.TestCase):
    def test_full_agreement_has_no_singletons(self):
        a = [_entry("Einleitung", 9), _entry("Schluss", 40)]
        b = [_entry("Einleitung", 9), _entry("Schluss", 40)]
        matched, only_a, only_b = diff_toc_entries(a, b)
        self.assertEqual(len(matched), 2)
        self.assertEqual(only_a, [])
        self.assertEqual(only_b, [])

    def test_partial_agreement_separates_singletons_per_side(self):
        a = [_entry("Einleitung", 9), _entry("Only in A", 20)]
        b = [_entry("Einleitung", 9), _entry("Only in B", 30)]
        matched, only_a, only_b = diff_toc_entries(a, b)
        self.assertEqual(len(matched), 1)
        self.assertEqual([e.title for e in only_a], ["Only in A"])
        self.assertEqual([e.title for e in only_b], ["Only in B"])

    def test_complete_disagreement_puts_everything_in_singletons(self):
        a = [_entry("A", 1)]
        b = [_entry("B", 2)]
        matched, only_a, only_b = diff_toc_entries(a, b)
        self.assertEqual(matched, [])
        self.assertEqual([e.title for e in only_a], ["A"])
        self.assertEqual([e.title for e in only_b], ["B"])

    def test_matched_pairs_hold_entry_objects_not_indices(self):
        a = [_entry("Einleitung", 9, authors=("A Author",))]
        b = [_entry("Einleitung", 9, authors=("B Author",))]
        matched, _, _ = diff_toc_entries(a, b)
        self.assertEqual(len(matched), 1)
        entry_a, entry_b = matched[0]
        self.assertEqual(entry_a.authors, ("A Author",))
        self.assertEqual(entry_b.authors, ("B Author",))


class TestGateBook(unittest.TestCase):
    def test_perfect_agreement_passes_with_union_equal_to_either_list(self):
        h = [_entry("Einleitung", 9), _entry("Schluss", 40)]
        l = [_entry("Einleitung", 9), _entry("Schluss", 40)]
        passed, entries = gate_book(h, l)
        self.assertTrue(passed)
        self.assertEqual([e.title for e in entries], ["Einleitung", "Schluss"])

    def test_below_threshold_rejects_whole_book(self):
        h = [_entry("Einleitung", 9), _entry("A", 20), _entry("B", 30), _entry("C", 40)]
        l = [_entry("Einleitung", 9)]  # agreement_rate = 1/4 = 0.25
        passed, entries = gate_book(h, l)
        self.assertFalse(passed)
        self.assertEqual(entries, [])

    def test_above_threshold_unions_singleton_entries_rather_than_dropping_them(self):
        # 9 of 10 heuristic entries agree with the LLM list -- rate 0.90.
        # The heuristic's 10th, LLM-missed entry must survive in the
        # merged result rather than being silently trimmed: the design's
        # core "no incomplete training target" requirement (spec section
        # 4.2) -- once a book clears the trust bar, a line only one
        # extractor caught is more likely a real miss than a hallucination.
        h = [_entry(f"Chapter {i}", i * 10) for i in range(1, 11)]
        l = h[:9]
        passed, entries = gate_book(h, l, threshold=0.90)
        self.assertTrue(passed)
        self.assertEqual(len(entries), 10)
        self.assertIn("Chapter 10", [e.title for e in entries])

    def test_empty_both_lists_rejects(self):
        self.assertEqual(gate_book([], []), (False, []))

    def test_merged_entries_sorted_by_printed_page_number(self):
        h = [_entry("Schluss", 40), _entry("Einleitung", 9)]
        l = [_entry("Schluss", 40), _entry("Einleitung", 9)]
        passed, entries = gate_book(h, l)
        self.assertTrue(passed)
        self.assertEqual([e.printed_page_number for e in entries], ["9", "40"])

    def test_matched_pair_keeps_heuristic_title_but_falls_back_to_llm_authors(self):
        # The heuristic (find_toc_candidates) almost never populates
        # authors -- only in a narrow "by <Name>" marker-line case -- so
        # always preferring the heuristic's own (empty) authors on a
        # matched pair would silently discard real author info the LLM
        # extracted. Falling back to the LLM's authors when the
        # heuristic's own are empty avoids that.
        h = [_entry("Einleitung", 9, authors=())]
        l = [_entry("Einleitung", 9, authors=("Jane Author",))]
        passed, entries = gate_book(h, l)
        self.assertTrue(passed)
        self.assertEqual(entries[0].title, "Einleitung")  # heuristic's title kept
        self.assertEqual(entries[0].authors, ("Jane Author",))  # LLM's authors used

    def test_matched_pair_keeps_heuristic_authors_when_heuristic_has_them(self):
        h = [_entry("Einleitung", 9, authors=("Regex Author",))]
        l = [_entry("Einleitung", 9, authors=("Different LLM Reading",))]
        passed, entries = gate_book(h, l)
        self.assertTrue(passed)
        self.assertEqual(entries[0].authors, ("Regex Author",))  # heuristic's own authors preferred when present

    def test_rejects_whole_book_when_a_matched_pair_has_near_but_not_exact_titles(self):
        # Real case (evaluation/experiments/dnb-toc-ground-truth.md's
        # bulk-gate spot-check, 2026-08-19, book 9783495485019): one side split a heading the
        # other read whole. The pair still aligns (page match + a high
        # partial_ratio score, since "Einleitung:" is a strict prefix of
        # the other), so the OLD gate_book would have silently kept the
        # worse (split) reading. Now the whole book must fail instead of
        # picking a side by fiat.
        a = [
            _entry("Einleitung:", 9),
            _entry("Endlichkeit und Verantwortung", 9),
            _entry("Biographie", 331),
        ]
        b = [
            _entry("Einleitung: Endlichkeit und Verantwortung", 9),
            _entry("Biographie", 331),
        ]
        passed, entries = gate_book(a, b)
        self.assertFalse(passed)
        self.assertEqual(entries, [])

    def test_null_page_agreement_merges_into_one_entry_not_a_duplicate(self):
        # Companion to the align_toc_entries null-page test: once the two
        # models' identical page-number-less readings actually align, the
        # merged book output must contain it ONCE, not twice.
        h = [_entry("Einleitung", 9), _entry("Anhang:", None)]
        l = [_entry("Einleitung", 9), _entry("Anhang:", None)]
        passed, entries = gate_book(h, l)
        self.assertTrue(passed)
        self.assertEqual([e.title for e in entries], ["Einleitung", "Anhang:"])


class TestTocEntryToGtDict(unittest.TestCase):
    def test_known_page_number_becomes_string(self):
        entry = _entry("Einleitung", 9, authors=("Jane Author",))
        self.assertEqual(
            toc_entry_to_gt_dict(entry),
            {"title": "Einleitung", "authors": ["Jane Author"], "printed_page_number": "9", "skip": False},
        )

    def test_unknown_page_number_becomes_none(self):
        entry = _entry("Bibliographie", -1)
        self.assertEqual(
            toc_entry_to_gt_dict(entry),
            {"title": "Bibliographie", "authors": [], "printed_page_number": None, "skip": False},
        )

    def test_skip_true_is_carried_through(self):
        entry = replace(_entry("Bibliographie", 200), skip=True)
        self.assertEqual(
            toc_entry_to_gt_dict(entry),
            {"title": "Bibliographie", "authors": [], "printed_page_number": "200", "skip": True},
        )


class TestTitleNearIdenticalNormalization(unittest.TestCase):
    """Regression tests for real false positives found measuring
    _title_near_identical against all 85 already-cached real dnb-toc-only
    book pairs (2026-08-19, see evaluation/experiments/dnb-toc-ground-truth.md's
    bulk-gate spot-check section) -- each pattern here is a title-FORMATTING
    difference, not a content disagreement, and must NOT flag a book for
    arbitration."""

    def test_leading_chapter_number_is_ignored(self):
        a = _entry("Decision-making", 9, authors=("John Hardman",))
        b = _entry("2 Decision-making", 9, authors=("John Hardman",))
        self.assertTrue(_title_near_identical(a, b))

    def test_leading_chapter_label_with_colon_is_ignored(self):
        a = _entry("Un/thinking Children in Development", 9, authors=("Erica Burman",))
        b = _entry("Chapter 1: Un/thinking Children in Development", 9, authors=("Erica Burman",))
        self.assertTrue(_title_near_identical(a, b))

    def test_leading_subsection_number_is_ignored(self):
        a = _entry("Extraordinary revenues, 1716–88", 9)
        b = _entry("1.4 Extraordinary revenues, 1716–88", 9)
        self.assertTrue(_title_near_identical(a, b))

    def test_own_author_embedded_with_dash_is_ignored(self):
        a = _entry("Introduction—Edward L. Smither", 9, authors=("Edward L. Smither",))
        b = _entry("Introduction", 9, authors=("Edward L. Smither",))
        self.assertTrue(_title_near_identical(a, b))

    def test_own_author_embedded_with_leading_period_is_ignored(self):
        a = _entry("JOSEPH ROTH. Die Eiche Goethes in Buchenwald", 9, authors=("JOSEPH ROTH",))
        b = _entry("Die Eiche Goethes in Buchenwald", 9, authors=("Joseph Roth",))
        self.assertTrue(_title_near_identical(a, b))

    def test_own_author_embedded_with_leading_colon_is_ignored(self):
        a = _entry("Leo Roth: Der Erziehungstheoretiker Erasmus von Rotterdam", 9, authors=("Leo Roth",))
        b = _entry("Der Erziehungstheoretiker Erasmus von Rotterdam", 9, authors=("Leo Roth",))
        self.assertTrue(_title_near_identical(a, b))

    def test_german_low_quote_style_is_ignored(self):
        a = _entry('"Mythologie der Vernunft"', 9, authors=("Uwe Beyer",))
        b = _entry("„Mythologie der Vernunft“", 9, authors=("Uwe Beyer",))
        self.assertTrue(_title_near_identical(a, b))

    def test_still_rejects_a_genuine_word_level_reading_difference(self):
        # Real case (book 0126135169): a genuine misread, not formatting.
        a = _entry("3. Interpretation of Pima Floating Quantifiers", 9)
        b = _entry("3. Interpretation of Pima Floated Quantifiers", 9)
        self.assertFalse(_title_near_identical(a, b))

    def test_still_rejects_a_dropped_subtitle(self):
        # Real case (book 9783835359208): one side's extra content is a
        # genuine, non-recoverable subtitle, not redundant metadata --
        # must still be flagged rather than silently discarded.
        a = _entry("Zur Einführung", 9)
        b = _entry("Krise der Kritik? Zur Einführung", 9)
        self.assertFalse(_title_near_identical(a, b))

    def test_trailing_parenthetical_stripped_even_with_a_trailing_period(self):
        # Real case (book 9783161636820): one model appended a trailing
        # period after the closing paren, the other didn't -- the period
        # defeated _TRAILING_PARENTHETICAL_RE's end-of-string anchor, so
        # only one side's parenthetical got stripped, scoring an
        # otherwise-identical pair below threshold.
        a = _entry('"Therefore I Quit, and I Am Consoled Over Dust and Ashes" (Job 42:6).', 315)
        b = _entry('"Therefore I Quit, and I Am Consoled Over Dust and Ashes" (Job 42:6)', 315)
        self.assertTrue(_title_near_identical(a, b))
