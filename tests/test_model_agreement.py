"""Unit tests for dnb_toc_ground_truth.model_agreement -- corpus-level
metrics for bulk-gate model selection: inter-model agreement, closeness
to arbitration-sourced ground truth, and a derived candidate-pair
ranking. See design spec
docs/superpowers/specs/2026-08-23-model-comparison-metrics-design.md."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dnb_toc_ground_truth import corpus, vision
from dnb_toc_ground_truth.model_agreement import (
    ModelGroundTruthMetrics,
    PairAgreement,
    arbitration_ground_truth_agreement,
    discover_all_cached_models,
    pairwise_model_agreement,
)
from dnb_toc_ground_truth.toc_entry import TocEntry


def _write_llm_cache_entry(key: str, model: str, entries: list[TocEntry]) -> None:
    vision.write_cached_llm_entries(corpus.llm_cache_dir(), key, model, entries)


def _write_manifest(keys: list[str]) -> None:
    corpus.manifest_path().write_text(
        json.dumps({"toc_only": True, "books": [{"filename": f"{k}.pdf", "doi": None} for k in keys]}),
        encoding="utf-8",
    )


class TestDiscoverAllCachedModels(unittest.TestCase):
    def test_finds_every_model_with_at_least_one_cache_entry(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            _write_llm_cache_entry("111", "model/a", [
                TocEntry(title="Introduction", printed_page_number="1", source_page_index=0),
            ])
            _write_llm_cache_entry("222", "model/b", [
                TocEntry(title="Introduction", printed_page_number="1", source_page_index=0),
            ])
            models = discover_all_cached_models()
            self.assertEqual(models, ["model__a", "model__b"])

    def test_returns_empty_list_when_no_cache_exists_yet(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            self.assertEqual(discover_all_cached_models(), [])

    def test_a_model_id_containing_a_dot_is_not_split_on_it(self):
        # cache filenames are "<key>.<safe_model>.json" -- the manifest
        # key never contains a dot, but a real model id can
        # ("mistralai/Mistral-Small-3.2-24B-Instruct-2506" sanitizes to
        # "mistralai__Mistral-Small-3.2-24B-Instruct-2506"). Splitting on
        # the FIRST dot only (not any dot) must recover the full model id.
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            _write_llm_cache_entry("111", "mistralai/Mistral-Small-3.2-24B-Instruct-2506", [
                TocEntry(title="Introduction", printed_page_number="1", source_page_index=0),
            ])
            models = discover_all_cached_models()
            self.assertEqual(models, ["mistralai__Mistral-Small-3.2-24B-Instruct-2506"])


class TestPairwiseModelAgreement(unittest.TestCase):
    def test_identical_readings_score_100_percent(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            _write_manifest(["111"])
            entries = [TocEntry(title="1. Introduction", printed_page_number="1", source_page_index=0)]
            _write_llm_cache_entry("111", "model/a", entries)
            _write_llm_cache_entry("111", "model/b", entries)
            result = pairwise_model_agreement(["model/a", "model/b"])
            self.assertEqual(len(result), 1)
            pair = result[0]
            self.assertEqual({pair.model_a, pair.model_b}, {"model/a", "model/b"})
            self.assertEqual(pair.mean_agreement, 1.0)
            self.assertEqual(pair.n_books, 1)

    def test_macro_averages_across_every_shared_book(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            _write_manifest(["111", "222"])
            same = [TocEntry(title="1. Introduction", printed_page_number="1", source_page_index=0)]
            _write_llm_cache_entry("111", "model/a", same)
            _write_llm_cache_entry("111", "model/b", same)
            _write_llm_cache_entry("222", "model/a", [
                TocEntry(title="1. Introduction", printed_page_number="1", source_page_index=0),
            ])
            _write_llm_cache_entry("222", "model/b", [
                TocEntry(title="Completely Different", printed_page_number="99", source_page_index=0),
            ])
            result = pairwise_model_agreement(["model/a", "model/b"])
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0].n_books, 2)
            self.assertAlmostEqual(result[0].mean_agreement, 0.5)

    def test_a_pair_sharing_zero_books_is_omitted_not_reported_as_zero(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            _write_manifest(["111", "222"])
            _write_llm_cache_entry("111", "model/a", [
                TocEntry(title="Introduction", printed_page_number="1", source_page_index=0),
            ])
            _write_llm_cache_entry("222", "model/b", [
                TocEntry(title="Introduction", printed_page_number="1", source_page_index=0),
            ])
            result = pairwise_model_agreement(["model/a", "model/b"])
            self.assertEqual(result, [])

    def test_scores_every_pair_among_three_or_more_models(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            _write_manifest(["111"])
            entries = [TocEntry(title="1. Introduction", printed_page_number="1", source_page_index=0)]
            _write_llm_cache_entry("111", "model/a", entries)
            _write_llm_cache_entry("111", "model/b", entries)
            _write_llm_cache_entry("111", "model/c", entries)
            result = pairwise_model_agreement(["model/a", "model/b", "model/c"])
            pairs = {frozenset((r.model_a, r.model_b)) for r in result}
            self.assertEqual(pairs, {
                frozenset({"model/a", "model/b"}),
                frozenset({"model/a", "model/c"}),
                frozenset({"model/b", "model/c"}),
            })


def _write_expected_json(key: str, entries: list[dict], source: str = "agent_arbitration") -> None:
    corpus.ground_truth_dir().mkdir(parents=True, exist_ok=True)
    corpus.expected_json_path(key).write_text(
        json.dumps({"entries": entries, "verified": source == "agent_arbitration", "source": source}),
        encoding="utf-8",
    )


class TestArbitrationGroundTruthAgreement(unittest.TestCase):
    def test_perfect_match_scores_1_0_across_all_three_metrics(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            _write_manifest(["111"])
            _write_expected_json("111", [
                {"title": "1. Introduction", "authors": [], "printed_page_number": "1", "skip": False},
            ])
            _write_llm_cache_entry("111", "model/a", [
                TocEntry(title="1. Introduction", printed_page_number="1", source_page_index=0),
            ])
            result = arbitration_ground_truth_agreement(["model/a"])
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0].model, "model/a")
            self.assertEqual(result[0].precision, 1.0)
            self.assertEqual(result[0].recall, 1.0)
            self.assertEqual(result[0].f1, 1.0)
            self.assertEqual(result[0].n_books, 1)

    def test_ignores_bulk_gate_sourced_ground_truth(self):
        # bulk_gate ground truth can be produced FROM this model's own
        # raw reading (if it was part of the winning pair) -- comparing
        # against it would be circular. Only agent_arbitration-sourced
        # ground truth (independent of any model's raw reading) is used.
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            _write_manifest(["111"])
            _write_expected_json("111", [
                {"title": "1. Introduction", "authors": [], "printed_page_number": "1", "skip": False},
            ], source="bulk_gate")
            _write_llm_cache_entry("111", "model/a", [
                TocEntry(title="1. Introduction", printed_page_number="1", source_page_index=0),
            ])
            result = arbitration_ground_truth_agreement(["model/a"])
            self.assertEqual(result, [])

    def test_a_model_with_no_cache_entry_for_any_arbitrated_book_is_omitted(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            _write_manifest(["111"])
            _write_expected_json("111", [
                {"title": "1. Introduction", "authors": [], "printed_page_number": "1", "skip": False},
            ])
            result = arbitration_ground_truth_agreement(["model/a"])
            self.assertEqual(result, [])

    def test_macro_averages_across_every_qualifying_book(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            _write_manifest(["111", "222"])
            _write_expected_json("111", [
                {"title": "1. Introduction", "authors": [], "printed_page_number": "1", "skip": False},
            ])
            _write_expected_json("222", [
                {"title": "1. Introduction", "authors": [], "printed_page_number": "1", "skip": False},
                {"title": "2. Methods", "authors": [], "printed_page_number": "20", "skip": False},
            ])
            _write_llm_cache_entry("111", "model/a", [
                TocEntry(title="1. Introduction", printed_page_number="1", source_page_index=0),
            ])
            _write_llm_cache_entry("222", "model/a", [
                TocEntry(title="1. Introduction", printed_page_number="1", source_page_index=0),
            ])  # misses "2. Methods" entirely -> recall 0.5 on book 222
            result = arbitration_ground_truth_agreement(["model/a"])
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0].n_books, 2)
            self.assertAlmostEqual(result[0].recall, (1.0 + 0.5) / 2)

    def test_compares_all_entries_including_skip_true_ones(self):
        # Unlike crossref_evaluation.evaluate_book (real chapters only),
        # this measures raw TOC-line extraction fidelity -- a divider
        # line the model correctly read counts as a true positive too.
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            _write_manifest(["111"])
            _write_expected_json("111", [
                {"title": "Part I", "authors": [], "printed_page_number": "9", "skip": True},
                {"title": "1. Introduction", "authors": [], "printed_page_number": "11", "skip": False},
            ])
            _write_llm_cache_entry("111", "model/a", [
                TocEntry(title="Part I", printed_page_number="9", source_page_index=0, skip=True),
                TocEntry(title="1. Introduction", printed_page_number="11", source_page_index=0, skip=False),
            ])
            result = arbitration_ground_truth_agreement(["model/a"])
            self.assertEqual(result[0].precision, 1.0)
            self.assertEqual(result[0].recall, 1.0)


if __name__ == "__main__":
    unittest.main()
