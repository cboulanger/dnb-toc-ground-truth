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
    PairAgreement,
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


if __name__ == "__main__":
    unittest.main()
