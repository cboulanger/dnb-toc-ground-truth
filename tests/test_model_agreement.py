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

from dnb_toc_ground_truth import corpus, model_agreement, vision
from dnb_toc_ground_truth.model_agreement import (
    ModelGroundTruthMetrics,
    PairAgreement,
    PairCandidateScore,
    arbitration_ground_truth_agreement,
    discover_all_cached_models,
    pairwise_model_agreement,
    rank_candidate_pairs,
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
    def test_finds_every_model_with_at_least_min_readings_cache_entries(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)), \
                patch.object(model_agreement, "_MIN_MODEL_READINGS", 1):
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
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)), \
                patch.object(model_agreement, "_MIN_MODEL_READINGS", 1):
            _write_llm_cache_entry("111", "mistralai/Mistral-Small-3.2-24B-Instruct-2506", [
                TocEntry(title="Introduction", printed_page_number="1", source_page_index=0),
            ])
            models = discover_all_cached_models()
            self.assertEqual(models, ["mistralai__Mistral-Small-3.2-24B-Instruct-2506"])

    def test_default_threshold_excludes_a_model_with_too_few_readings(self):
        # A one-off smoke-test endpoint's handful of readings isn't a
        # meaningful sample for a corpus-level comparison metric -- same
        # rationale as cli/corpus_status.py's own _MIN_MODEL_READINGS.
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            for i in range(49):
                _write_llm_cache_entry(f"below-{i}", "model/below-threshold", [
                    TocEntry(title="Introduction", printed_page_number="1", source_page_index=0),
                ])
            for i in range(50):
                _write_llm_cache_entry(f"above-{i}", "model/at-threshold", [
                    TocEntry(title="Introduction", printed_page_number="1", source_page_index=0),
                ])
            models = discover_all_cached_models()
            self.assertEqual(models, ["model__at-threshold"])


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


class TestRankCandidatePairs(unittest.TestCase):
    def test_kappa_near_zero_when_agreement_matches_chance_prediction(self):
        # f1_a=0.8, f1_b=0.8 -> expected = 0.8*0.8 + 0.2*0.2 = 0.68.
        # Observed exactly at that expected value -> kappa == 0.
        agreements = [PairAgreement(model_a="a", model_b="b", mean_agreement=0.68, n_books=10)]
        gt_metrics = [
            ModelGroundTruthMetrics(model="a", precision=0.8, recall=0.8, f1=0.8, n_books=20),
            ModelGroundTruthMetrics(model="b", precision=0.8, recall=0.8, f1=0.8, n_books=15),
        ]
        scored, unscored = rank_candidate_pairs(agreements, gt_metrics)
        self.assertEqual(unscored, [])
        self.assertEqual(len(scored), 1)
        self.assertAlmostEqual(scored[0].expected_agreement, 0.68)
        self.assertAlmostEqual(scored[0].kappa, 0.0, places=6)
        self.assertAlmostEqual(scored[0].score, 0.8, places=6)

    def test_positive_kappa_when_agreement_exceeds_chance_prediction(self):
        agreements = [PairAgreement(model_a="a", model_b="b", mean_agreement=0.98, n_books=10)]
        gt_metrics = [
            ModelGroundTruthMetrics(model="a", precision=0.8, recall=0.8, f1=0.8, n_books=20),
            ModelGroundTruthMetrics(model="b", precision=0.8, recall=0.8, f1=0.8, n_books=15),
        ]
        scored, unscored = rank_candidate_pairs(agreements, gt_metrics)
        self.assertGreater(scored[0].kappa, 0.0)
        # score penalizes the excess: min(f1_a, f1_b) - kappa < min(f1_a, f1_b)
        self.assertLess(scored[0].score, 0.8)

    def test_pair_missing_arbitration_gt_for_either_model_is_unscored(self):
        agreements = [PairAgreement(model_a="a", model_b="b", mean_agreement=0.9, n_books=10)]
        gt_metrics = [ModelGroundTruthMetrics(model="a", precision=0.8, recall=0.8, f1=0.8, n_books=20)]
        scored, unscored = rank_candidate_pairs(agreements, gt_metrics)
        self.assertEqual(scored, [])
        self.assertEqual(unscored, [("a", "b")])

    def test_sorted_best_score_first(self):
        agreements = [
            PairAgreement(model_a="a", model_b="b", mean_agreement=0.68, n_books=10),
            PairAgreement(model_a="a", model_b="c", mean_agreement=0.99, n_books=10),
        ]
        gt_metrics = [
            ModelGroundTruthMetrics(model="a", precision=0.8, recall=0.8, f1=0.8, n_books=20),
            ModelGroundTruthMetrics(model="b", precision=0.8, recall=0.8, f1=0.8, n_books=15),
            ModelGroundTruthMetrics(model="c", precision=0.8, recall=0.8, f1=0.8, n_books=15),
        ]
        scored, _ = rank_candidate_pairs(agreements, gt_metrics)
        self.assertEqual((scored[0].model_a, scored[0].model_b), ("a", "b"))
        self.assertGreater(scored[0].score, scored[1].score)

    def test_both_models_perfect_does_not_divide_by_zero(self):
        # expected_agreement = 1.0*1.0 + 0.0*0.0 = 1.0 -> (1 - expected) == 0.
        agreements = [PairAgreement(model_a="a", model_b="b", mean_agreement=1.0, n_books=5)]
        gt_metrics = [
            ModelGroundTruthMetrics(model="a", precision=1.0, recall=1.0, f1=1.0, n_books=5),
            ModelGroundTruthMetrics(model="b", precision=1.0, recall=1.0, f1=1.0, n_books=5),
        ]
        scored, _ = rank_candidate_pairs(agreements, gt_metrics)
        self.assertEqual(scored[0].kappa, 0.0)
        self.assertEqual(scored[0].score, 1.0)

    def test_asymmetric_f1_pair_uses_min_for_score_and_correct_cross_term(self):
        # f1_a=0.9, f1_b=0.6 -> expected = 0.9*0.6 + 0.1*0.4 = 0.54 + 0.04 = 0.58.
        # Distinguishes min(f1_a, f1_b) from max/either single value, and confirms
        # each model's own F1 is attributed to the correct side (a vs b).
        agreements = [PairAgreement(model_a="a", model_b="b", mean_agreement=0.58, n_books=10)]
        gt_metrics = [
            ModelGroundTruthMetrics(model="a", precision=0.9, recall=0.9, f1=0.9, n_books=20),
            ModelGroundTruthMetrics(model="b", precision=0.6, recall=0.6, f1=0.6, n_books=15),
        ]
        scored, unscored = rank_candidate_pairs(agreements, gt_metrics)
        self.assertEqual(unscored, [])
        self.assertEqual(scored[0].f1_a, 0.9)
        self.assertEqual(scored[0].f1_b, 0.6)
        self.assertAlmostEqual(scored[0].expected_agreement, 0.58)
        self.assertAlmostEqual(scored[0].kappa, 0.0, places=6)
        # score uses min(f1_a, f1_b) = 0.6, not max (0.9) or either average.
        self.assertAlmostEqual(scored[0].score, 0.6, places=6)


if __name__ == "__main__":
    unittest.main()
