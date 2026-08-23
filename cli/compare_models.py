#!/usr/bin/env python3
"""Corpus-level model-comparison metrics for the bulk-gate two-model
agreement gate: how similar two models' raw TOC readings are to each
other, how close each model's raw reading is to arbitration-sourced (and
Crossref) ground truth, and a derived score ranking candidate pairs by
combining both -- see design spec
docs/superpowers/specs/2026-08-23-model-comparison-metrics-design.md.

Usage:
    uv run python cli/compare_models.py
    uv run python cli/compare_models.py --corpus pilot
"""

import argparse
from pathlib import Path

from dnb_toc_ground_truth import corpus, inference
from dnb_toc_ground_truth.crossref_evaluation import evaluate_model_corpus
from dnb_toc_ground_truth.model_agreement import (
    arbitration_ground_truth_agreement,
    discover_all_cached_models,
    pairwise_model_agreement,
    rank_candidate_pairs,
)


def _print_agreement_matrix(models: list[str], agreements: list) -> None:
    print("\n=== Pairwise agreement ===")
    by_pair = {frozenset((a.model_a, a.model_b)): a for a in agreements}
    for i, model_a in enumerate(models):
        for model_b in models[i + 1:]:
            pair = by_pair.get(frozenset((model_a, model_b)))
            if pair is None:
                print(f"{model_a} <-> {model_b}: no shared books")
            else:
                print(f"{model_a} <-> {model_b}: {pair.mean_agreement:.0%} (n={pair.n_books})")


def _print_accuracy_table(models: list[str], gt_metrics: list, crossref_results: dict) -> None:
    print("\n=== Per-model accuracy ===")
    gt_by_model = {m.model: m for m in gt_metrics}
    for model in models:
        gt = gt_by_model.get(model)
        gt_str = (
            f"P={gt.precision:.0%} R={gt.recall:.0%} F1={gt.f1:.0%} (n={gt.n_books})"
            if gt else "no arbitration-GT coverage"
        )
        cr_results, _ = crossref_results.get(model, ([], []))
        if cr_results:
            mean_f1 = sum(r.f1 for r in cr_results) / len(cr_results)
            cr_str = f"F1={mean_f1:.0%} (n={len(cr_results)})"
        else:
            cr_str = "no crossref coverage"
        print(f"{model}: arbitration-GT[{gt_str}] crossref[{cr_str}]")


def _print_candidate_ranking(scored: list, unscored: list) -> None:
    print("\n=== Candidate pair ranking (best first) ===")
    for s in scored:
        print(
            f"{s.model_a} + {s.model_b}: score={s.score:+.2f} "
            f"(f1_a={s.f1_a:.0%} f1_b={s.f1_b:.0%} observed={s.observed_agreement:.0%} "
            f"expected={s.expected_agreement:.0%} kappa={s.kappa:+.2f} n={s.n_books})"
        )
    if unscored:
        print("\nUnscored pairs (missing arbitration-GT coverage for one or both models):")
        for model_a, model_b in unscored:
            print(f"  {model_a} + {model_b}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--corpus", default=None,
        help=f"Corpus to operate on (default: config file's \"corpus\", or {corpus.DEFAULT_CORPUS_NAME!r})",
    )
    parser.add_argument(
        "--config-file", type=Path, default=Path(inference.DEFAULT_CONFIG_FILENAME),
        help=f"Path to the config file (default: {inference.DEFAULT_CONFIG_FILENAME})",
    )
    args = parser.parse_args()

    config = inference.load_config(args.config_file)
    corpus.set_corpus(args.corpus or config.get("corpus") or corpus.DEFAULT_CORPUS_NAME)

    models = discover_all_cached_models()
    if not models:
        print("No cached model readings found.")
        return 0

    agreements = pairwise_model_agreement(models)
    gt_metrics = arbitration_ground_truth_agreement(models)
    crossref_results = {model: evaluate_model_corpus(model) for model in models}

    _print_agreement_matrix(models, agreements)
    _print_accuracy_table(models, gt_metrics, crossref_results)
    scored, unscored = rank_candidate_pairs(agreements, gt_metrics)
    _print_candidate_ranking(scored, unscored)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
