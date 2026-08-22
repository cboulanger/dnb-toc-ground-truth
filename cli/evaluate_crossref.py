#!/usr/bin/env python3
"""Measures precision/recall/F1 between this corpus's own ground truth and
its committed Crossref evaluation corpus, plus optionally one or more
llm-cache models' raw extractions -- see
dnb_toc_ground_truth.crossref_evaluation for the full methodology.

Usage:
    uv run python cli/evaluate_crossref.py
    uv run python cli/evaluate_crossref.py --full
    uv run python cli/evaluate_crossref.py --min-f1 0.5
    uv run python cli/evaluate_crossref.py --model Qwen/Qwen3-Omni-30B-A3B-Instruct
    uv run python cli/evaluate_crossref.py --all-models --full
"""

import argparse
import asyncio
from pathlib import Path

from dnb_toc_ground_truth import corpus, inference
from dnb_toc_ground_truth.crossref_evaluation import (
    BookMetrics,
    backfill_model_cache,
    discover_cached_models,
    evaluate_corpus,
    evaluate_model_corpus,
)


def _print_book_lines(results: list[BookMetrics]) -> None:
    for r in results:
        print(f"[{r.key}] precision={r.precision:.0%} recall={r.recall:.0%} f1={r.f1:.0%} tp={r.tp} fp={r.fp} fn={r.fn}")


def _print_model_block(model: str, results: list[BookMetrics], no_cache: list[str], full: bool) -> None:
    print(f"\n=== model: {model} ===")
    if full:
        _print_book_lines(results)
        print()
    if results:
        mean_precision = sum(r.precision for r in results) / len(results)
        mean_recall = sum(r.recall for r in results) / len(results)
        mean_f1 = sum(r.f1 for r in results) / len(results)
        print(
            f"{len(results)} book(s) compared, mean precision={mean_precision:.0%} "
            f"mean recall={mean_recall:.0%} mean f1={mean_f1:.0%}"
        )
    else:
        print("No crossref-sample books had a cache entry for this model.")
    print(f"{len(no_cache)} crossref-sample book(s) with no cache entry for this model.")


def _run_backfill(args: argparse.Namespace, config: dict) -> None:
    """Resolves each --model against --endpoints-file and backfills its
    missing Crossref-evaluation-corpus llm-cache entries -- see design
    spec docs/superpowers/specs/2026-08-22-evaluate-crossref-backfill-
    design.md. Raises SystemExit if --backfill was given with no --model
    at all, and lets resolve_model_endpoints' own ValueError (naming the
    unmatched model id) propagate uncaught -- same fail-loud-and-early
    convention as generate_ground_truth.py's own endpoint resolution."""
    if not args.model:
        raise SystemExit("--backfill requires at least one --model")
    endpoints_file = Path(args.endpoints_file or config.get("endpoints_file", inference.DEFAULT_ENDPOINTS_FILENAME))
    entries = inference.load_endpoint_entries(endpoints_file)
    for model in args.model:
        endpoint = inference.resolve_model_endpoints([model], "vision", entries)[0]
        succeeded, failed = asyncio.run(backfill_model_cache(model, endpoint, corpus.llm_cache_dir()))
        print(f"[backfill] {model}: {len(succeeded)} written, {len(failed)} failed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--full", action="store_true",
        help="Print a per-book precision/recall/F1 line for every compared book, not just the aggregate mean",
    )
    parser.add_argument(
        "--min-f1", type=float, default=None,
        help="Exit 1 if the aggregate mean ground-truth F1 falls below this (0-1). Unset: no gate enforced.",
    )
    parser.add_argument(
        "--model", action="append", default=None,
        help="Also score this model's cached llm-cache extraction against the crossref sample, alongside "
             "ground truth (repeatable). Model id as it appears in data/corpus/pilot/llm-cache/v2/ filenames "
             "(e.g. 'Qwen/Qwen3-Omni-30B-A3B-Instruct' or its sanitized 'Qwen__Qwen3-Omni-30B-A3B-Instruct' form).",
    )
    parser.add_argument(
        "--all-models", action="store_true",
        help="Score every model with at least one llm-cache entry for a crossref-sample book, without naming "
             "them individually. Combines with --model.",
    )
    parser.add_argument(
        "--backfill", action="store_true",
        help="Before scoring, extract and cache any --model book missing a llm-cache entry, "
             "resolved against --endpoints-file (vision-input models only)",
    )
    parser.add_argument(
        "--endpoints-file", type=Path, default=None,
        help=f"Path to the endpoints file, only used with --backfill (default: "
             f"{inference.DEFAULT_ENDPOINTS_FILENAME}, or config file's \"endpoints_file\")",
    )
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

    if args.backfill:
        _run_backfill(args, config)

    results, no_coverage = evaluate_corpus()
    if args.full:
        _print_book_lines(results)
        print()

    if results:
        mean_precision = sum(r.precision for r in results) / len(results)
        mean_recall = sum(r.recall for r in results) / len(results)
        mean_f1 = sum(r.f1 for r in results) / len(results)
        print(
            f"{len(results)} book(s) compared, mean precision={mean_precision:.0%} "
            f"mean recall={mean_recall:.0%} mean f1={mean_f1:.0%}"
        )
    else:
        mean_f1 = None
        print("No books had both ground truth and a Crossref evaluation-corpus entry.")
    print(f"{len(no_coverage)} book(s) with ground truth but no Crossref evaluation coverage.")

    models = list(dict.fromkeys((args.model or []) + (discover_cached_models() if args.all_models else [])))
    for model in models:
        model_results, no_cache = evaluate_model_corpus(model)
        _print_model_block(model, model_results, no_cache, args.full)

    if args.min_f1 is not None and (mean_f1 is None or mean_f1 < args.min_f1):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
