#!/usr/bin/env python3
"""Measures precision/recall/F1 between this corpus's own ground truth and
its committed Crossref evaluation corpus
(data/corpus/pilot/evaluation/<key>.expected.json) -- see design spec
docs/superpowers/specs/2026-08-22-crossref-evaluation-corpus-design.md
(revises docs/superpowers/specs/2026-08-21-crossref-cross-validation-design.md).

For every book with both a .expected.json (ground truth) and an
evaluation/<key>.expected.json (written by cli/backfill_crossref.py or
cli/fetch_corpus.py's real-time hook, via crossref.write_evaluation_entry
-- already filtered to Crossref book-chapter items with real page data,
already at least min_chapters strong), compares the ground truth's real
chapters ("skip": false) against the evaluation corpus's entries via
matching.diff_toc_entries -- reused completely unmodified, since it
already aligns on title (chapter-number-prefix and capitalization
normalized) and first-page-number equivalence. From the resulting
(matched, only_in_gt, only_in_crossref) counts: true positives = matched,
false negatives = only_in_gt (a real GT chapter Crossref didn't register
or match), false positives = only_in_crossref (a Crossref chapter with no
GT match) -- standard precision/recall/F1 from there. Books with no
evaluation-corpus entry at all are reported separately, not silently
dropped.

Usage:
    uv run python cli/evaluate_crossref.py
    uv run python cli/evaluate_crossref.py --full
    uv run python cli/evaluate_crossref.py --min-f1 0.5
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from dnb_toc_ground_truth import corpus, crossref, matching
from dnb_toc_ground_truth.toc_entry import TocEntry, _parse_toc_page_number


@dataclass(frozen=True)
class BookMetrics:
    key: str
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float


def _load_entries(path: Path) -> tuple[TocEntry, ...]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return tuple(
        TocEntry(
            title=e["title"], authors=tuple(e.get("authors", [])),
            printed_page_number=e["printed_page_number"], source_page_index=-1, skip=e.get("skip", False),
        )
        for e in data["entries"]
    )


def _page_sort_key(entry: TocEntry) -> tuple:
    # matching.diff_toc_entries' underlying align_toc_entries does a
    # single greedy, order-preserving scan ("TOC order is book order" --
    # true for two independent reads of the same printed TOC). Crossref's
    # /works response order is NOT page order (confirmed empirically,
    # 2026-08-21, isbn:9783111702681: sorting its chapter list by page
    # before diffing took that book's match count from 5/20 to 20/20) --
    # so both sides are re-sorted into page order here before ever
    # reaching diff_toc_entries. Mirrors matching.gate_book's own
    # merge-output sort key exactly (unknown page sorts last).
    value = _parse_toc_page_number(entry.printed_page_number) if entry.printed_page_number else None
    return (entry.printed_page_number is None, value if value is not None else 0, entry.printed_page_number or "")


def evaluate_book(key: str, gt_entries: tuple[TocEntry, ...], crossref_entries: tuple[TocEntry, ...]) -> BookMetrics:
    gt_real = sorted((e for e in gt_entries if not e.skip), key=_page_sort_key)
    crossref_sorted = sorted(crossref_entries, key=_page_sort_key)
    matched_pairs, only_in_gt, only_in_crossref = matching.diff_toc_entries(gt_real, crossref_sorted)
    tp, fn, fp = len(matched_pairs), len(only_in_gt), len(only_in_crossref)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return BookMetrics(key=key, tp=tp, fp=fp, fn=fn, precision=precision, recall=recall, f1=f1)


def evaluate_corpus() -> tuple[list[BookMetrics], list[str]]:
    """Returns (results, keys_with_no_evaluation_coverage) for every
    manifest book that has a .expected.json. The evaluation-corpus file
    is looked up by the NORMALIZED isbn -- write_evaluation_entry always
    writes under it, but a manifest filename stem (the raw lookup key)
    could in principle be a hyphenated or differently-cased variant of
    the same ISBN."""
    results = []
    no_coverage = []
    for book in corpus.load_manifest_books():
        key = corpus.manifest_key(book)
        if not corpus.expected_json_path(key).exists():
            continue
        eval_path = corpus.evaluation_json_path(crossref.normalize_isbn(key) or key)
        if not eval_path.exists():
            no_coverage.append(key)
            continue
        gt_entries = _load_entries(corpus.expected_json_path(key))
        crossref_entries = _load_entries(eval_path)
        results.append(evaluate_book(key, gt_entries, crossref_entries))
    return results, no_coverage


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--full", action="store_true",
        help="Print a per-book precision/recall/F1 line for every compared book, not just the aggregate mean",
    )
    parser.add_argument(
        "--min-f1", type=float, default=None,
        help="Exit 1 if the aggregate mean F1 falls below this (0-1). Unset: no gate enforced.",
    )
    args = parser.parse_args()

    results, no_coverage = evaluate_corpus()
    if args.full:
        for r in results:
            print(f"[{r.key}] precision={r.precision:.0%} recall={r.recall:.0%} f1={r.f1:.0%} tp={r.tp} fp={r.fp} fn={r.fn}")
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

    if args.min_f1 is not None and (mean_f1 is None or mean_f1 < args.min_f1):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
