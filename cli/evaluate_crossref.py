#!/usr/bin/env python3
"""Measures agreement between this corpus's own ground truth and each
book's cached Crossref chapter data -- see design spec
docs/superpowers/specs/2026-08-21-crossref-cross-validation-design.md.

For every book with both a .expected.json and a cached
.crossref-cache/<isbn>.crossref.json carrying at least one chapter,
compares the ground truth's real chapters ("skip": false) against the
Crossref chapter list via matching.diff_toc_entries -- reused completely
unmodified, since it already aligns on title (chapter-number-prefix and
capitalization normalized) and first-page-number equivalence, exactly
this script's comparison spec. Books with no cached Crossref data at all
are reported separately, not silently dropped.

Usage:
    uv run python cli/evaluate_crossref.py
    uv run python cli/evaluate_crossref.py --min-agreement 0.8
"""

import argparse
import json
from dataclasses import dataclass

from dnb_toc_ground_truth import corpus, crossref, matching
from dnb_toc_ground_truth.crossref import CrossrefBookData, _load_cache as _load_crossref_cache
from dnb_toc_ground_truth.toc_entry import TocEntry, _parse_toc_page_number


@dataclass(frozen=True)
class BookAgreement:
    key: str
    matched: int
    only_in_gt: int
    only_in_crossref: int
    agreement_rate: float


def _load_gt_entries(key: str) -> tuple[TocEntry, ...]:
    data = json.loads(corpus.expected_json_path(key).read_text(encoding="utf-8"))
    return tuple(
        TocEntry(
            title=e["title"], authors=tuple(e.get("authors", [])),
            printed_page_number=e["printed_page_number"], source_page_index=-1, skip=e.get("skip", False),
        )
        for e in data["entries"]
    )


def _load_crossref_data(key: str) -> CrossrefBookData | None:
    return _load_crossref_cache(corpus.crossref_cache_dir(), crossref.normalize_isbn(key) or key)


def _page_sort_key(entry: TocEntry) -> tuple:
    # matching.diff_toc_entries' underlying align_toc_entries does a
    # single greedy, order-preserving scan ("TOC order is book order" --
    # true for two independent reads of the same printed TOC, which is
    # all this function was built to diff). Crossref's /works response
    # order is NOT page order (confirmed empirically, 2026-08-21,
    # isbn:9783111702681: sorting its chapter list by page before
    # diffing took that book's match count from 5/20 to 20/20) -- so
    # both sides are re-sorted into page order here before ever reaching
    # diff_toc_entries, restoring the precondition that function assumes
    # without needing to change it. Mirrors matching.gate_book's own
    # merge-output sort key exactly (unknown page sorts last).
    value = _parse_toc_page_number(entry.printed_page_number) if entry.printed_page_number else None
    return (entry.printed_page_number is None, value if value is not None else 0, entry.printed_page_number or "")


def evaluate_book(key: str, gt_entries: tuple[TocEntry, ...], crossref_data: CrossrefBookData) -> BookAgreement:
    gt_real = sorted((e for e in gt_entries if not e.skip), key=_page_sort_key)
    crossref_chapters = sorted(crossref_data.chapters, key=_page_sort_key)
    matched_pairs, only_in_gt, only_in_crossref = matching.diff_toc_entries(gt_real, crossref_chapters)
    denominator = max(len(gt_real), len(crossref_data.chapters))
    agreement_rate = len(matched_pairs) / denominator if denominator else 0.0
    return BookAgreement(
        key=key, matched=len(matched_pairs), only_in_gt=len(only_in_gt),
        only_in_crossref=len(only_in_crossref), agreement_rate=agreement_rate,
    )


def evaluate_corpus() -> tuple[list[BookAgreement], list[str]]:
    """Returns (results, keys_with_no_crossref_coverage) for every
    manifest book that has a .expected.json."""
    results = []
    no_coverage = []
    for book in corpus.load_manifest_books():
        key = corpus.manifest_key(book)
        if not corpus.expected_json_path(key).exists():
            continue
        crossref_data = _load_crossref_data(key)
        if crossref_data is None or not crossref_data.chapters:
            no_coverage.append(key)
            continue
        gt_entries = _load_gt_entries(key)
        results.append(evaluate_book(key, gt_entries, crossref_data))
    return results, no_coverage


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--min-agreement", type=float, default=None,
        help="Exit 1 if the aggregate mean agreement rate falls below this (0-1). Unset: no gate enforced.",
    )
    args = parser.parse_args()

    results, no_coverage = evaluate_corpus()
    for result in results:
        print(
            f"[{result.key}] agreement={result.agreement_rate:.0%} "
            f"matched={result.matched} only_in_gt={result.only_in_gt} only_in_crossref={result.only_in_crossref}"
        )

    if results:
        mean_agreement = sum(r.agreement_rate for r in results) / len(results)
        print(f"\n{len(results)} book(s) compared, mean agreement {mean_agreement:.0%}")
    else:
        mean_agreement = None
        print("\nNo books had both ground truth and cached Crossref chapter data.")
    print(f"{len(no_coverage)} book(s) with ground truth but no Crossref coverage.")

    if args.min_agreement is not None and (mean_agreement is None or mean_agreement < args.min_agreement):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
