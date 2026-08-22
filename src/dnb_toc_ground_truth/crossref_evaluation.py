"""Measures precision/recall/F1 between this corpus's own ground truth
(or an individual llm-cache model's raw extraction) and its committed
Crossref evaluation corpus (data/corpus/pilot/evaluation/<key>.expected.json)
-- see design spec
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
GT match) -- standard precision/recall/F1 from there.

evaluate_model_corpus scores one llm-cache model's raw, pre-gate/
pre-arbitration extraction (cli/generate_ground_truth.py's cache under
data/corpus/pilot/llm-cache/) against the same crossref-sample books, for
whichever of those books that model has a cache entry for --
dnb_toc_ground_truth.vision.load_cached_llm_entries already knows how to
read that cache, so this module reuses it unmodified rather than
re-implementing cache loading here.

Kept as a library module (not inline in cli/evaluate_crossref.py) so more
than one script -- the CLI and cli/generate_evaluation_site.py -- can
compute the same scores without importing one script from another (same
reasoning as vision.py's own cache-loading functions)."""

import json
from dataclasses import dataclass
from pathlib import Path

from dnb_toc_ground_truth import corpus, crossref, matching, vision
from dnb_toc_ground_truth.inference import ModelEndpoint
from dnb_toc_ground_truth.nuextract import nuextract_vision_extract_toc_entries
from dnb_toc_ground_truth.toc_entry import TocEntry, _parse_toc_page_number
from dnb_toc_ground_truth.vision import vision_extract_toc_entries


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


def evaluate_model_corpus(model: str) -> tuple[list[BookMetrics], list[str]]:
    """Returns (results, keys_with_no_cache_entry) -- for every manifest
    book with a committed Crossref evaluation-corpus entry, compares
    MODEL's cached llm-cache extraction (cli/generate_ground_truth.py's
    raw per-model output, before the two-model agreement gate or
    arbitration) against it. Lets a model's raw extraction quality be
    scored against the same independent Crossref signal used for ground
    truth, without running the full pipeline. A book with no cache entry
    for this model is listed in keys_with_no_cache_entry rather than
    silently dropped -- mirrors evaluate_corpus's own no_coverage list. A
    book with no Crossref evaluation-corpus entry at all is skipped
    entirely (not a meaningful comparison either way), same as
    evaluate_corpus does for ground truth."""
    results = []
    no_cache = []
    for book in corpus.load_manifest_books():
        key = corpus.manifest_key(book)
        eval_path = corpus.evaluation_json_path(crossref.normalize_isbn(key) or key)
        if not eval_path.exists():
            continue
        model_entries = vision.load_cached_llm_entries(corpus.llm_cache_dir(), key, model)
        if model_entries is None:
            no_cache.append(key)
            continue
        crossref_entries = _load_entries(eval_path)
        results.append(evaluate_book(key, tuple(model_entries), crossref_entries))
    return results, no_cache


async def backfill_model_cache(
    model: str, endpoint: ModelEndpoint, cache_directory: Path,
) -> tuple[list[str], list[str]]:
    """For every Crossref-evaluation-corpus book missing a llm-cache
    entry for `model` (reuses evaluate_model_corpus's own no_cache list
    -- the exact same book-selection logic scoring already trusts),
    extracts once via `endpoint` and writes the cache entry. Returns
    (succeeded_keys, failed_keys) -- a failure (missing PDF, network
    error, unparseable response, empty result) is printed and skipped,
    not retried; this is a manual one-off utility run, not
    generate_ground_truth.py's unattended batch job. `model` and
    `endpoint.model_id` are expected to match (the caller resolved
    `endpoint` FOR this `model`); kept as two separate parameters rather
    than reading `endpoint.model_id` directly so the cache is written
    under exactly the model id the caller/CLI asked to backfill, not
    whatever string the endpoints file happened to resolve it to."""
    _, missing_keys = evaluate_model_corpus(model)
    succeeded, failed = [], []
    for key in missing_keys:
        pdf_path = corpus.pdf_path(key)
        if not pdf_path.exists():
            print(f"[backfill] {key}: skipped, no PDF at {pdf_path}")
            failed.append(key)
            continue
        try:
            if endpoint.extraction_api == "nuextract":
                entries = await nuextract_vision_extract_toc_entries(
                    pdf_path, endpoint.model_id, endpoint.client,
                    use_instructions=endpoint.extraction_instructions,
                )
            else:
                entries = await vision_extract_toc_entries(pdf_path, endpoint.model_id, endpoint.client)
        except Exception as exc:  # noqa: BLE001 -- one book's failure must not abort the whole backfill
            print(f"[backfill] {key}: failed -- {type(exc).__name__}: {exc}")
            failed.append(key)
            continue
        if entries:
            vision.write_cached_llm_entries(cache_directory, key, model, entries, kind="vision")
            succeeded.append(key)
        else:
            print(f"[backfill] {key}: extraction returned no entries, not cached")
            failed.append(key)
    return succeeded, failed


def discover_cached_models() -> list[str]:
    """Every distinct model id with at least one llm-cache entry for a
    book that also has a committed Crossref evaluation-corpus entry --
    lets a caller (--all-models, or the evaluation-site generator) run
    without already knowing which model ids exist (they vary run to run
    as generate_ground_truth.py is pointed at new endpoints). Ids are
    read back from the cache filename's sanitized model segment
    (vision.cache_path's "/" -> "__"), same value evaluate_model_corpus
    needs -- passing it straight back in round-trips through
    cache_path's sanitization as a no-op."""
    models: set[str] = set()
    cache_dir = vision.versioned_cache_dir(corpus.llm_cache_dir())
    for book in corpus.load_manifest_books():
        key = corpus.manifest_key(book)
        eval_path = corpus.evaluation_json_path(crossref.normalize_isbn(key) or key)
        if not eval_path.exists():
            continue
        for path in cache_dir.glob(f"{key}.*.json"):
            models.add(path.name[len(key) + 1 : -len(".json")])
    return sorted(models)
