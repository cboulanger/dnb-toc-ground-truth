"""Corpus-level metrics for bulk-gate model selection: how similar two
models' raw TOC readings are to each other, how close each model is to
arbitration-sourced ground truth, and a derived score ranking candidate
pairs -- see design spec
docs/superpowers/specs/2026-08-23-model-comparison-metrics-design.md.
Every function here only ever reads already-committed data (llm-cache/,
ground-truth/*.expected.json) -- no new caching, no network calls."""

import itertools
import json
from dataclasses import dataclass

from dnb_toc_ground_truth import corpus, matching, vision
from dnb_toc_ground_truth.toc_entry import TocEntry, _parse_toc_page_number


@dataclass(frozen=True)
class PairAgreement:
    model_a: str
    model_b: str
    mean_agreement: float
    n_books: int


def discover_all_cached_models() -> list[str]:
    """Every distinct (sanitized) model id with at least one llm-cache
    entry anywhere in the currently-selected corpus, regardless of
    Crossref-sample coverage (unlike
    crossref_evaluation.discover_cached_models, which is scoped to
    Crossref-sample books only). Cache filenames are
    "<key>.<safe_model>.json" (vision.cache_path) -- the manifest key
    never contains a dot, so splitting each filename stem on its FIRST
    dot recovers the full sanitized model id even when the model id
    itself contains one (e.g. "mistralai__Mistral-Small-3.2-24B-Instruct-2506")."""
    cache_dir = vision.versioned_cache_dir(corpus.llm_cache_dir())
    if not cache_dir.exists():
        return []
    models: set[str] = set()
    for path in cache_dir.glob("*.json"):
        stem = path.name[: -len(".json")]
        _key, _sep, model = stem.partition(".")
        models.add(model)
    return sorted(models)


def pairwise_model_agreement(models: list[str]) -> list[PairAgreement]:
    """For every unordered pair in `models`, and every manifest book
    where BOTH have an llm-cache entry, computes matching.diff_toc_entries
    and agreement_rate = len(matched_pairs) / max(len(a), len(b)) -- the
    exact formula matching.gate_book already gates on. Macro-averages
    that rate across every book the pair shares. Returns one
    PairAgreement per pair with n_books > 0 -- a pair sharing zero books
    is omitted entirely, not reported as 0% (0% would wrongly imply they
    disagree on everything, when in fact there is nothing to compare)."""
    keys = [corpus.manifest_key(book) for book in corpus.load_manifest_books()]
    cache_dir = corpus.llm_cache_dir()
    results = []
    for model_a, model_b in itertools.combinations(sorted(models), 2):
        rates = []
        for key in keys:
            entries_a = vision.load_cached_llm_entries(cache_dir, key, model_a)
            entries_b = vision.load_cached_llm_entries(cache_dir, key, model_b)
            if entries_a is None or entries_b is None:
                continue
            matched_pairs, _, _ = matching.diff_toc_entries(entries_a, entries_b)
            rates.append(len(matched_pairs) / max(len(entries_a), len(entries_b)))
        if rates:
            results.append(PairAgreement(
                model_a=model_a, model_b=model_b,
                mean_agreement=sum(rates) / len(rates), n_books=len(rates),
            ))
    return results


@dataclass(frozen=True)
class ModelGroundTruthMetrics:
    model: str
    precision: float
    recall: float
    f1: float
    n_books: int


def _entries_from_dicts(entries: list[dict]) -> list[TocEntry]:
    return [
        TocEntry(
            title=e["title"], authors=tuple(e.get("authors", [])),
            printed_page_number=e["printed_page_number"], source_page_index=-1, skip=e.get("skip", False),
        )
        for e in entries
    ]


def _page_sort_key(entry: TocEntry) -> tuple:
    value = _parse_toc_page_number(entry.printed_page_number) if entry.printed_page_number else None
    return (entry.printed_page_number is None, value if value is not None else 0, entry.printed_page_number or "")


def arbitration_ground_truth_agreement(models: list[str]) -> list[ModelGroundTruthMetrics]:
    """For every model, and every manifest book whose ground truth has
    "source": "agent_arbitration" (verified: true -- Claude-transcribed
    directly from the TOC page images, independent of any model's own
    raw reading, so there is no circularity risk in using this corpus-wide
    set rather than just the eval-tier subset of it), compares that
    model's raw llm-cache entries against the arbitrated ground truth's
    entries via matching.diff_toc_entries.

    Unlike crossref_evaluation.evaluate_book (which only knows about real
    chapters), this compares ALL entries including skip:true ones -- it
    measures raw TOC-line extraction fidelity, not chapter classification.

    TP = matched, FN = only_in_gt, FP = only_in_model; precision/recall/F1
    from there, macro-averaged across the model's covered books. A model
    with zero qualifying books is omitted, not reported as 0%."""
    books = corpus.load_manifest_books()
    cache_dir = corpus.llm_cache_dir()
    results = []
    for model in sorted(models):
        book_scores = []
        for book in books:
            key = corpus.manifest_key(book)
            gt_path = corpus.expected_json_path(key)
            if not gt_path.exists():
                continue
            gt_data = json.loads(gt_path.read_text(encoding="utf-8"))
            if gt_data.get("source") != "agent_arbitration":
                continue
            model_entries = vision.load_cached_llm_entries(cache_dir, key, model)
            if model_entries is None:
                continue
            gt_entries = sorted(_entries_from_dicts(gt_data["entries"]), key=_page_sort_key)
            model_sorted = sorted(model_entries, key=_page_sort_key)
            matched_pairs, only_in_gt, only_in_model = matching.diff_toc_entries(gt_entries, model_sorted)
            tp, fn, fp = len(matched_pairs), len(only_in_gt), len(only_in_model)
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            book_scores.append((precision, recall, f1))
        if book_scores:
            n = len(book_scores)
            results.append(ModelGroundTruthMetrics(
                model=model,
                precision=sum(s[0] for s in book_scores) / n,
                recall=sum(s[1] for s in book_scores) / n,
                f1=sum(s[2] for s in book_scores) / n,
                n_books=n,
            ))
    return results


@dataclass(frozen=True)
class PairCandidateScore:
    model_a: str
    model_b: str
    f1_a: float
    f1_b: float
    observed_agreement: float
    expected_agreement: float
    kappa: float
    score: float
    n_books: int


def rank_candidate_pairs(
    agreements: list[PairAgreement], gt_metrics: list[ModelGroundTruthMetrics],
) -> tuple[list[PairCandidateScore], list[tuple[str, str]]]:
    """Combines pairwise_model_agreement's output with
    arbitration_ground_truth_agreement's per-model F1 (treated as each
    model's per-entry "probability of being correct") into a Cohen's
    kappa: expected_agreement = f1_a*f1_b + (1-f1_a)*(1-f1_b) is the
    agreement rate two models with these individual accuracies would
    show if their errors were independent; kappa = (observed - expected)
    / (1 - expected) is how much OBSERVED agreement exceeds that
    baseline. kappa ~ 0 means genuine independence; kappa well above 0
    means they agree more than their accuracy alone explains -- a
    direct, quantified signal of correlated errors (same architecture
    family, shared training data, same systematic misreading of some
    layout). score = min(f1_a, f1_b) - max(0.0, kappa) rewards
    individually-accurate models and penalizes only EXCESS correlation
    (kappa below 0 isn't the failure mode this guards against).

    Only pairs where BOTH models have an arbitration-GT F1 (gt_metrics)
    are scored -- a pair missing coverage for either model is returned
    in the second list instead, by (model_a, model_b) name, never
    silently dropped or scored with a fabricated stand-in accuracy.
    Guards the `expected_agreement == 1.0` degenerate case (both models
    at F1 0.0 or both at F1 1.0) by defining kappa=0.0 there rather than
    dividing by zero -- there is no "excess" to measure when the
    baseline already claims total agreement is expected.

    Returns (scored_pairs sorted by score descending, unscored_pairs)."""
    f1_by_model = {m.model: m.f1 for m in gt_metrics}
    scored = []
    unscored = []
    for pair in agreements:
        f1_a = f1_by_model.get(pair.model_a)
        f1_b = f1_by_model.get(pair.model_b)
        if f1_a is None or f1_b is None:
            unscored.append((pair.model_a, pair.model_b))
            continue
        expected = f1_a * f1_b + (1 - f1_a) * (1 - f1_b)
        kappa = 0.0 if expected >= 1.0 else (pair.mean_agreement - expected) / (1 - expected)
        scored.append(PairCandidateScore(
            model_a=pair.model_a, model_b=pair.model_b, f1_a=f1_a, f1_b=f1_b,
            observed_agreement=pair.mean_agreement, expected_agreement=expected,
            kappa=kappa, score=min(f1_a, f1_b) - max(0.0, kappa), n_books=pair.n_books,
        ))
    scored.sort(key=lambda s: s.score, reverse=True)
    return scored, unscored
