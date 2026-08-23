"""Corpus-level metrics for bulk-gate model selection: how similar two
models' raw TOC readings are to each other, how close each model is to
arbitration-sourced ground truth, and a derived score ranking candidate
pairs -- see design spec
docs/superpowers/specs/2026-08-23-model-comparison-metrics-design.md.
Every function here only ever reads already-committed data (llm-cache/,
ground-truth/*.expected.json) -- no new caching, no network calls."""

import itertools
from dataclasses import dataclass

from dnb_toc_ground_truth import corpus, matching, vision


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
