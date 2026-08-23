# Model Comparison Metrics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two corpus-level metrics (inter-model TOC-reading agreement,
closeness to arbitration-sourced ground truth) plus a derived
candidate-pair ranking (Cohen's kappa) to inform which two models to pair
for the bulk-tier two-model agreement gate, exposed via a new CLI script
and a new GitHub Pages section.

**Architecture:** A new pure-function library module
(`src/dnb_toc_ground_truth/model_agreement.py`) computes all three
metrics by reading only already-committed data (`llm-cache/`,
`ground-truth/*.expected.json`) -- no new caching, no network calls. A
new CLI script (`cli/compare_models.py`) prints them. `evaluation_site.py`
gains a new per-corpus page rendering the same data as an HTML heatmap +
tables, wired into the existing `write_site()` entry point.

**Tech Stack:** Python 3, stdlib `dataclasses`/`itertools`/`json`,
`rapidfuzz` (via `matching.diff_toc_entries`, unchanged), `unittest`
(project's existing test convention).

**Design spec:** `docs/superpowers/specs/2026-08-23-model-comparison-metrics-design.md`

---

## Task 1: `PairAgreement` + `pairwise_model_agreement` + `discover_all_cached_models`

**Files:**
- Create: `src/dnb_toc_ground_truth/model_agreement.py`
- Test: `tests/test_model_agreement.py`

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_model_agreement.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dnb_toc_ground_truth.model_agreement'`

- [ ] **Step 3: Implement `model_agreement.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_model_agreement.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add src/dnb_toc_ground_truth/model_agreement.py tests/test_model_agreement.py
git commit -m "feat: add pairwise inter-model TOC-agreement metric"
```

---

## Task 2: `ModelGroundTruthMetrics` + `arbitration_ground_truth_agreement`

**Files:**
- Modify: `src/dnb_toc_ground_truth/model_agreement.py`
- Modify: `tests/test_model_agreement.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_model_agreement.py` (add these imports to the
existing `from dnb_toc_ground_truth.model_agreement import (...)` line:
`ModelGroundTruthMetrics`, `arbitration_ground_truth_agreement`):

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_model_agreement.py -v`
Expected: FAIL with `ImportError: cannot import name 'ModelGroundTruthMetrics'`

- [ ] **Step 3: Implement**

Add to `src/dnb_toc_ground_truth/model_agreement.py` (new imports plus
new dataclass/function):

```python
from dnb_toc_ground_truth.toc_entry import TocEntry, _parse_toc_page_number
```

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_model_agreement.py -v`
Expected: PASS (12 tests)

- [ ] **Step 5: Commit**

```bash
git add src/dnb_toc_ground_truth/model_agreement.py tests/test_model_agreement.py
git commit -m "feat: add arbitration-ground-truth closeness metric per model"
```

---

## Task 3: `PairCandidateScore` + `rank_candidate_pairs` (Cohen's kappa)

**Files:**
- Modify: `src/dnb_toc_ground_truth/model_agreement.py`
- Modify: `tests/test_model_agreement.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_model_agreement.py` (add `PairCandidateScore`,
`rank_candidate_pairs` to the existing import line):

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_model_agreement.py -v`
Expected: FAIL with `ImportError: cannot import name 'PairCandidateScore'`

- [ ] **Step 3: Implement**

Add to `src/dnb_toc_ground_truth/model_agreement.py`:

```python
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
    means the two models agree more than their accuracy alone explains --
    a quantified sign of correlated errors (e.g. same architecture
    family). score = min(f1_a, f1_b) - max(0.0, kappa) rewards
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_model_agreement.py -v`
Expected: PASS (17 tests)

- [ ] **Step 5: Commit**

```bash
git add src/dnb_toc_ground_truth/model_agreement.py tests/test_model_agreement.py
git commit -m "feat: add Cohen's-kappa candidate-pair ranking"
```

---

## Task 4: `cli/compare_models.py`

**Files:**
- Create: `cli/compare_models.py`
- Test: `tests/test_compare_models.py`

- [ ] **Step 1: Write the failing test**

```python
"""Smoke test for cli/compare_models.py -- the actual metrics are
tested at the library level in tests/test_model_agreement.py; this only
checks main() runs end-to-end against a tiny corpus without crashing
and prints the expected section headers."""

import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cli"))

from compare_models import main

from dnb_toc_ground_truth import corpus, vision
from dnb_toc_ground_truth.toc_entry import TocEntry


class TestMain(unittest.TestCase):
    def test_runs_end_to_end_and_prints_all_three_sections(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            corpus.manifest_path().write_text(
                json.dumps({"toc_only": True, "books": [{"filename": "111.pdf", "doi": None}]}),
                encoding="utf-8",
            )
            corpus.ground_truth_dir().mkdir(parents=True, exist_ok=True)
            corpus.expected_json_path("111").write_text(
                json.dumps({
                    "entries": [{"title": "1. Introduction", "authors": [], "printed_page_number": "1", "skip": False}],
                    "verified": True, "source": "agent_arbitration",
                }),
                encoding="utf-8",
            )
            entries = [TocEntry(title="1. Introduction", printed_page_number="1", source_page_index=0)]
            vision.write_cached_llm_entries(corpus.llm_cache_dir(), "111", "model/a", entries)
            vision.write_cached_llm_entries(corpus.llm_cache_dir(), "111", "model/b", entries)

            out = StringIO()
            with patch("sys.argv", ["compare_models.py", "--corpus", Path(tmp).name]), redirect_stdout(out):
                with patch.object(corpus, "_CORPUS_ROOT", Path(tmp).parent):
                    exit_code = main()

            self.assertEqual(exit_code, 0)
            output = out.getvalue()
            self.assertIn("Pairwise agreement", output)
            self.assertIn("Per-model accuracy", output)
            self.assertIn("Candidate pair ranking", output)

    def test_reports_no_cached_models_without_crashing(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            corpus.manifest_path().write_text(
                json.dumps({"toc_only": True, "books": []}), encoding="utf-8",
            )
            out = StringIO()
            with patch("sys.argv", ["compare_models.py", "--corpus", Path(tmp).name]), redirect_stdout(out):
                with patch.object(corpus, "_CORPUS_ROOT", Path(tmp).parent):
                    exit_code = main()
            self.assertEqual(exit_code, 0)
            self.assertIn("No cached model readings found", out.getvalue())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_compare_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'compare_models'`

- [ ] **Step 3: Implement `cli/compare_models.py`**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_compare_models.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add cli/compare_models.py tests/test_compare_models.py
git commit -m "feat: add cli/compare_models.py"
```

---

## Task 5: GitHub Pages page -- data collection + rendering

**Files:**
- Modify: `src/dnb_toc_ground_truth/evaluation_site.py`
- Modify: `tests/test_evaluation_site.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_evaluation_site.py`'s imports:

```python
from dnb_toc_ground_truth.evaluation_site import (
    CorpusData,
    ModelComparisonData,
    SourceScores,
    collect_corpus_data,
    collect_model_comparison_data,
    render_corpus_html,
    render_index_html,
    render_model_comparison_html,
    write_site,
)
from dnb_toc_ground_truth.model_agreement import ModelGroundTruthMetrics, PairAgreement, PairCandidateScore
```

Append new test classes:

```python
class TestCollectModelComparisonData(unittest.TestCase):
    def test_gathers_agreement_gt_metrics_and_crossref_for_every_cached_model(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            corpus.manifest_path().write_text(
                json.dumps({"toc_only": True, "books": [{"filename": "111.pdf", "doi": None}]}),
                encoding="utf-8",
            )
            corpus.ground_truth_dir().mkdir(parents=True, exist_ok=True)
            corpus.expected_json_path("111").write_text(
                json.dumps({
                    "entries": [{"title": "1. Introduction", "authors": [], "printed_page_number": "1", "skip": False}],
                    "verified": True, "source": "agent_arbitration",
                }),
                encoding="utf-8",
            )
            entries = [TocEntry(title="1. Introduction", printed_page_number="1", source_page_index=0)]
            vision.write_cached_llm_entries(corpus.llm_cache_dir(), "111", "model/a", entries)
            vision.write_cached_llm_entries(corpus.llm_cache_dir(), "111", "model/b", entries)

            data = collect_model_comparison_data()

            self.assertEqual(data.name, "pilot")
            self.assertEqual(data.models, ["model__a", "model__b"])
            self.assertEqual(len(data.agreements), 1)
            self.assertEqual(len(data.gt_metrics), 2)
            self.assertIn("model__a", data.crossref_results)
            self.assertEqual(len(data.scored_pairs), 1)
            self.assertEqual(data.unscored_pairs, [])

    def test_no_cached_models_yields_empty_data_without_crashing(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            corpus.manifest_path().write_text(json.dumps({"toc_only": True, "books": []}), encoding="utf-8")
            data = collect_model_comparison_data()
            self.assertEqual(data.models, [])
            self.assertEqual(data.agreements, [])


class TestRenderModelComparisonHtml(unittest.TestCase):
    def _sample_data(self) -> ModelComparisonData:
        return ModelComparisonData(
            name="pilot",
            models=["model__a", "model__b"],
            agreements=[PairAgreement(model_a="model__a", model_b="model__b", mean_agreement=0.9, n_books=5)],
            gt_metrics=[
                ModelGroundTruthMetrics(model="model__a", precision=0.8, recall=0.8, f1=0.8, n_books=10),
                ModelGroundTruthMetrics(model="model__b", precision=0.7, recall=0.7, f1=0.7, n_books=8),
            ],
            crossref_results={"model__a": ([], []), "model__b": ([], [])},
            scored_pairs=[PairCandidateScore(
                model_a="model__a", model_b="model__b", f1_a=0.8, f1_b=0.7,
                observed_agreement=0.9, expected_agreement=0.62, kappa=0.74, score=-0.04, n_books=5,
            )],
            unscored_pairs=[],
        )

    def test_renders_all_three_sections(self):
        html = render_model_comparison_html(self._sample_data())
        self.assertIn("Pairwise agreement", html)
        self.assertIn("Per-model accuracy", html)
        self.assertIn("Candidate pair ranking", html)
        self.assertIn("model__a", html)
        self.assertIn("model__b", html)

    def test_heatmap_cell_shows_rate_and_n(self):
        html = render_model_comparison_html(self._sample_data())
        self.assertIn("90% (n=5)", html)

    def test_no_models_renders_a_placeholder_without_crashing(self):
        data = ModelComparisonData(
            name="pilot", models=[], agreements=[], gt_metrics=[], crossref_results={},
            scored_pairs=[], unscored_pairs=[],
        )
        html = render_model_comparison_html(data)
        self.assertIn("No cached model readings found", html)

    def test_unscored_pairs_are_listed_separately(self):
        data = ModelComparisonData(
            name="pilot", models=["model__a", "model__b"],
            agreements=[PairAgreement(model_a="model__a", model_b="model__b", mean_agreement=0.9, n_books=5)],
            gt_metrics=[], crossref_results={}, scored_pairs=[],
            unscored_pairs=[("model__a", "model__b")],
        )
        html = render_model_comparison_html(data)
        self.assertIn("Unscored pairs", html)
        self.assertIn("model__a + model__b", html)

    def test_back_links_to_corpora_and_corpus_page(self):
        html = render_model_comparison_html(self._sample_data())
        self.assertIn('<a href="index.html">&larr; Corpora</a>', html)
        self.assertIn('<a href="pilot.html">&larr; pilot</a>', html)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_evaluation_site.py -v`
Expected: FAIL with `ImportError: cannot import name 'ModelComparisonData'`

- [ ] **Step 3: Implement**

Add these imports near the top of `src/dnb_toc_ground_truth/evaluation_site.py`
(alongside the existing `crossref_evaluation` import):

```python
from dnb_toc_ground_truth.model_agreement import (
    ModelGroundTruthMetrics,
    PairAgreement,
    PairCandidateScore,
    arbitration_ground_truth_agreement,
    discover_all_cached_models,
    pairwise_model_agreement,
    rank_candidate_pairs,
)
```

Add the dataclass and collector (near `CorpusData`/`collect_corpus_data`):

```python
@dataclass(frozen=True)
class ModelComparisonData:
    name: str
    models: list[str]
    agreements: list[PairAgreement]
    gt_metrics: list[ModelGroundTruthMetrics]
    crossref_results: dict[str, tuple[list[BookMetrics], list[str]]]
    scored_pairs: list[PairCandidateScore]
    unscored_pairs: list[tuple[str, str]]


def collect_model_comparison_data() -> ModelComparisonData:
    """Same "operates on corpus.py's CURRENTLY SELECTED corpus" contract
    as collect_corpus_data()."""
    models = discover_all_cached_models()
    agreements = pairwise_model_agreement(models)
    gt_metrics = arbitration_ground_truth_agreement(models)
    crossref_results = {model: evaluate_model_corpus(model) for model in models}
    scored_pairs, unscored_pairs = rank_candidate_pairs(agreements, gt_metrics)
    return ModelComparisonData(
        name=corpus.corpus_dir().name, models=models, agreements=agreements, gt_metrics=gt_metrics,
        crossref_results=crossref_results, scored_pairs=scored_pairs, unscored_pairs=unscored_pairs,
    )
```

Add rendering helpers (near `_render_source_section`/`_render_overview_table`):

```python
def _heatmap_color(rate: float) -> str:
    """White (0%) -> a mid green (100%), linearly interpolated -- the
    printed "NN% (n=NNN)" text is always the primary read, this is a
    secondary visual cue only."""
    r = round(255 + (46 - 255) * rate)
    g = round(255 + (125 - 255) * rate)
    b = round(255 + (50 - 255) * rate)
    return f"rgb({r},{g},{b})"


def _render_agreement_heatmap(models: list[str], agreements: list[PairAgreement]) -> str:
    by_pair = {frozenset((a.model_a, a.model_b)): a for a in agreements}
    header = "".join(f"<th>{_html.escape(m)}</th>" for m in models)
    rows = []
    for row_model in models:
        cells = []
        for col_model in models:
            if row_model == col_model:
                cells.append("<td></td>")
                continue
            pair = by_pair.get(frozenset((row_model, col_model)))
            if pair is None:
                cells.append('<td class="num">-</td>')
            else:
                color = _heatmap_color(pair.mean_agreement)
                cells.append(f'<td class="num" style="background:{color}">{pair.mean_agreement:.0%} (n={pair.n_books})</td>')
        rows.append(f"<tr><th>{_html.escape(row_model)}</th>{''.join(cells)}</tr>")
    return f"""<table>
<thead><tr><th></th>{header}</tr></thead>
<tbody>
{''.join(rows)}
</tbody>
</table>"""


def _render_model_accuracy_table(
    models: list[str], gt_metrics: list[ModelGroundTruthMetrics],
    crossref_results: dict[str, tuple[list[BookMetrics], list[str]]],
) -> str:
    gt_by_model = {m.model: m for m in gt_metrics}
    ordered = sorted(models, key=lambda m: -(gt_by_model[m].f1 if m in gt_by_model else -1.0))
    rows = []
    for model in ordered:
        gt = gt_by_model.get(model)
        gt_cells = (
            f'<td class="num">{gt.precision:.0%}</td><td class="num">{gt.recall:.0%}</td>'
            f'<td class="num">{gt.f1:.0%}</td><td class="num">{gt.n_books}</td>'
            if gt else '<td class="num">-</td><td class="num">-</td><td class="num">-</td><td class="num">-</td>'
        )
        cr_results, _ = crossref_results.get(model, ([], []))
        if cr_results:
            mean_f1 = _mean([r.f1 for r in cr_results])
            cr_cells = (
                f'<td class="num">{_mean([r.precision for r in cr_results]):.0%}</td>'
                f'<td class="num">{_mean([r.recall for r in cr_results]):.0%}</td>'
                f'<td class="num">{mean_f1:.0%}</td><td class="num">{len(cr_results)}</td>'
            )
        else:
            cr_cells = '<td class="num">-</td><td class="num">-</td><td class="num">-</td><td class="num">-</td>'
        bar_width = f"{gt.f1:.0%}" if gt else "0%"
        rows.append(
            f"<tr><td>{_html.escape(model)}</td>{gt_cells}{cr_cells}"
            f'<td><div style="width:{bar_width}; background:#2e7d32; height:0.5rem;"></div></td></tr>'
        )
    return f"""<table>
<thead><tr><th>Model</th><th class="num">Arb. P</th><th class="num">Arb. R</th><th class="num">Arb. F1</th>
<th class="num">Arb. N</th><th class="num">Crossref P</th><th class="num">Crossref R</th>
<th class="num">Crossref F1</th><th class="num">Crossref N</th><th>F1</th></tr></thead>
<tbody>
{''.join(rows)}
</tbody>
</table>"""


def _render_candidate_ranking_table(scored_pairs: list[PairCandidateScore], unscored_pairs: list[tuple[str, str]]) -> str:
    if not scored_pairs:
        table = "<p>No pair has arbitration-GT coverage for both models yet.</p>"
    else:
        rows = "\n".join(
            f"<tr><td>{_html.escape(p.model_a)}</td><td>{_html.escape(p.model_b)}</td>"
            f'<td class="num">{p.f1_a:.0%}</td><td class="num">{p.f1_b:.0%}</td>'
            f'<td class="num">{p.observed_agreement:.0%}</td><td class="num">{p.expected_agreement:.0%}</td>'
            f'<td class="num">{p.kappa:+.2f}</td><td class="num">{p.score:+.2f}</td><td class="num">{p.n_books}</td></tr>'
            for p in scored_pairs
        )
        table = f"""<table>
<thead><tr><th>Model A</th><th>Model B</th><th class="num">F1 A</th><th class="num">F1 B</th>
<th class="num">Observed</th><th class="num">Expected</th><th class="num">Kappa</th><th class="num">Score</th>
<th class="num">N</th></tr></thead>
<tbody>
{rows}
</tbody>
</table>"""
    if unscored_pairs:
        items = "".join(f"<li>{_html.escape(a)} + {_html.escape(b)}</li>" for a, b in unscored_pairs)
        table += f"<p>Unscored pairs (missing arbitration-GT coverage for one or both models):</p><ul>{items}</ul>"
    return table


def render_model_comparison_html(data: ModelComparisonData) -> str:
    back_links = (
        f'<p>{_link("index.html", "&larr; Corpora", new_tab=False)} '
        f'{_link(f"{data.name}.html", f"&larr; {_html.escape(data.name)}", new_tab=False)}</p>'
    )
    if not data.models:
        body = f"""{back_links}
<h1>{_html.escape(data.name)} -- Model comparison</h1>
<p>No cached model readings found for this corpus.</p>
"""
        return _page(f"{data.name} -- Model comparison", body)
    body = f"""{back_links}
<h1>{_html.escape(data.name)} -- Model comparison</h1>
<p>Corpus-level metrics for choosing which two models to pair for the
bulk-tier two-model agreement gate: how much two models' raw TOC
readings agree with each other, how close each model is to
arbitration-sourced ground truth (Claude-transcribed directly from the
TOC page images, independent of any model's own raw reading) and to an
independent Crossref cross-check, and a derived ranking of candidate
pairs.</p>
<div class="caveats">
<strong>Caveats -- read before trusting a number below:</strong>
<ul>
<li><strong>Agreement measures how much two models say the same
thing, not whether either is right.</strong> Two models sharing the
same systematic blind spot (e.g. the same architecture family) would
still show high agreement.</li>
<li><strong>A HIGH kappa is the specific red flag for that failure
mode</strong> -- agreement in excess of what each model's own
arbitration-GT accuracy already predicts under independent errors is
exactly why the candidate-pair ranking penalizes it rather than
simply sorting by raw agreement.</li>
<li><strong>Arbitration-GT coverage differs per model and grows over
time</strong> as more of the corpus gets arbitrated -- a low N means a
score is based on fewer books and can shift as more books are
arbitrated.</li>
<li><strong>Crossref coverage is small and skewed</strong> toward
larger, more prominent publishers -- see this corpus's main page for
the full caveat.</li>
</ul>
</div>
<h2>Pairwise agreement</h2>
{_render_agreement_heatmap(data.models, data.agreements)}
<h2>Per-model accuracy</h2>
{_render_model_accuracy_table(data.models, data.gt_metrics, data.crossref_results)}
<h2>Candidate pair ranking</h2>
{_render_candidate_ranking_table(data.scored_pairs, data.unscored_pairs)}
"""
    return _page(f"{data.name} -- Model comparison", body)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_evaluation_site.py -v`
Expected: PASS (all tests, including the new ones)

- [ ] **Step 5: Commit**

```bash
git add src/dnb_toc_ground_truth/evaluation_site.py tests/test_evaluation_site.py
git commit -m "feat: render model-comparison page (heatmap + accuracy + ranking)"
```

---

## Task 6: Wire the new page into `write_site()` and cross-link

**Files:**
- Modify: `src/dnb_toc_ground_truth/evaluation_site.py`
- Modify: `tests/test_evaluation_site.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_evaluation_site.py`:

```python
class TestWriteSiteModelComparison(unittest.TestCase):
    def test_writes_a_model_comparison_page_per_corpus(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _init_corpus(root, "pilot", [])
            with patch.object(corpus, "_CORPUS_ROOT", root), patch.object(corpus, "CORPUS_DIR", root / "pilot"):
                with tempfile.TemporaryDirectory() as out:
                    output_dir = Path(out) / "site"
                    write_site(output_dir)
                    self.assertTrue((output_dir / "pilot-model-comparison.html").exists())
```

Also update the EXISTING `test_every_off_site_link_opens_in_a_new_tab`
test (it currently excludes only `href="index.html"` and
`href="pilot.html"` from the "must open in a new tab" check -- the new
in-tab model-comparison link needs the same exclusion):

```python
    def test_every_off_site_link_opens_in_a_new_tab(self):
        html = render_corpus_html(self._sample_data())
        anchors = re.findall(r"<a\b[^>]*>", html)
        off_site = [
            a for a in anchors
            if 'href="index.html"' not in a
            and 'href="pilot.html"' not in a
            and 'href="pilot-model-comparison.html"' not in a
        ]
        self.assertTrue(off_site)
        for anchor in off_site:
            self.assertIn('target="_blank"', anchor)
            self.assertIn('rel="noopener noreferrer"', anchor)
```

And add one new assertion to `test_lists_each_corpus_with_a_link`:

```python
    def test_lists_each_corpus_with_a_link(self):
        html = render_index_html(["pilot"])
        self.assertIn('href="pilot.html"', html)
        self.assertIn(">pilot<", html)
        self.assertIn('href="pilot-model-comparison.html"', html)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_evaluation_site.py -v`
Expected: `test_writes_a_model_comparison_page_per_corpus` and
`test_lists_each_corpus_with_a_link` FAIL (page not written / link not
present yet); `test_every_off_site_link_opens_in_a_new_tab` currently
PASSES (no model-comparison link exists yet to break it) but will be
needed once Step 3 adds that link to `render_corpus_html`.

- [ ] **Step 3: Implement**

Modify `render_index_html` in `src/dnb_toc_ground_truth/evaluation_site.py`:

```python
def render_index_html(corpus_names: list[str]) -> str:
    links = "\n".join(
        f"<li>{_link(f'{name}.html', _html.escape(name), new_tab=False)} "
        f"({_link(f'{name}-model-comparison.html', 'model comparison', new_tab=False)})</li>"
        for name in corpus_names
    )
```

(rest of the function body is unchanged -- it already builds `body` from
`links` and returns `_page("Crossref evaluation", body)`.)

Modify `render_corpus_html` to add a link to the sibling page, right
after the existing `<p>{_link("index.html", "&larr; Corpora", ...)}</p>`
line:

```python
def render_corpus_html(data: CorpusData) -> str:
    overview = _render_overview_table(data.sources)
    sections = "\n".join(_render_source_section(source, data.titles, data.toc_urls) for source in data.sources)
    body = f"""<p>{_link("index.html", "&larr; Corpora", new_tab=False)}</p>
<h1>{_html.escape(data.name)}</h1>
<p>Per-book precision/recall/F1 against the committed Crossref evaluation
corpus. Ground truth is the project's own committed data (highlighted
below); each model section scores that model's raw, pre-agreement-gate
llm-cache extraction over the same books.</p>
<p>{_link(f"{data.name}-model-comparison.html", "Model comparison &rarr;", new_tab=False)}</p>
{overview}
{sections}
"""
    return _page(f"{data.name} -- Crossref evaluation", body)
```

Modify `write_site` to also write the new page per corpus:

```python
def write_site(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_names = corpus.list_corpora()
    previous = corpus.corpus_dir().name
    try:
        for name in corpus_names:
            corpus.set_corpus(name)
            (output_dir / f"{name}.html").write_text(render_corpus_html(collect_corpus_data()), encoding="utf-8")
            (output_dir / f"{name}-model-comparison.html").write_text(
                render_model_comparison_html(collect_model_comparison_data()), encoding="utf-8",
            )
    finally:
        corpus.set_corpus(previous)
    (output_dir / "index.html").write_text(render_index_html(corpus_names), encoding="utf-8")
    (output_dir / ".nojekyll").write_text("", encoding="utf-8")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_evaluation_site.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add src/dnb_toc_ground_truth/evaluation_site.py tests/test_evaluation_site.py
git commit -m "feat: wire model-comparison page into write_site and cross-link it"
```

---

## Task 7: README documentation

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add the "Model comparison" section**

Insert immediately after the existing "Crossref evaluation" section
(right before its `## Setup` heading, i.e. after the paragraph ending
"...don't read a lower per-model score as Crossref disagreeing with
this project's own ground truth."):

```markdown
## Model comparison

Beyond checking ground truth against Crossref, this project also
compares the models feeding the bulk-tier two-model agreement gate
(`matching.gate_books`) against each other, to inform which two models
to actually pair. Two raw metrics alone don't answer that question --
and naively optimizing either one in isolation is wrong:

- **Maximizing agreement is wrong.** Two models from a similar
  architecture/training lineage can agree with each other often while
  sharing the same systematic misreadings -- high agreement in that
  case reflects a shared blind spot, not a real independent check.
- **Minimizing agreement is equally wrong.** Two models that disagree a
  lot because one or both are simply unreliable aren't a good pairing
  either -- that's not healthy independence, it's noise.

What matters is the combination: each model individually close to the
truth, AND their agreement no higher than what that individual accuracy
alone would already predict if their errors were independent. This
project measures all three:

1. **Inter-model agreement** (`model_agreement.pairwise_model_agreement`)
   -- for every pair of models with cached readings for the same books,
   the same match-rate formula the bulk gate itself gates on
   (`matching.diff_toc_entries`), macro-averaged across every book the
   pair shares.
2. **Closeness to ground truth**
   (`model_agreement.arbitration_ground_truth_agreement`) -- each
   model's raw cached reading scored (precision/recall/F1) against
   every book whose ground truth came from full agent arbitration
   (`"source": "agent_arbitration"` -- Claude-transcribed directly from
   the TOC page images, independent of any model's own raw reading).
3. **A derived candidate-pair score**
   (`model_agreement.rank_candidate_pairs`) -- combines both via a
   Cohen's-kappa-style excess-agreement calculation: treating each
   model's arbitration-GT F1 as its per-entry "probability of being
   correct", `expected_agreement = f1_a*f1_b + (1-f1_a)*(1-f1_b)` is
   what two that-accurate-but-independent models would show by chance;
   `kappa = (observed - expected) / (1 - expected)` quantifies how much
   the OBSERVED agreement exceeds that -- well above 0 is a direct flag
   for correlated errors. `score = min(f1_a, f1_b) - max(0, kappa)`
   rewards individually-accurate models while penalizing only that
   excess correlation.

**Run it:**

```bash
uv run python cli/compare_models.py
```

**[View the current numbers on GitHub Pages](https://cboulanger.github.io/dnb-toc-ground-truth/pilot-model-comparison.html)**
-- a pairwise agreement heatmap, a per-model accuracy table (against
both arbitration-GT and Crossref), and the candidate-pair ranking,
rebuilt automatically on every push to `main` alongside the Crossref
pages.

**Constraints:**

- **Arbitration-GT coverage differs per model and grows over time** as
  more of the corpus gets arbitrated -- always read a score alongside
  its N, not in isolation.
- **The kappa calculation is a simplification, not a measured
  probability** -- good enough to flag an obvious excess, not precise
  enough to treat small kappa differences between two candidate pairs
  as decisive.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: document the model-comparison metrics and how to read them"
```

---

## Task 8: Full test suite + final review pass

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite**

Run: `uv run pytest`
Expected: all tests PASS, no failures or errors.

- [ ] **Step 2: Regenerate the site locally as a smoke check**

Run: `uv run python cli/generate_evaluation_site.py --output-dir /tmp/site-check`
Expected: exits 0; `/tmp/site-check/pilot-model-comparison.html` exists
and, opened in a browser, shows the heatmap/accuracy/ranking sections
without a Python traceback in the terminal output.

- [ ] **Step 3: Run `cli/compare_models.py` against the real corpus**

Run: `uv run python cli/compare_models.py`
Expected: exits 0; prints all three sections with real model ids
(Qwen/Mistral/NuExtract3) and non-empty data.
