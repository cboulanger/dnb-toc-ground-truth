# Model comparison metrics for bulk-gate model selection

## Why

The bulk tier's two-model agreement gate (`matching.gate_books`) currently
gets its model pairing chosen without any corpus-level evidence about
those models specifically: which pairs actually agree with each other
often enough to clear the gate, and which models are individually closest
to truth. This adds two new corpus-level metrics -- inter-model
similarity, and closeness to ground truth -- plus a way to see them, so a
future re-pairing decision has real numbers behind it instead of a guess.

Both metrics are macro-averaged across books and always reported together
with the book count (N) they're based on, since coverage is uneven across
models (as of 2026-08-23: Qwen3-Omni 762 books, NuExtract3 461, Mistral
430) and a rate without its N is misleading.

## What changes

### 1. New library module: `src/dnb_toc_ground_truth/model_agreement.py`

Two pure functions, both operating over whatever models currently have
`llm-cache` entries (`corpus.llm_cache_dir()`, read via
`vision.load_cached_llm_entries`) -- no new caching, no network calls.

```python
@dataclass(frozen=True)
class PairAgreement:
    model_a: str
    model_b: str
    mean_agreement: float
    n_books: int

def pairwise_model_agreement(models: list[str]) -> list[PairAgreement]:
    """For every unordered pair in `models`, and every manifest book
    where BOTH have an llm-cache entry, computes matching.diff_toc_entries
    and agreement_rate = len(matched_pairs) / max(len(a), len(b)) -- the
    exact formula matching.gate_book already gates on. Macro-averages
    that rate across every book the pair shares. Returns one PairAgreement
    per pair with n_books > 0 (a pair sharing zero books is omitted, not
    reported as 0%)."""
```

```python
@dataclass(frozen=True)
class ModelGroundTruthMetrics:
    model: str
    precision: float
    recall: float
    f1: float
    n_books: int

def arbitration_ground_truth_agreement(models: list[str]) -> list[ModelGroundTruthMetrics]:
    """For every model, and every manifest book whose ground truth has
    "source": "agent_arbitration" (verified: true -- Claude-transcribed
    directly from the TOC page images, independent of any model's own
    raw reading regardless of whether that model was ever part of a
    bulk_gate pair elsewhere in the corpus, so there is no circularity
    risk in using the FULL arbitration set rather than just the eval-tier
    subset of it), compares that model's raw llm-cache entries against
    the arbitrated ground truth's entries via matching.diff_toc_entries.

    Unlike crossref_evaluation.evaluate_book (which only knows about real
    chapters), this compares ALL entries including skip:true ones -- it
    measures raw TOC-line extraction fidelity, not chapter
    classification, the same scope the bulk/eval tier convention already
    treats as directly comparable (see evaluation/README.md's "Bulk
    tier"/"Eval tier" sections in the chapter-segmentation repo this
    project was extracted from).

    TP = matched, FN = only_in_gt, FP = only_in_model; precision/recall/F1
    from there, macro-averaged across the model's covered books. A model
    with zero qualifying books is omitted, not reported as 0%."""
```

`discover_cached_models()` (already public in `crossref_evaluation.py`,
but scoped there to Crossref-sample books only) needs a corpus-wide
sibling here -- reuse the existing filename-parsing logic
(`vision.versioned_cache_dir(...).glob(f"{key}.*.json")`) without the
Crossref-eval-path filter, e.g. `model_agreement.discover_all_cached_models()`.

### 2. New CLI script: `cli/compare_models.py`

Mirrors `evaluate_crossref.py`'s argparse/structure. Corpus-level only --
no `--full` per-book dump (not needed for model selection). Prints:

- A pairwise agreement matrix (models x models, upper triangle, each
  cell `NN% (n=NNN)`).
- A per-model table: arbitration-GT precision/recall/F1/N, plus a
  Crossref precision/recall/F1/N column reusing
  `crossref_evaluation.evaluate_model_corpus` unmodified (already
  approved as a third reference column) -- one row per model, all three
  reference sources side by side.

```bash
uv run python cli/compare_models.py
uv run python cli/compare_models.py --all-models   # default; explicit --model also supported, same convention as evaluate_crossref.py
```

### 3. GitHub Pages site: new page `<corpus>-model-comparison.html`

One page per corpus, generated inside `write_site()`'s existing
per-corpus loop (same `corpus.set_corpus(name)` iteration that already
writes `<name>.html`, same `_page`/`_STYLE`/`_th`/`_link` helpers -- no
new JS or chart library, consistent with the site's current plain-HTML
approach) -- only one corpus (`pilot`) exists today, but the metrics are
as corpus-scoped as the Crossref page already is, so this follows that
precedent rather than assuming there will only ever be one. Each
corpus's existing `<name>.html` gains a link to its
`<name>-model-comparison.html` sibling (and back), and `index.html`'s
per-corpus list links both.

- **Pairwise agreement heatmap**: an HTML table, one row/column per
  model, each cell background-colored by `mean_agreement` (white -> green
  gradient, computed in Python as an inline `style="background:..."`,
  same "the number is always printed, color is a secondary read" stance
  the existing site treats data hardness with) and showing `NN% (n=NNN)`
  as text -- N is never hidden behind color alone. Diagonal cells blank
  (a model isn't compared against itself).
- **Per-model accuracy table**: one row per model, columns grouped
  Arbitration-GT (P/R/F1/N) then Crossref (P/R/F1/N), sortable-by-F1 same
  as the existing overview table's row order, each F1 cell also carrying
  a small inline-CSS bar (`<div style="width:{f1:.0%}">`) for a quick
  visual read alongside the number.
- A short caveats block (matching the existing Crossref page's
  `.caveats` box) noting: agreement measures how much two models say the
  same thing, not whether either is right (two models sharing the same
  systematic blind spot would still show high agreement); arbitration-GT
  coverage differs per model and grows over time as more books get
  arbitrated; Crossref coverage is small and skewed (existing caveat,
  unchanged).

### 4. README

`README.md` gains a short "Model comparison" section (after "Crossref
evaluation", same tier of permanence -- this describes a repeatable
procedure, not a point-in-time snapshot) covering what the two metrics
mean, how to run `compare_models.py`, and a link to the new site page.
Point-in-time numbers themselves are not hand-written into README (same
convention Crossref evaluation already follows -- the site is
auto-regenerated, README only describes the procedure).

## Non-goals

- No automated model **re-pairing** of the bulk gate itself -- this
  produces the evidence a human (or a future follow-up) uses to decide
  that; `matching.gate_books` is unchanged.
- No per-book output from the new CLI or site page (explicitly descoped
  -- corpus-level only).
- No new caching or LLM calls -- both metrics only ever read data that
  `generate_ground_truth.py`/`arbitrate.py` already wrote.

## Open questions / risks

- **Arbitration-GT coverage will be uneven and will grow over time** as
  more of the corpus gets arbitrated -- the reported N per model should
  make this visible rather than something a reader has to intuit from
  the "Current status" table separately.
- **Agreement is not accuracy.** Two models with a shared systematic
  blind spot (e.g. both mis-parsing multi-column TOC layouts the same
  way) would show high pairwise agreement while both being wrong --
  this is exactly why the arbitration-GT metric exists as a separate,
  independent check rather than agreement being used alone to pick a
  pair. Called out explicitly in the site's caveats block (section 3)
  rather than left implicit.
