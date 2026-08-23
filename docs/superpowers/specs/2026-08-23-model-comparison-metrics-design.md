# Model comparison metrics for bulk-gate model selection

## Why

The bulk tier's two-model agreement gate (`matching.gate_books`) currently
gets its model pairing chosen without any corpus-level evidence about
those models specifically. Neither raw metric alone answers the actual
question, and naively optimizing either one in isolation is actively
wrong:

- **Maximizing agreement alone is wrong.** Two models from a similar
  architecture/training lineage can agree with each other often while
  sharing the same systematic misreadings -- high agreement in that case
  reflects a shared blind spot, not a real independent check. The gate
  needs a *control*, and two near-identical readers don't provide one.
- **Minimizing agreement alone is equally wrong.** Two models that
  disagree a lot because one or both are simply unreliable aren't a good
  pairing either -- that's not healthy independence, it's noise.

What actually matters is the combination: **each model individually close
to the truth, AND their agreement no higher than what that individual
accuracy alone would already predict if their errors were independent.**
This adds three things -- inter-model agreement, closeness to ground
truth, and a derived score combining both -- plus a way to see them, so a
future re-pairing decision has real numbers behind it instead of a guess.

All three are macro-averaged across books and always reported together
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

#### Derived candidate-pair score (Cohen's kappa)

A third function combines the two metrics above into a single,
named-statistic-based signal for "is this pair's agreement suspicious
given how accurate each one individually is", rather than leaving a
human to eyeball two separate tables and guess:

```python
@dataclass(frozen=True)
class PairCandidateScore:
    model_a: str
    model_b: str
    f1_a: float
    f1_b: float
    observed_agreement: float   # PairAgreement.mean_agreement
    expected_agreement: float   # P_e, chance-level agreement given f1_a/f1_b alone
    kappa: float                # (observed - expected) / (1 - expected)
    score: float                # min(f1_a, f1_b) - max(0.0, kappa)
    n_books: int                # PairAgreement.n_books (pairwise-overlap coverage)

def rank_candidate_pairs(
    agreements: list[PairAgreement], gt_metrics: list[ModelGroundTruthMetrics],
) -> tuple[list[PairCandidateScore], list[tuple[str, str]]]:
    """Treats each model's arbitration-GT F1 (gt_metrics) as its
    per-entry "probability of being correct", p. For a pair (A, B):

        expected_agreement = p_a*p_b + (1-p_a)*(1-p_b)
        kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)
        score = min(f1_a, f1_b) - max(0.0, kappa)

    kappa ~ 0 means the two models agree exactly as much as their
    individual accuracy alone predicts under independent errors --
    healthy, genuine independence. kappa well above 0 means they agree
    MORE than accuracy alone explains -- a direct, quantified signal of
    correlated errors (same architecture family, shared training data,
    same systematic misreading of some layout). Only max(0, kappa) is
    penalized in `score` -- kappa below 0 (agreeing less than chance
    would predict) is not the failure mode this guards against, so it
    isn't rewarded or punished beyond already being reflected in a lower
    observed_agreement.

    Returns (scored_pairs sorted by score descending, unscored_pairs) --
    a pair where either model has no arbitration-GT F1 at all (zero
    qualifying books in gt_metrics) cannot be scored and is reported
    separately by (model_a, model_b) name, never silently dropped or
    scored with a fabricated stand-in accuracy."""
```

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
- A **candidate-pair ranking** table (`rank_candidate_pairs`'s output,
  best score first): model A, model B, f1_a, f1_b, observed agreement,
  expected agreement, kappa, score, n_books -- the table a human actually
  reads to pick the next bulk-gate pairing. Followed by a line listing
  any unscored pairs (missing arbitration-GT coverage for one/both
  models), so a pair's absence from the ranking is never mistaken for a
  bad score.

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
- **Candidate-pair ranking table**: `rank_candidate_pairs`'s output,
  best `score` first -- same columns as the CLI's version (f1_a, f1_b,
  observed/expected agreement, kappa, score, n_books), plus a link from
  each row back up to that pair's cell in the heatmap above. Unscored
  pairs listed underneath as a plain list, not silently omitted.
- A short caveats block (matching the existing Crossref page's
  `.caveats` box) noting: agreement measures how much two models say the
  same thing, not whether either is right; a HIGH kappa is the specific
  red flag for that failure mode (agreement in excess of what each
  model's own arbitration-GT accuracy already predicts -- a sign of
  shared blind spots, e.g. same architecture family) and is exactly why
  the ranking penalizes it rather than just sorting by raw agreement;
  arbitration-GT coverage differs per model and grows over time as more
  books get arbitrated; Crossref coverage is small and skewed (existing
  caveat, unchanged).

### 4. README

`README.md` gains a short "Model comparison" section (after "Crossref
evaluation", same tier of permanence -- this describes a repeatable
procedure, not a point-in-time snapshot) covering what the two raw
metrics and the derived candidate-pair score mean (including *why*
maximizing or minimizing raw agreement alone would each be the wrong
goal -- the same reasoning as this spec's "Why" section, condensed), how
to run `compare_models.py`, and a link to the new site page.
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
  the "Current status" table separately. `rank_candidate_pairs` inherits
  this: a pair's score is only as stable as the smaller of the two
  models' arbitration-GT sample sizes, and a book newly arbitrated can
  shift a pair's rank run to run. Not smoothed away -- N is always
  printed alongside the score for exactly this reason.
- **The kappa model is a simplification, not a measured probability.**
  Treating arbitration-GT F1 as a per-entry "probability of being
  correct" and assuming a wrong-and-wrong coincidental match is
  negligible are both modeling choices, not empirically verified for
  this corpus. Good enough to flag a large, obvious excess (two models
  agreeing far more than their accuracy explains), not precise enough to
  treat small kappa differences between two candidate pairs as
  decisive -- documented in the site caveats as an approximation to
  read directionally, not a precise probability.
- **A pair with no arbitration-GT coverage for either model can still
  have a real, useful agreement rate** (e.g. two well-established models
  neither of which happens to cover many arbitrated books yet) -- it's
  reported in the raw pairwise-agreement matrix (section 3's heatmap)
  even though `rank_candidate_pairs` can't score it. The heatmap and the
  ranking table are deliberately kept as separate outputs rather than
  the ranking being the only view, so this case stays visible instead of
  vanishing along with the unscorable pair.
