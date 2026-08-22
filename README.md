# DNB Table of Contents Ground Truth

A pilot case for generating structured, machine-checkable ground truth
from openly available data -- the Deutsche Nationalbibliothek's
CC0-licensed "Kataloganreicherung" table-of-contents scans -- using
independent LLM reads gated against each other for agreement, with
arbitration by a strong, multimodal AI agent (such as Claude) or a
human for the disagreements.

The output (`data/corpus/pilot/ground-truth/*.expected.json`) is meant
as an input to *other* pipelines, not an end in itself: fine-tuning a
smaller structured-extraction model, benchmarking chapter/TOC extraction
heuristics, or training a lightweight classifier could all consume this
corpus without depending on this repo's own LLM pipeline. The LLM-based
generation pipeline in this repo is the means of producing that data,
not the point -- the point is the ground-truth data itself, general
enough to feed pipelines this repo doesn't build.


## Methodology

Each book's ground truth comes from one of two routes:

- **Two-model agreement** (`"source": "bulk_gate"`, `"verified": false`)
  -- two independent vision-LLM reads of the same TOC page images are
  diffed against each other (`dnb_toc_ground_truth.matching.gate_books`);
  if the best-agreeing pair matches on at least 90% of entries, that
  merge is written out directly, with no review by anyone.
- **Full arbitration** (`"source": "agent_arbitration"`, `"verified": true`)
  -- for every book the gate can't auto-resolve (the two models disagree,
  or one/both fail outright), a strong, multimodal AI agent such as
  Claude reads the actual TOC page images directly and transcribes the
  ground truth by hand (see `AGENTS.md`). This is generally done by an
  AI agent, not a human, since it scales far better across a corpus this
  size.

As of the last count (`docs/history.md`'s "Current status"), **~69% of
ground-truthed books (375 of 547) needed full arbitration** -- the
two-model gate alone resolved the rest. That arbitration rate is the
project's core efficiency metric: the aim is to iteratively improve the
gate (better prompts, better-paired models, a higher-quality second
reader) so fewer books need a frontier-model agent's direct attention,
not to keep leaning on arbitration indefinitely.

## Crossref evaluation

Since 2026-08-22, this repo also cross-checks its ground truth against
an independent, non-LLM data source: Crossref's own per-chapter
registration metadata (title, authors, page range), for books that have
one. This is the only correctness signal in this repo that comes from
outside its own LLM extraction pipeline entirely.

**[View the current scores on GitHub Pages](https://cboulanger.github.io/dnb-toc-ground-truth/)**
-- rebuilt automatically from committed corpus data on every push to
`main` (`.github/workflows/pages.yml`).

**How it works:**

1. Whenever a book's Crossref data is fetched -- either by
   `cli/backfill_crossref.py` (for the existing manifest backlog) or in
   real time by `cli/fetch_corpus.py` as new books are acquired --
   `dnb_toc_ground_truth.crossref.write_evaluation_entry` filters the
   returned chapters to those Crossref registered **with real page
   data** (a chapter with none can never be matched by this repo's own
   alignment logic, since it never matches a known-page ground-truth
   entry against an unknown-page one) and writes them to a **committed**
   `data/corpus/pilot/evaluation/<key>.expected.json`, in the same
   shape as a ground-truth file (`"source": "crossref"`), but only if at
   least `--min-chapters` (default 3) survive that filter -- a book with
   too few usable Crossref-registered chapters isn't a meaningful
   sample.
2. `cli/evaluate_crossref.py` then compares, for every book that has
   both a ground-truth file and an evaluation-corpus entry, the ground
   truth's real chapters (`"skip": false`) against the evaluation
   corpus's entries, reusing `dnb_toc_ground_truth.matching.diff_toc_entries`
   completely unmodified (title, chapter-number-prefix and
   capitalization normalized, plus first-page-number equivalence -- the
   same logic that gates the two-model TOC-extraction agreement above).
   From the resulting matched/only-in-gt/only-in-crossref counts: true
   positives = matched, false negatives = a real ground-truth chapter
   Crossref didn't register or match, false positives = a Crossref
   chapter with no ground-truth match -- standard precision, recall, and
   F1 from there.
3. `--model` (repeatable) or `--all-models` additionally scores a
   specific vision-LLM's *raw, pre-gate/pre-arbitration* cached
   extraction (`data/corpus/pilot/llm-cache/`, written by
   `cli/generate_ground_truth.py`) against the same crossref-sample
   books, for whichever of them that model happens to have a cache entry
   for -- so a model's extraction quality can be scored against this
   corpus's one non-LLM signal directly, without needing a ground-truth
   file (gated or arbitrated) for that book at all.

**Run it:**

```bash
uv run python cli/backfill_crossref.py       # populate/refresh the evaluation corpus
uv run python cli/evaluate_crossref.py       # aggregate mean only
uv run python cli/evaluate_crossref.py --full  # + a line per compared book
uv run python cli/evaluate_crossref.py --all-models  # + every cached model's own score
```

**Constraints, read before trusting a number this produces:**

- **Aggregate scores are macro-averaged** -- the mean of each compared
  book's own precision/recall/F1, every book weighted equally regardless
  of its chapter count. A single short book and a single 300-entry
  handbook count the same toward the mean.
- **Coverage is small and not a random sample of the corpus.** As of the
  first real run (2026-08-22), only 54 of 547 ground-truthed books had
  enough Crossref-registered, page-numbered chapters to produce an
  evaluation entry at all (mean precision 85%, recall 77%, F1 76%
  across those 54) -- Crossref registration skews toward larger and
  more prominent publishers, so this check says nothing about books it
  has no data for.
- **A Crossref "miss" doesn't necessarily mean the ground truth is
  wrong.** Crossref's own chapter registration can itself be incomplete
  or scoped differently than this corpus's TOC-page extraction -- e.g. a
  handbook's many short entries registered as one part rather than
  individually, which this check would score as many false negatives
  even though the ground truth is correct.
- **Some publishers' Crossref metadata has its own quirks** already
  found and corrected for (see `crossref.py`'s
  `_strip_glued_page_prefix` and `evaluate_crossref.py`'s page-sorting
  before comparison) -- but the underlying Crossref data is
  publisher-submitted and not otherwise vetted by this repo.
- **`--model`/`--all-models` scores a model's *raw* cache output, not
  this repo's finished ground truth.** It's the same single-read
  extraction this repo's own two-model gate (`matching.gate_books`) is
  designed to catch errors in *before* they reach a ground-truth file --
  so expect these numbers to run lower than the ground-truth-vs-Crossref
  numbers above, and don't read a lower per-model score as Crossref
  disagreeing with this project's own ground truth.

## Setup

1. Install dependencies:

   ```bash
   uv sync
   ```

2. External binaries, on `PATH` or via a `--<tool>-bin` flag / env var
   (same convention as `PDFALTO_BIN` below): `ocrmypdf` (for the
   OCR-text extraction path -- `brew install ocrmypdf`), and a sibling
   [`pdfalto`](https://github.com/kermitt2/pdfalto) checkout for ALTO
   reconstruction (not vendored, not installable via brew -- build it
   next to this repo and point `PDFALTO_BIN` at the resulting binary).

3. Copy the example credential/config files and fill in real values:

   ```bash
   cp .endpoints.dist .endpoints
   cp .config.json.dist .config.json
   ```

   `.endpoints` lists every inference endpoint you can call (see
   `docs/llm-inference-providers.md`); `.config.json` sets defaults for
   the CLI scripts' flags (which models to use, gate threshold,
   concurrency, and which corpus to operate on) so you don't have to
   repeat them on every invocation.

4. This repo can hold more than one corpus under `data/corpus/<name>/`
   (today, in practice, just `pilot`). Every script except
   `generate_evaluation_site.py` takes a `--corpus` flag to select one,
   defaulting to `.config.json`'s `"corpus"` key, then `"pilot"`. A name
   with no `data/corpus/<name>/manifest.json` yet is created fresh by
   `cli/fetch_corpus.py --corpus <name> ...` (every other script expects
   one to already exist).

5. Smoke-check the install before pointing anything at a real endpoint:

   ```bash
   uv run python cli/fetch_corpus.py --help
   uv run python cli/generate_ground_truth.py --help
   ```

6. See `cli/README.md` for the full flag reference of every script, and
   `data/corpus/pilot/README.md` for the corpus's current size and
   status.

## Development

If you are an AI agent asked to work on this repo, read `AGENTS.md`
first -- it documents the arbitration workflow you're expected to
follow for books the automated gate can't resolve on its own.
