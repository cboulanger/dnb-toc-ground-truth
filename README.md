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
   `cli/generate_ground_truth.py`'s flags (which models to use, gate
   threshold, concurrency) so you don't have to repeat them on every
   invocation.

4. Smoke-check the install before pointing anything at a real endpoint:

   ```bash
   uv run python cli/fetch_corpus.py --help
   uv run python cli/generate_ground_truth.py --help
   ```

5. See `cli/README.md` for the full flag reference of every script, and
   `data/corpus/pilot/README.md` for the corpus's current size and
   status.

## Development

If you are an AI agent asked to work on this repo, read `AGENTS.md`
first -- it documents the arbitration workflow you're expected to
follow for books the automated gate can't resolve on its own.
