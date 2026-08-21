# dnb-toc-ground-truth

A pilot case for generating structured, machine-checkable ground truth
from openly available data -- the Deutsche Nationalbibliothek's
CC0-licensed "Kataloganreicherung" table-of-contents scans -- using
independent LLM reads gated against each other for agreement, with
human/Claude arbitration for the disagreements.

The output (`data/corpus/pilot/ground-truth/*.expected.json`) is meant
as an input to *other* pipelines, not an end in itself: fine-tuning a
smaller structured-extraction model, benchmarking chapter/TOC extraction
heuristics, or training a lightweight classifier could all consume this
corpus without depending on this repo's own LLM pipeline. The LLM-based
generation pipeline in this repo is the means of producing that data,
not the point -- the point is the ground-truth data itself, general
enough to feed pipelines this repo doesn't build.

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
   cp .config.dist .config
   ```

   `.endpoints` lists every inference endpoint you can call (see
   `docs/llm-inference-providers.md`); `.config` sets defaults for
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
