# Agent notes

## Starting a fresh batch (generation step, before arbitration)

Before running `cli/generate_ground_truth.py`, a few things aren't
obvious from that script's own `--help` and aren't written down
anywhere else in this repo:

- **`.endpoints`'s `status` field (e.g. "Building") is unreliable in
  both directions** -- a session that's actually fully loaded can still
  show "Building", and vice versa. The only real readiness signal is
  hitting the endpoint's own `<url>/v1/models` with its `key` as a
  bearer token and checking for HTTP 200. Do this for every endpoint
  before kicking off a batch rather than trusting the status column.
- **There is no `.env` in this repo.** Model credentials live entirely
  in `.endpoints` (one `key`/`url`/`model` per pasted dashboard session
  block, or a JSON array -- see `src/dnb_toc_ground_truth/inference.py`).
  If someone says "fresh session credentials in `.env`", they mean a
  freshly-regenerated `.endpoints` file.
- **A model id in `.endpoints` must exactly match the id in
  `.config.json`'s `use_vision`/`use_text`, or resolution fails
  outright** (`ValueError: no endpoint found for model ...`), even if
  it's obviously "the same model" to a human. In particular, an
  endpoint serving a fine-tuned variant (e.g. `cmboulanger/nuextract3-toc`
  instead of the base `numind/NuExtract3`) needs BOTH files updated to
  the exact same string, AND needs `extraction_api\tnuextract` added
  explicitly to that block in `.endpoints` -- the NuExtract template-mode
  extraction path only auto-applies for the literal model id
  `numind/NuExtract3` (see `_resolve_extraction_fields` in
  `inference.py`); any other id silently falls back to the plain-text
  prompt path unless told otherwise. This is a pipeline-shape decision
  (which model backs future "NuExtract3" readings), not a mechanical
  fix -- ask before switching it if unclear which model is intended.
- **`data/corpus/pilot/pdf/`, `.lobid-cache/`, `.crossref-cache/`, and
  `.locks/` are all gitignored** -- a fresh checkout has none of them,
  even though `manifest.json` (committed) lists 1000+ books. On a fresh
  checkout, `generate_ground_truth.py` will pick its usual candidates
  but skip every one of them as `missing_pdf`, reporting `0/0 books
  passed the gate` with no other explanation.
  - **`--limit` is applied to the eligible-book list BEFORE filtering
    on local PDF existence** (see `_generate` in
    `cli/generate_ground_truth.py`), so a missing-PDF book silently
    eats one slot of your batch instead of being skipped in favor of
    the next eligible book. A `--limit 50` run with no local PDFs
    processes zero books, not 50 different ones.
  - **`cli/fetch_corpus.py` does NOT fix this.** It only discovers and
    downloads NEW manifest entries (via `--from-dump` or
    `--isbns-file`), skipping anything already in the manifest --
    there's no built-in "re-download PDFs for existing manifest
    entries" command. To rehydrate, download each candidate book's
    manifest `toc_download_url` field directly into
    `data/corpus/pilot/pdf/<filename>` yourself (a plain `httpx` GET is
    enough; no need to re-derive `_acquire_record`'s full logic since
    the manifest entry, lobid cache, and Crossref data already exist).
    Compute the exact candidate set first with the same
    `_still_needs_a_decision`/`--limit` logic `generate_ground_truth.py`
    uses, so you fetch precisely the books that batch will actually
    attempt.

## Arbitrating below-gate books

`cli/generate_ground_truth.py`'s agreement gate discards a book outright
when no pair of its resolved endpoints' reads agrees well enough (below
0.90 agreement) or fewer than two endpoints produce usable output -- but
it never deletes any endpoint's cached raw extraction
(`data/corpus/pilot/llm-cache/<key>.<model>.json`). Rather than
re-running the whole book from scratch or leaving it discarded, walk
through the following after a generation run leaves books below the
gate (design spec `docs/superpowers/specs/2026-08-16-dnb-toc-arbitration-design.md`).

This arbitration work -- and manual inspection generally, wherever the
gate's weaker models fail -- is done by a strong, multimodal AI agent
(such as Claude), not by a human reading the scans directly:

1. List every book still needing a decision:

   ```bash
   uv run python cli/arbitrate.py
   ```

   This prints, per book: its title and PDF path, every cached model's
   entry count, and (for exactly two cached models) their agreement rate
   plus every entry each side found that the other didn't -- or, if only
   one model produced usable output, that model's full list with a note
   to verify it directly.

2. For each book, read the printed diff. The disagreement patterns found
   in practice so far (`docs/history.md`'s "Current status") usually
   make the right call obvious from the text alone: one side dropping
   real content, one side including front/back matter or a part-divider
   that should have been skipped, a two-line title wrongly split into
   two entries, or a deeply nested TOC segmented at different
   granularities.

3. When the text alone doesn't settle it, open the book's actual TOC
   page images directly: use the `Read` tool on the PDF with a `pages`
   parameter (1-based viewer pages).

4. Write the final `data/corpus/pilot/ground-truth/<key>.expected.json`
   yourself -- same schema as a passing book
   (`{"entries": [...], "verified": true, "source": "agent_arbitration"}`,
   each entry via `dnb_toc_ground_truth.matching.toc_entry_to_gt_dict`),
   but with `"verified": true` rather than `false`: unlike the bulk-tier
   gate's own output, this went through direct scrutiny (including the
   images, when needed) -- excluded from `_spot_check`'s sampling pool
   going forward. The `"source": "agent_arbitration"` field (vs. the
   bulk gate's own `"source": "bulk_gate"`) records that this entry's
   ground truth came from an arbitrated review, not the automated
   agreement gate.

   **Transcribe every printed line, not just the ones you'd call real
   chapters** -- part/section dividers and front/back matter (preface,
   bibliography, index, ...) get their own entry too, with
   `"skip": true`; real chapters get `"skip": false` (see `TocEntry.skip`'s
   docstring in `src/dnb_toc_ground_truth/toc_entry.py`).

5. If a book is genuinely unrecoverable (every model hallucinates, the
   scan itself is too degraded to read even directly), record that
   instead of leaving it to resurface every run:

   ```bash
   uv run python cli/arbitrate.py reject <key> "<short reason>"
   ```

   This writes to the committed `data/corpus/pilot/arbitration-rejected.json`
   -- refuses (rather than silently overwriting) if `<key>` is already
   present, so re-running this step is safe.

6. If instead only ONE model consistently hangs or emits malformed
   output on a specific book (the book's other models still read it
   fine, and/or you don't want to permanently give up on that model for
   it), don't reject the whole book -- record the (key, model) pair
   instead so future `generate_ground_truth.py` runs stop retrying it
   into the hard-timeout budget:

   ```bash
   uv run python cli/skip_list.py add <key> "<model>" "<short reason>"
   ```

   This writes to the committed `data/corpus/pilot/model-skip-list.json`.
   Unlike step 5's rejection, this is meant to be temporary -- once a fix
   is worth trying (e.g. guided/structured JSON decoding), clear it with
   `uv run python cli/skip_list.py remove <key> "<model>"` so the next
   run attempts that pair again.

7. After a generation or arbitration batch changes the corpus's
   coverage numbers, refresh the top-level `README.md`'s "Current
   status" table:

   ```bash
   uv run python cli/corpus_status.py
   ```

   This rewrites the table in place (between its `<!-- corpus-status:
   -->` markers) with fresh counts -- manifest size, ground truth (and
   the bulk-gate/arbitration split), per-model reading counts, the
   arbitration backlog, and the Crossref evaluation-corpus size. Run it
   as the last step of any session that wrote new ground truth or cache
   entries, and commit the resulting README diff alongside the data.
