# `pilot` corpus

Real, born-scanned table-of-contents pages acquired from the **Deutsche
Nationalbibliothek's "Kataloganreicherung" program** via the `lobid-resources`
API (`lobid.org/resources`). Each entry's `.pdf` (under `pdf/`) is a short
(1-4 page), 300dpi bitonal scan of *just the TOC*, with an embedded OCR
text layer, released under **CC0** (no restriction, no attribution) — full
provenance and acquisition details in
`docs/superpowers/specs/2026-08-14-dnb-toc-corpus-acquisition-design.md`.

There is no full book here, no chapter-location result to produce — just
what each TOC page prints. `ground-truth/<id>.expected.json` is a flat
list of that:

```json
{
  "entries": [
    {"title": "...", "authors": ["..."], "printed_page_number": "N", "skip": false}
  ],
  "verified": true,
  "source": "claude_arbitration"
}
```

Extraction is deliberately **verbatim, not editorial**: every line the TOC
page actually prints gets an entry, including part/section dividers and
front/back matter (preface, bibliography, index, ...) — `"skip": true`
marks those, `"skip": false` marks an actual chapter, but nothing is ever
omitted outright (see `TocEntry.skip`'s docstring in
`src/dnb_toc_ground_truth/toc_entry.py`) — before this, the extraction
prompt told the vision models to leave non-chapter lines out entirely,
which made two independent models' agreement rate sensitive to editorial
judgment calls ("is this back matter or a chapter?") on top of genuine
reading mismatches, and meant a below-threshold book could just as easily
be a disagreement about what to *call* a line as about what was actually
printed. Any `.expected.json` written before this change has no `"skip"`
key on its entries at all and is missing whatever lines its extraction
chose to omit.

`manifest.json` carries `"toc_only": true` at the top level and, per book,
`filename`, `title`, `language`, `doi`, `toc_download_url` (the original DNB
scan URL), `license`/`license_source` (`"CC0-1.0"`/`"dnb"` for essentially
every entry), and `lobid_url` (the source bibliographic record — also where
to re-fetch metadata if a field looks wrong).

## Why it exists

Two consumers, one acquisition pipeline:

- **Layout-based TOC/chapter-first-page classifier training data** — real
  scan noise (skew, bitonal artifacts, genuine font/contrast variation)
  at a scale (~1,251 books) a small hand-scanned corpus can't approach.
- **A large, cheap "does automated TOC extraction get this right" ground
  truth set** — since each book is just its TOC page(s), building ground
  truth here means transcribing what's printed, not locating chapter
  boundaries in a full book — cheap enough to attempt at scale via
  independent LLM reads instead of hand-transcribing every book.

## Ground-truth generation: two-tier workflow

Both tiers write `ground-truth/<id>.expected.json` in the schema above;
only the `"verified"`/`"source"` values and how much human judgment went
in differ.

**Bulk tier** (`"verified": false, "source": "bulk_gate"`) — automated, no
human review. `cli/generate_ground_truth.py` renders each book's page
images (`pdftoppm`, no OCR) or its OCR'd text and sends them to every
model named via `--use-vision`/`--use-text` (resolved against
`.endpoints`); a book's `.expected.json` is written only when the
best-agreeing pair of reads agrees on at least 90% of entries
(`dnb_toc_ground_truth.matching.gate_books`). Already-decided books (an
existing, current-schema `.expected.json`) and rejected ones (below) are
skipped automatically, so re-running the same command just picks up
where the last run left off. A pre-2026-08-17 `bulk_gate` file (missing
the `"skip"` key -- see above) is the one exception: it's treated as
undecided and silently regenerated under the current verbatim standard,
since it was never human-reviewed anyway; a pre-2026-08-17
`claude_arbitration` file is left untouched instead, per this repo's own
`CLAUDE.md`'s note on retrofitting those by hand:

```bash
cp .endpoints.dist .endpoints  # fill in real values, see docs/llm-inference-providers.md
uv run python cli/generate_ground_truth.py --use-vision <model-a>,<model-b> --limit 100 --concurrency 4
```

**Arbitration** (`"verified": true, "source": "claude_arbitration"`) — for
books the bulk tier skipped (models disagreed, or one/both failed outright).
`cli/arbitrate.py` surfaces each one's raw extractions
(from `llm-cache/<schema-version>/<key>.<model>.json`, kept regardless of
gate outcome) side by side; a human (or Claude Code, per this repo's own
`CLAUDE.md`) reads the disagreement, opens the actual TOC page images when
the text diff alone doesn't settle it, and hand-writes the final
`.expected.json`, transcribing every printed line (not just chapters) with
the same `"skip"` flag convention as the bulk tier. A book that turns out
genuinely unrecoverable (every model hallucinates, the scan is too
degraded to read even directly) gets recorded in
`arbitration-rejected.json` via `cli/arbitrate.py reject <key> "<reason>"`
instead of resurfacing on every run.

`llm-cache/`'s per-(book, model) files are versioned by extraction standard
(`vision.py`'s `versioned_cache_dir`, currently `v2`) rather than
overwritten in place when the standard changes -- an older version's files
are left alone on disk (some are already git-committed) but never read by
current code, so a schema change can't silently resurrect a stale,
incomplete extraction from cache instead of asking the model again.

See this repo's own top-level `README.md` for setup and the eval-tier
(fully hand-transcribed, held-out) sample procedure
(`cli/select_eval_sample.py`), and `docs/history.md`'s "Current status"
section for current coverage numbers, observed failure modes, and
endpoint-provider characteristics of running this at scale.
