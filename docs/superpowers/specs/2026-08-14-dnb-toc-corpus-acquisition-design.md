# DNB/lobid digitized-TOC acquisition: a real-scan corpus for TOC-page classification

Status: approved for planning
Date: 2026-08-14

## Problem

The layout-based TOC/chapter-first-page classifier's `copyrighted-scans`
training data is small (13 books) and cannot be redistributed, and its
scan-degradation story is currently synthetic: `alto_scan_noise.py`
(see `docs/superpowers/specs/2026-08-12-layout-classifier-context-features-and-scan-augmentation-design.md`)
perturbs born-digital `open-access` ALTO with hand-picked constants --
font-size jitter drawn from ~U(0.96, 1.04), title/body contrast
compression from α ~U(0.3, 0.7) -- chosen to *resemble* the noise
measured in real scanned ALTO, not derived from it. Image-level
augmentation (render → degrade → re-OCR) was explicitly rejected in that
spec for cost and non-determinism.

A research pass into whether the lobid/GND ecosystem could supply
additional real scanned front-matter found a usable source. `lobid-gnd`
(the GND authority-file API) itself carries no digitized content -- only
entity metadata -- but its sibling service **`lobid-resources`**
(`lobid.org/resources`, the hbz union catalog as Linked Open Data) indexes
bibliographic records that frequently carry a populated `tableOfContents`
field, and the underlying scans come from the **Deutsche
Nationalbibliothek's (DNB) "Kataloganreicherung" program**: as of ~2024,
~5 million digitized tables of contents (10M+ pages), covering every
German monograph published since 2008 plus a retrospective sweep back to
1945 (~2.1-2.7M books), produced in a uniform format -- 300dpi bitonal
PDF with an embedded OCR text layer -- and released under **CC0** (no
restriction, no attribution required). The same lobid-resources record
that carries the TOC link also carries the book's full bibliographic
metadata (title, publisher, year, ISBN, DOI where known, GND subject
links) at no extra request cost -- worth keeping in full, not just the
few fields the acquisition script strictly needs (see Scope 1).

Confirmed live against one of this project's own corpus books
(`evaluation/corpus/copyrighted-scans/9783899718188.pdf`, "Systemtheorie
in den Fachwissenschaften"):

```
GET https://lobid.org/resources/search?q=isbn:9783899718188&format=json
```

returns a record typed `["BibliographicResource","EditedVolume","Book"]`,
`natureOfContent` = *Aufsatzsammlung* (GND `https://d-nb.info/gnd/4143413-4`),
and:

```json
"tableOfContents": [{
  "label": "Inhaltsverzeichnis",
  "id": "https://digitale-objekte.hbz-nrw.de/storage/2011/03/19/file_10/4104671.pdf"
}]
```

That PDF was fetched and read directly: a 2-page CCITT-Fax bitonal scan
with embedded OCR text, real chapter-title/author/page-number lines, and
a footer stamped "digitalisiert durch Deutsche Nationalbibliothek"
linking back to `http://d-nb.info/1008652334` -- i.e. exactly the visual
and layout shape `layout_features.py` targets as the `toc` positive
class, sourced from a real scanner, not a synthetic perturbation.

`type:EditedVolume` alone already matches ~497,000 hbz records (measured
2026-08-14), and `natureOfContent.id` gives a GND-anchored way to narrow
to edited volumes/essay collections specifically -- the book template
this project's evaluation corpus already skews toward (Festschriften,
Handbuch/reference-series volumes, edited German-language academic
collections).

**Known constraint that shapes the acquisition design below:**
`tableOfContents` is not queryable server-side in `lobid-resources` --
`_exists_:tableOfContents`, `tableOfContents.id:*`, and the documented
`nested=` query parameter all either return zero or error, even though
the field is present and populated in individual records fetched
directly. There is no API call that returns "only records with a TOC
scan." The field is real and present, just not indexed for that query.

## Scope

### 1. Acquisition (`evaluation/scripts/fetch_dnb_toc_corpus.py`)

New standalone script, manual run (not part of `uv run pytest` or CI),
same convention as `fetch_evaluation_pdfs.py`. Two modes, because the
`tableOfContents`-not-queryable constraint above rules out a single
targeted API query:

- **`--from-dump`** (primary/bulk mode): stream `lobid-resources`'
  documented weekly full JSON-Lines dump (or the daily-update variant;
  `Accept: application/x-jsonlines` / `format=jsonl`, see `lobid.org/resources/api`
  "Bulk-Downloads") and filter client-side for records with a non-empty
  `tableOfContents` array and `type` including `EditedVolume`
  specifically (not "Book" alone -- see the 2026-08-15 corrections plan
  for why that broader filter let single-author/thesis/textbook records
  through). Preferred over paging the live search API across ~497,000
  `type:EditedVolume` hits one page at a time, which would be far more
  requests against a shared public service for the same result.
- **`--isbns-file`**: targeted per-ISBN lookup via
  `resources/search?q=isbn:<isbn>&format=json` against a supplied list.
  Secondary use case: opportunistically check whether books already
  under consideration for `evaluation/corpus/*/manifest*.json` (or
  candidates not yet added) already have a free DNB-scanned TOC available
  -- see section 4.

Each matched record already has to be fetched in full to read
`tableOfContents`, so there's no extra request cost in keeping the whole
bibliographic record rather than distilling it down to a few fields.
For each match: download the linked TOC PDF (politely rate-limited; DNB
explicitly does not offer bulk PDF export -- "Eine Bereitstellung von
PDF-Dateien der digitalisierten Inhaltsverzeichnisse erfolgt nicht", so
this is inherently one HTTP request per record), capped by a `--limit`
flag, and write into a **new corpus directory that follows the existing
per-corpus shape**, so the same manifest/ground-truth conventions apply,
marked as TOC-only:

```
evaluation/corpus/dnb-toc-only/
  manifest.json            # committed -- the only file this script writes to git
  manifest.local.json      # optional, gitignored, same convention as other corpora
  <id>.pdf                  # gitignored -- never committed, see below
  .lobid-cache/              # gitignored -- full lobid-resources record per book, see below
    <id>.lobid.json
  .layout-cache/            # gitignored, same convention as other corpora
```

`fetch_dnb_toc_corpus.py` itself does not produce `<id>.expected.json` --
see "Out of scope" below.

`manifest.json` gains a corpus-level `"toc_only": true` sentinel (sibling
to `"books": [...]`) so downstream code can recognize the corpus shape
without special-casing on the directory name. Each book entry keeps the
fields shared with other corpora (`filename`, `title`, `language`,
`doi`), drops the ones that presuppose a full-book PDF
(`extraction_type`, `embedded_toc`, `oa`, `download_url` -- there is no
"native vs scan" or "downloadable whole book" here), and adds:

- `"toc_download_url"` -- the DNB-hosted TOC PDF link
  (`tableOfContents[].id` from the source record)
- `"license": "CC0-1.0"`, `"license_source": "dnb"`
- `"lobid_url"` -- a directly re-fetchable URL for this record's full
  lobid-resources data (the record's own lobid URI, `format=json`
  appended). The full record itself is NOT embedded in `manifest.json`
  -- an earlier version of this design did that under a `"lobid_record"`
  key, but at real corpus scale (~1,000 lines per book, mostly library
  holdings data no code reads) that made `manifest.json` an
  unreviewable multi-hundred-thousand-line file. Instead,
  `fetch_dnb_toc_corpus.py` writes the full record to a separate,
  gitignored `.lobid-cache/<id>.lobid.json` file (see
  `evaluation/.gitignore`'s `.lobid-cache/` entry) -- available locally
  without a network round-trip, but never committed, same rationale as
  the PDFs themselves. Kept in its own subdirectory rather than loose in
  the corpus dir's top level, the same way `.ocr-cache/`/`.layout-cache/`
  already are for other corpora, so the directory listing stays readable
  at real corpus scale (500+ books) instead of interleaving two files per
  book with the PDFs.

  Publisher, year, subjects/GND links (including
  `natureOfContent`/*Aufsatzsammlung*), and whatever else
  `lobid-resources` returns for future analysis this spec doesn't scope
  live only in that sidecar file (or a re-fetch via `lobid_url`), not in
  `manifest.json` itself; `filename`, `title`, `language`, `doi`,
  `toc_download_url`, `license*`, and `lobid_url` above remain the only
  fields any tooling in this spec actually reads from `manifest.json`.
  `manifest.json` stays the single file this script writes to git --
  `.lobid-cache/<id>.lobid.json` is the companion per-book metadata file,
  gitignored like the PDF itself.

Note on what a future `<id>.expected.json` for this corpus could and
couldn't hold, since it shapes the "equivalent metadata shape" framing
above: `citation_pages` *is* recoverable here -- unlike
`pdf_start_index`/`pdf_end_index`, which name a position in the full
book's page sequence and require the full book to determine, the printed
page numbers a `citation_pages` range is built from are exactly what the
TOC scan itself prints next to each chapter title. Generating that file
(OCR/parse each TOC scan into `{title, authors, citation_pages}` per
chapter) is a well-defined, plausible follow-up, but this spec doesn't
build it -- see "Out of scope."

**Harness compatibility.** `evaluation.harness.list_corpora()`
(`evaluation/harness.py:41`) currently returns every
`evaluation/corpus/` subfolder that has a `manifest.json`,
unconditionally -- every existing script that loops over it
(`fetch_evaluation_pdfs.py`, `ocr_evaluation_pdfs.py`,
`evaluate_chapter_segmentation_strategies.py`,
`generate_public_evaluation_cache.py`, `generate_report.py`,
`refresh_llm_cache.py`) assumes a corpus has fetchable PDFs and full
`.expected.json` chapter fields. Since `dnb-toc-only/` has neither,
`list_corpora()` needs one small, backward-compatible change: an
`include_toc_only: bool = False` parameter, checking the manifest's
`toc_only` sentinel, so every existing call keeps silently skipping this
corpus by default and only the new scripts in sections 2-3 below pass
`include_toc_only=True` explicitly. This is the one change this spec
makes to shared harness code; see "Decision criteria."

**PDFs are never committed.** Same as every other corpus
(`evaluation/.gitignore` already has a blanket `*.pdf`), not a special
case introduced for this one -- CC0 licensing would permit committing
them, but git is still the wrong distribution channel for what could
grow to hundreds of PDFs. The intended distribution path is a separate
**Zenodo dataset upload**, assembled by hand from a completed
`--from-dump` run and published outside this repo once the corpus is
judged large/clean enough (see "Decision criteria"); scripting that
upload is out of scope (see below). Only `manifest.json` -- plain text,
small -- is committed to this repo.

### 2. Feature extraction

Reuse `pdfalto_runner.py` unchanged against each downloaded PDF. Cache
ALTO output at `evaluation/corpus/dnb-toc-only/.layout-cache/<id>.alto.xml`,
same manual-invalidation, gitignored convention as the other corpora.
Every page in every one of these PDFs is a confirmed `toc` page by
construction (DNB only digitizes the TOC itself, never surrounding book
pages) -- no per-page labeling step is needed, unlike the existing
ground-truth workflow.

### 3. Consumption: calibrating `alto_scan_noise.py`, not replacing it

New script `evaluation/scripts/measure_dnb_scan_noise_stats.py` calls
`list_corpora(include_toc_only=True)` (or targets `dnb-toc-only`
directly) and computes, across the acquired corpus's ALTO: per-line
font-size dispersion (to compare against the current ~U(0.96, 1.04)
jitter range) and title/body contrast ratio distribution (to compare
against the current α~U(0.3, 0.7) compression range). Output: a
comparison table (current hand-picked constants vs. measured real
distribution), written up in `evaluation/RESULTS.md` as a new follow-up
subsection, same prose-plus-tables convention as the rest of that
document. If the measured distributions diverge meaningfully from the
current constants, update `alto_scan_noise.py`'s ranges to match; if
they're already close, that's a useful confirmation the synthetic
approach was well-calibrated and needs no change.

A second, purely diagnostic use: run the already-trained page-local
layout classifier against this corpus's page features as an
out-of-population recall check -- does it still recognize a `toc` page
sampled from a completely disjoint set of ~hundreds of thousands of
books it never trained on. This is a sanity signal only, not a new
training input -- these pages have no surrounding book (no
`page_position_fraction`, no `prev_last_text_vpos_fraction`, no
chapter-first counterpart to pair with), so they cannot participate in
the existing per-book LOBO harness or `add_book_context_features`.

### 4. Workflow note (`evaluation/CLAUDE.md`)

Add a short note to Step 1 ("Transcribe the table of contents") of the
new-evaluation-book workflow: before manually transcribing a new
corpus candidate's TOC by hand, check `dnb-toc-only/manifest.json` and
`lobid-resources` by ISBN (`resources/search?q=isbn:<isbn>&format=json`)
for an existing `tableOfContents` link. When present, it's a ready-made,
already-OCR'd TOC to transcribe from instead of working from the raw PDF
alone -- still requires the same by-hand verification against the actual
book (citation-page-to-`pdf_index` matching is unaffected by this
shortcut, per the design spec section 2/5 rationale `evaluation/CLAUDE.md`
already documents), but removes the need to visually locate and read the
TOC pages from scratch.

## Decision criteria

- `fetch_dnb_toc_corpus.py --from-dump` acquires at least a few hundred
  verified positive TOC-page scans spanning multiple publishers/years/
  publication types, with a spot-check pass (open a sample of the
  downloaded PDFs, confirm each really is a table of contents, not a
  mismatched or corrupt link) finding no more than an occasional bad
  record.
- `measure_dnb_scan_noise_stats.py` produces a concrete, reported
  before/after comparison of `alto_scan_noise.py`'s constants against
  measured real distributions in `RESULTS.md` -- whether or not that
  leads to changing the constants, the measurement itself is the
  deliverable (same tempering stance as every other follow-up in this
  project).
- No change to the existing `open-access`/`copyrighted-scans` corpora,
  their schema, or the LOBO evaluation harness, beyond the single opt-in
  `include_toc_only` parameter on `list_corpora()` needed so those
  corpora's own scripts keep ignoring `dnb-toc-only/` by default.

## Out of scope

- **Growing the per-book evaluation corpora directly** -- DNB digitizes
  only the TOC excerpt, never the surrounding book, so this source
  cannot supply new `open-access`/`copyrighted-scans` entries (no
  chapters, no full-book page sequence, no `.expected.json` with
  `pdf_start_index`/`pdf_end_index` to build).
- **Replacing `alto_scan_noise.py` with real-data splicing** (stitching
  genuine degraded ALTO fragments into synthetic book context) --
  plausible future direction if calibration in section 3 shows the
  current synthetic approach is off, but not this spec's deliverable.
  Deferred, not rejected.
- **GND-based automatic genre/subject classification** beyond the single
  `natureOfContent` = Aufsatzsammlung filter used for targeting.
- **Any change to `evaluate_layout_toc_classifier.py`'s model, feature
  set, or `TocExtractionStrategy` production wiring.**
- **Automating the Zenodo upload.** This spec covers acquisition into a
  local, gitignored PDF set plus a small committed manifest/metadata
  shell; assembling and publishing the full PDF corpus as a Zenodo
  dataset is a manual follow-up action, not a script this spec
  delivers.
- **Generating `<id>.expected.json` chapter lists** (`title`, `authors`,
  `citation_pages`) by parsing each TOC scan's OCR text. Feasible in
  principle -- `citation_pages` is directly legible from the TOC, unlike
  `pdf_start_index`/`pdf_end_index` -- but it needs its own
  extraction/verification step (OCR-to-structured-chapters, likely
  LLM-assisted, per-book spot-checking) that this spec doesn't design or
  build. `fetch_dnb_toc_corpus.py` only acquires PDFs and bibliographic
  metadata.
