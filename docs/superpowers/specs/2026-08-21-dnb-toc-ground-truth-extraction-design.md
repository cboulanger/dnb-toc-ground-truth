# Extracting `dnb-toc-ground-truth` into a standalone repository

## Motivation

The `dnb-toc-only` ground-truth generation pipeline (fetch DNB-digitized
TOC scans, extract via two independent vision/text LLM reads, gate on
agreement, arbitrate disagreements) has grown into a self-contained
system with its own corpus, its own endpoint-configuration mechanism, and
its own history (`evaluation/experiments/dnb-toc-ground-truth.md`, 853
lines). It has no real coupling to `chapter_segmentation`'s actual
chapter-boundary-finding logic beyond a few small, stable data types. Its
current home under `evaluation/` in `chapter-segmentation` also ties it
to KISSKI (Academic Cloud) as the only zero-config inference provider,
and to a two-model-only agreement gate.

This spec covers extracting the pipeline into a new standalone repo,
`dnb-toc-ground-truth`, generalizing the inference-provider story to any
number of generic OpenAI-compatible endpoints (dropping KISSKI-specific
code), and supporting more than two candidate models per book (gated on
"any two agree", not "these exact two agree").

## Scope

### Moves to `dnb-toc-ground-truth`

Code:
- `evaluation/dnb_toc_vision.py`, `evaluation/dnb_toc_ocr.py`,
  `evaluation/dnb_toc_matching.py` — extraction + whole-book agreement
  gate
- `evaluation/scripts/generate_dnb_toc_ground_truth.py`,
  `evaluation/scripts/arbitrate_dnb_toc.py`,
  `evaluation/scripts/fetch_dnb_toc_corpus.py`,
  `evaluation/scripts/select_dnb_toc_eval_sample.py` — the CLI entry
  points
- `evaluation/inference_endpoints.py` — forked and genericized (see
  "Endpoint and config system" below); the fork in `chapter-segmentation`
  keeps serving `refresh_llm_cache.py` unchanged
- `evaluation/scripts/pdfalto_runner.py` — vendored (copied) since
  `dnb_toc_ocr.py`'s OCR-mode text reconstruction needs it, but the
  original stays in `chapter-segmentation` too (`evaluate_layout_toc_classifier.py`
  and `build_crossref_gt_ground_truth.py` still need it there)
- `TocEntry`, `_parse_toc_page_number`, `_toc_items_to_entries` (from
  `src/chapter_segmentation/segmentation.py`) and `parse_json_array`
  (from `src/chapter_segmentation/_llm_json.py`) — vendored into the new
  repo's own module (`src/dnb_toc_ground_truth/toc_entry.py`), not
  imported from `chapter_segmentation`. This is a deliberate fork: these
  are stable TOC-line data shapes/parsers, and vendoring them is what
  makes the new repo have zero dependency on the `chapter_segmentation`
  package.

Data (filesystem copy, not git history — most of it is gitignored today):
- `evaluation/corpus/dnb-toc-only/` in full: `manifest.json`, every
  `*.pdf`, every `*.expected.json`, `llm-cache/`, `.lobid-cache/`,
  `.locks/`, `.layout-cache/`, `eval_tier_ids.json`,
  `arbitration-rejected.json`, `README.md`

Docs:
- `evaluation/experiments/dnb-toc-ground-truth.md` → `docs/history.md`
  (verbatim, then maintained going forward under the same
  Current-status/History split convention it already uses)
- `evaluation/hpc/llm-mpcdf.md` → generalized, KISSKI-specific framing
  removed, moved under `docs/`
- `evaluation/CLAUDE.md`'s **"Arbitrating below-gate dnb-toc-only books"**
  section → becomes the new repo's own `CLAUDE.md`, adapted for the new
  paths/CLI names (`cli/arbitrate.py` instead of
  `evaluation/scripts/arbitrate_dnb_toc.py`,
  `data/corpus/pilot/<key>.expected.json` instead of
  `evaluation/corpus/dnb-toc-only/<key>.expected.json`, etc.) but
  otherwise unchanged workflow guidance (list → read diff → arbitrate via
  Read tool on page images → write `.expected.json` with
  `"source": "claude_arbitration"` → `reject` for unrecoverable books).
  The old repo's `evaluation/CLAUDE.md` loses this section entirely (see
  "Old-repo CLAUDE.md updates" below) rather than keeping a stale copy.
- Specs/plans scoped to this pipeline from `docs/superpowers/{specs,plans}/`:
  `2026-08-14-dnb-toc-corpus-acquisition{,-design}.md`,
  `2026-08-15-dnb-toc-ground-truth-generation{,-design}.md`,
  `2026-08-15-dnb-toc-corpus-corrections.md`,
  `2026-08-15-dnb-toc-ground-truth-and-consumers-design.md`,
  `2026-08-16-dnb-toc-uniform-ocr-design.md`,
  `2026-08-16-dnb-toc-arbitration{,-design}.md`,
  `2026-08-16-dnb-toc-vision-extraction.md`,
  `2026-08-18-inference-endpoint-abstraction{,-design}.md`,
  `2026-08-20-dnb-toc-vision-text-pairing-{plan,design}.md` — copied into
  `docs/superpowers/{specs,plans}/` in the new repo, preserved as
  historical record even where this spec supersedes parts of them (e.g.
  the KISSKI-specific sections)

Tests: `test_dnb_toc_vision.py`, `test_dnb_toc_ocr.py`,
`test_dnb_toc_matching.py`, `test_arbitrate_dnb_toc.py`,
`test_generate_dnb_toc_ground_truth.py`, `test_fetch_dnb_toc_corpus.py`,
`test_select_dnb_toc_eval_sample.py`, `test_inference_endpoints.py`
(rewritten for the new resolution logic). `test_kisski.py` is dropped —
no KISSKI-specific code remains to test.

### Stays in `chapter-segmentation`, unchanged

- The layout-based TOC classifier pilot
  (`evaluate_layout_toc_classifier.py`, `layout_features.py`,
  `layout_labels.py`, `alto_scan_noise.py`,
  `measure_dnb_scan_noise_stats.py`) and the NuExtract fine-tuning pilot
  (`nuextract2_common.py`, `nuextract_baseline.py`,
  `finetune_nuextract.py`, `evaluate_nuextract_baseline.py`,
  `evaluate_nuextract_finetune.py`, `merge_nuextract_lora.py`,
  `prepare_nuextract_finetune_data.py`) — different infra (sklearn/pdfalto,
  no LLM calls), consumers of the corpus rather than part of the
  generation pipeline
- `evaluation/inference_endpoints.py` + `evaluation/kisski.py` — still
  used by `refresh_llm_cache.py` for the main workflow's own LLM-strategy
  cache; untouched by this migration
- `evaluation/scripts/add_toc_ground_truth.py` — operates across
  `pending`/`open-access`/`copyrighted-scans`, not DNB-specific
- Everything else under `evaluation/`

### Old-repo `CLAUDE.md` updates

Beyond deleting the "Arbitrating below-gate dnb-toc-only books" section
outright, `evaluation/CLAUDE.md`'s Step 1 ("Transcribe the table of
contents") has a live cross-reference into the corpus that's moving —
"check whether a DNB-digitized TOC scan already exists... look in
`evaluation/corpus/dnb-toc-only/manifest.json`". That path gets
repointed at the sibling-checkout location
(`../dnb-toc-ground-truth/data/corpus/pilot/manifest.json`, same
convention as the `--dnb-toc-corpus-dir` default below) rather than left
dangling.

## Corpus access from `chapter-segmentation` after the move

Sibling-checkout convention, matching the existing `../pdfalto` pattern:
`dnb-toc-ground-truth` is cloned next to `chapter-segmentation` on disk.
`evaluate_layout_toc_classifier.py` and `measure_dnb_scan_noise_stats.py`
gain a `--dnb-toc-corpus-dir` flag (default
`../dnb-toc-ground-truth/data/corpus/pilot`, overridable via
`DNB_TOC_CORPUS_DIR` env var, same override style as `PDFALTO_BIN`) so
`evaluation/harness.py`'s corpus-loading helpers can point at the new
location for this one corpus name instead of
`evaluation/corpus/dnb-toc-only/`. No submodule, no data duplication —
one copy of the corpus, read from two repos.

## Endpoint and config system

`evaluation/kisski.py`'s auto-discovery (`fetch_kisski_models`,
demand-based selection) is dropped entirely in the new repo — every
model must be named explicitly. `inference_endpoints.py` is forked and
extended:

**Endpoints file** (`.endpoints`, gitignored; `.endpoints.dist` committed
as a documented example) — the single source of truth for what
endpoints exist, accepting either shape:

1. **JSON array** (officially supported) — a list of objects as pasted
   from the provider dashboard. Only three fields are consumed per
   entry: `url`, `key`, and the model id — read from `model` if present,
   else parsed out of `framework_args`'s `--model=...` token (today's
   `_MODEL_ARG_RE` logic, reused as-is). Every other field
   (`framework`, `gpus`, `job_id`, `status`, ...) is ignored except
   `status`, used only for tie-breaking (see below).
2. **Plain-text pasted-session-table** (backward-compatible alternative)
   — today's existing tab-separated-block format
   (`load_mpcdf_sessions`/`_parse_session_block`), unchanged.

**Config file** (`.config`, gitignored; `.config.dist` committed) — JSON,
holding defaults for values otherwise passed as CLI flags:
`endpoints_file` path, `use_vision` / `use_text` model-id lists,
`concurrency`, `limit`, `gate_threshold`. A CLI flag always overrides its
config-file default; a value present in neither uses the script's
built-in default (mirrors argparse's normal precedence, just with an
extra config-file layer beneath the CLI).

**CLI**: `--endpoint`/`--config-file`/`--text-endpoint`/`--text-config-file`
(env-var-alias and single-purpose file flags) are replaced by:

```
--use-vision <model>[,<model>...]
--use-text <model>[,<model>...]
--endpoints-file PATH   # default: .endpoints, or config file's endpoints_file
--config-file PATH      # default: .config
```

`--use-vision`/`--use-text` name model ids to resolve against
`--endpoints-file`'s entries by exact model-id match. Two or more models
may be given for either side (see "N-way gating" below); at least one
vision model is required, `--use-text` is optional (mirrors today's
"pair one vision read with a text read" mode, generalized to N models
total across both). If a model id matches more than one endpoint entry,
prefer an entry whose `status` is `"Running"` (JSON format only); if that
still leaves more than one candidate, or the format has no `status`
field, raise a clear error naming the ambiguous id rather than silently
picking one.

**`OpenAICompatibleLLMClient`** and `ModelEndpoint` carry over unchanged
in shape; only the resolution functions around them change.

## N-way gating

Extends `gate_book`'s two-list agreement gate to N >= 2 extractions
without changing its core alignment algorithm: run today's exact
pairwise `align_toc_entries` + agreement-rate + near-identical-title
check (`evaluation/dnb_toc_matching.py`, unchanged) on **every pair**
among the N resolved endpoints' extractions. The book passes if **at
least one pair** clears the 0.90 agreement threshold; among passing
pairs, the one with the highest agreement rate is used, and that pair's
existing merge logic (union of matched pairs plus each side's
singletons, sorted by page number) produces the `.expected.json`
entries — identical to today's two-model output shape. This is the
literal reading of "only two of N need to agree": extra models just
widen the chance of finding an agreeing pair; the proven merge algorithm
and its extensive edge-case handling (title normalization, near-identical
threshold, page-equivalence) is reused verbatim rather than redesigned
for a multi-way consensus.

## Repo layout

```
dnb-toc-ground-truth/
  cli/
    fetch_corpus.py            # was evaluation/scripts/fetch_dnb_toc_corpus.py
    generate_ground_truth.py   # was evaluation/scripts/generate_dnb_toc_ground_truth.py
    arbitrate.py                # was evaluation/scripts/arbitrate_dnb_toc.py
    select_eval_sample.py       # was evaluation/scripts/select_dnb_toc_eval_sample.py
    README.md                   # --help dump per script (same convention as evaluation/scripts/README.md)
  data/
    corpus/
      pilot/
        pdf/                   # gitignored
        llm-cache/
        ground-truth/          # *.expected.json
        README.md              # static corpus size/status/content overview
  docs/
    superpowers/
      specs/
      plans/
    history.md                 # from evaluation/experiments/dnb-toc-ground-truth.md
  src/
    dnb_toc_ground_truth/
      __init__.py
      toc_entry.py              # vendored TocEntry, _parse_toc_page_number,
                                 # _toc_items_to_entries, parse_json_array
      vision.py                 # was dnb_toc_vision.py
      ocr.py                    # was dnb_toc_ocr.py
      matching.py                # was dnb_toc_matching.py, extended for N-way gating
      inference.py               # forked inference_endpoints.py, KISSKI-free, N-model resolution
      pdfalto_runner.py          # vendored copy
      corpus.py                  # slim harness.py equivalent: corpus_dir, load_manifest_books,
                                  # llm_cache_dir -- scoped to this repo's single corpus
  tests/
    test_vision.py  test_ocr.py  test_matching.py  test_arbitrate.py
    test_generate_ground_truth.py  test_fetch_corpus.py
    test_select_eval_sample.py  test_inference.py
  .gitignore
  .endpoints          # gitignored, real credentials
  .endpoints.dist      # committed example (JSON array shape)
  .config              # gitignored
  .config.dist          # committed example
  pyproject.toml        # uv + hatchling, mirroring chapter-segmentation's conventions
  README.md
```

CLI script filenames drop the redundant `dnb_toc` prefix (the whole repo
is already scoped to dnb-toc). Internal module names likewise drop it
(`vision.py` not `dnb_toc_vision.py`) since the package namespace
(`dnb_toc_ground_truth`) already carries that context.

## Top-level `README.md`

The one file a stranger to this project opens first, so it carries the
purpose statement `evaluation/CLAUDE.md`'s doc-organization note never
had to spell out (it could assume the reader already knew this was part
of `chapter-segmentation`). Two required parts:

**Purpose.** This repo is a pilot case for generating structured,
machine-checkable ground truth from openly available data — DNB's
CC0-licensed "Kataloganreicherung" table-of-contents scans — using
independent LLM reads gated against each other for agreement, with
human/Claude arbitration for the disagreements. The output
(`data/corpus/pilot/ground-truth/*.expected.json`) is meant as an input
to *other* pipelines, not an end in itself — e.g. fine-tuning a smaller
structured-extraction model (NuExtract), benchmarking chapter/TOC
extraction heuristics, or training a lightweight classifier — each of
which can consume this corpus without depending on this repo's own LLM
pipeline. State plainly that the LLM-based generation pipeline is the
means, not the point: the point is the ground-truth data itself, general
enough to feed pipelines this repo doesn't build.

**Setup instructions**, in the order a new clone actually needs them:
1. `uv sync` (Python >=3.12, matching `chapter-segmentation`'s
   convention)
2. External binaries on `PATH` (or via `--<tool>-bin`/env var, same
   convention as `PDFALTO_BIN`): `ocrmypdf` for the OCR-text extraction
   path, and a sibling `pdfalto` checkout for ALTO reconstruction (link
   to [kermitt2/pdfalto](https://github.com/kermitt2/pdfalto), same as
   `chapter-segmentation`'s own note)
3. Copy `.endpoints.dist` → `.endpoints` and `.config.dist` → `.config`,
   fill in real endpoint credentials (either supported format) and
   default model selections
4. `uv run python cli/fetch_corpus.py --help` through to
   `uv run python cli/generate_ground_truth.py --help` as a smoke check
   that the install works, before pointing either script at a real
   endpoint
5. A pointer to `cli/README.md` for full flag reference and to
   `data/corpus/pilot/README.md` for current corpus size/status

## Migration mechanics

1. Create a git worktree off `dnb-toc-ground-truth-wip` to build the new
   repo's initial content without disturbing the current working tree.
2. Initialize the new repo (git init, pyproject.toml, `.gitignore`), port
   code with the vendoring/renaming above, port and adapt tests.
3. Get `uv run pytest` fully green in the new repo before moving on to
   data or cross-repo wiring.
4. Copy `evaluation/corpus/dnb-toc-only/`'s full contents (tracked and
   gitignored alike) into `data/corpus/pilot/` in the new repo via
   filesystem copy — this is data transfer, not a git operation, since
   most of it (PDFs, caches) was never committed in the source repo
   either.
5. Wire `chapter-segmentation`'s classifier/NuExtract pilot scripts to
   the sibling-checkout path (`--dnb-toc-corpus-dir` /
   `DNB_TOC_CORPUS_DIR`), confirm they still run against the relocated
   corpus.
6. Stop and ask the user to start two new model endpoints and provide an
   `.endpoints` file for the new repo.
7. Smoke test: run `generate_ground_truth.py --limit N` against the real
   endpoints and confirm output parity with the old pipeline's behavior
   (agreement gate passes/fails as expected, `.expected.json` shape
   unchanged, caching/locking behaves the same).
8. Only once the smoke test succeeds: remove every dnb-toc-ground-truth-
   related file from `chapter-segmentation` (the moved code, tests, docs,
   and `evaluation/corpus/dnb-toc-only/`), leaving a one-line note in
   `evaluation/experiments/README.md` that the experiment moved to its
   own repository (with a pointer, no path/credentials).

## Testing

Every ported test file is adapted in place: fixtures/imports repointed
at the new package layout (`dnb_toc_ground_truth.*` instead of
`chapter_segmentation.segmentation`/`evaluation.*`), assertions
unchanged where the underlying logic is unchanged (vision/OCR extraction,
pairwise matching/merge). New or substantially-rewritten test coverage
is needed for:
- `inference.py`'s endpoints-file parsing (both JSON-array and
  plain-text shapes) and `--use-vision`/`--use-text` N-model resolution,
  including the ambiguous-match error path
- `matching.py`'s N-way "best pair wins" gating on top of the unchanged
  pairwise primitives
- `generate_ground_truth.py`'s CLI surface for the new flags

`test_kisski.py` is deleted outright (nothing left to test).

## Completion criteria

- `uv run pytest` fully green in the new repo, no KISSKI references
  remaining anywhere in it
- Real two-endpoint smoke test against user-provided endpoints succeeds
  and produces output consistent with the old pipeline
- `chapter-segmentation`'s classifier/NuExtract pilots still pass against
  the relocated corpus via the sibling-checkout path
- Old repo cleaned of all dnb-toc-ground-truth-specific files, with only
  a redirect note left in `evaluation/experiments/README.md`
