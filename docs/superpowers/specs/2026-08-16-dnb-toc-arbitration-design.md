# dnb-toc-only: Claude-arbitrated resolution of below-gate books

## 1. Problem

`generate_dnb_toc_ground_truth.py` (per
`docs/superpowers/specs/2026-08-16-dnb-toc-uniform-ocr-design.md`) gates
each book on agreement between two independent vision-model TOC
extractions. Across three smoke-test rounds against the same 15-book
sample (see `evaluation/RESULTS.md` § "dnb-toc-only ground truth:
two-vision-model gate" and `evaluation/EXPERIMENTS.md` for the full
history), the pass rate has settled around 53-60%: books below the 0.90
agreement threshold are discarded outright -- no `.expected.json` is
written, and the two models' conflicting work is simply left unused.

The diagnosed remaining disagreement patterns (content omission by one
model, front/back-matter inclusion disagreements, two-line-title
splitting, nested sub-point granularity) are all cases where a careful
reader -- comparing the two lists, and checking the actual TOC page
images when the lists alone don't settle it -- can almost always tell
which side is right, or reconcile a hybrid of both. That reconciliation
work is exactly what discarding the book throws away.

## 2. Goal

Instead of discarding a below-threshold (or single-model-error) book,
preserve enough information for a strong reasoning model (Claude, acting
in an interactive session -- not a scripted API call) to arbitrate the
conflict and produce a final, trusted `.expected.json` for it, pushing
the corpus's realized ground-truth yield from ~53-60% of attempted books
towards the "almost 100%" the auto-gate alone can't reach on its own,
without discarding the two models' partial agreement.

## 3. Non-goals

- **Not an automated third-model API call.** The arbitrator is a Claude
  Code session working through flagged books interactively, per the raw
  data a reporting script surfaces -- not a new stage the pipeline calls
  out to unattended. (Considered and explicitly rejected: real per-book
  API spend, and no one positioned to catch judgment calls a script
  can't make well, e.g. "is this line a part-divider or a real
  chapter.")
- **Not a new schema for preserving model output.** Each book's raw
  per-model extraction is already preserved today --
  `generate_dnb_toc_ground_truth.py` never deletes
  `llm-cache/<key>.<model>.json` when a book fails the gate, it only
  skips writing `.expected.json`. Nothing new needs to be added to keep
  that data around; only a way to surface and diff it is missing.
- **Not a replacement for the two-vision-model gate.** Books that
  already pass at >= 0.90 agreement are untouched by this design.

## 4. Architecture

### 4.1 `diff_toc_entries` (new, in `evaluation/dnb_toc_matching.py`)

```python
def diff_toc_entries(
    a: list[TocEntry], b: list[TocEntry],
) -> tuple[list[tuple[TocEntry, TocEntry]], list[TocEntry], list[TocEntry]]:
    """Returns (matched_pairs, only_in_a, only_in_b) using the same
    alignment `gate_book` uses internally."""
```

Factored out of `gate_book`'s existing matched/singleton computation
(`align_toc_entries`, then partitioning `a`/`b` by matched index). Pure
refactor: `gate_book` calls this internally and its own behavior/tests
are unchanged.

### 4.2 Cache helpers move to `evaluation/dnb_toc_vision.py`

`_cache_path`, `_load_cached_llm_entries`, `_write_cached_llm_entries`
currently live in `evaluation/scripts/generate_dnb_toc_ground_truth.py`.
They move to `evaluation/dnb_toc_vision.py` (renamed without the leading
underscore: `cache_path`, `load_cached_llm_entries`,
`write_cached_llm_entries`) since they cache `vision_extract_toc_entries`
results and both `generate_dnb_toc_ground_truth.py` and the new
arbitration script need to read them, without one script importing
another. `generate_dnb_toc_ground_truth.py`'s call sites update to the
new import location and names; behavior is unchanged. The existing
direct tests of these functions in `tests/test_generate_dnb_toc_ground_truth.py`
move to `tests/test_dnb_toc_vision.py` along with the functions.

### 4.3 New script: `evaluation/scripts/arbitrate_dnb_toc.py`

**Reporting and rejection-recording only -- it never decides.**

```
uv run python evaluation/scripts/arbitrate_dnb_toc.py [list]
```

Scans `evaluation/corpus/dnb-toc-only/llm-cache/` for book keys that:
- have at least one cached per-model result, AND
- have no `evaluation/corpus/dnb-toc-only/<key>.expected.json`, AND
- are not already present in `arbitration-rejected.json` (§4.4).

For each such book, prints:
- the book's key, title (from `manifest.json`), and PDF path;
- the two model names and their raw entry counts (or, if only one model
  has a cached result, a note that the other returned no usable
  response);
- the matched-pair count and agreement rate (via `diff_toc_entries`);
- every unmatched entry from each side, with its title and printed page
  number, so the actual disagreement is visible without opening any
  other file.

```
uv run python evaluation/scripts/arbitrate_dnb_toc.py reject <key> "<reason>"
```

Appends `{"key": <key>, "reason": "<reason>", "rejected_at": "<today's
date, YYYY-MM-DD>"}` to `arbitration-rejected.json` (creating the file
if it doesn't exist). Errors (non-zero exit, clear message) if `<key>`
is already present in the rejected list, rather than silently
overwriting -- a second rejection of the same book is almost always a
mistake (wrong key typed, or forgetting it was already handled) rather
than an intentional reason update.

There is no "accept" subcommand. Writing the final `.expected.json` is a
direct reuse of `toc_entry_to_gt_dict` (already in
`evaluation/dnb_toc_matching.py`) plus `json.dumps` with `"verified":
true` -- the same two lines `generate_dnb_toc_ground_truth.py` already
uses for a passing book, just with the boolean flipped. Not worth a
dedicated command.

### 4.4 New data file: `arbitration-rejected.json`

`evaluation/corpus/dnb-toc-only/arbitration-rejected.json`:

```json
{
  "rejected": [
    {"key": "9783515114868", "reason": "both models hallucinate distinct entries on the same ambiguous scan; page images too degraded to resolve", "rejected_at": "2026-08-16"}
  ]
}
```

Created on first use by the `reject` subcommand. Committed to the repo
(not gitignored) -- it's part of this corpus's ground-truth-generation
state, same tier as `manifest.json`.

## 5. Workflow (documented in `evaluation/CLAUDE.md`, see §7)

1. Run `generate_dnb_toc_ground_truth.py` as today. Passing books get
   `.expected.json` (`"verified": false`).
2. Run `arbitrate_dnb_toc.py` (no args) to list every remaining book
   that needs a decision.
3. For each: read the printed diff. The four previously-diagnosed
   patterns (content omission, front/back-matter inclusion
   disagreement, two-line-title splitting, nested sub-point count
   mismatch -- see `evaluation/RESULTS.md`) usually make the right call
   obvious from the text alone.
4. When the text alone doesn't settle it, open the actual TOC page
   images: `Read` on the book's PDF with `pages: "N-M"` (1-based viewer
   pages -- same convention as `evaluation/CLAUDE.md`'s existing Step 3
   for hand-transcribing ground truth).
5. Write `evaluation/corpus/dnb-toc-only/<key>.expected.json` directly
   (same schema as a passing book, `"verified": true`).
6. If truly unrecoverable, run
   `arbitrate_dnb_toc.py reject <key> "<reason>"` instead, so it's never
   resurfaced by a future arbitration pass.

## 6. Testing

- `diff_toc_entries`: unit tests covering full agreement, partial
  agreement, and complete disagreement (mirrors `gate_book`'s existing
  test cases, since it's the same underlying computation).
- Cache helpers: existing tests move to `tests/test_dnb_toc_vision.py`
  unchanged in behavior.
- `arbitrate_dnb_toc.py`: unit tests against fixture `llm-cache/`
  directories covering (a) a two-model disagreement report, (b) a
  single-surviving-model report, (c) a book already `.expected.json`'d
  being excluded from the listing, (d) a book already in
  `arbitration-rejected.json` being excluded, (e) `reject` creating the
  file fresh, appending to an existing one, and erroring on a duplicate
  key.

## 7. Documentation

New section in `evaluation/CLAUDE.md`, "Arbitrating below-gate
dnb-toc-only books," describing the §5 workflow as the standard
procedure to run after `generate_dnb_toc_ground_truth.py` whenever
books remain below the gate.

## 8. Open questions

None -- scope confirmed via brainstorming session on 2026-08-16 (see
conversation history: arbitrator is Claude Code in-session, not a
scripted API call; output is a direct `.expected.json` write, not a
reconciled list fed back through `gate_book`'s merge; single-model
error books go through the same queue; rejections are permanently
recorded; this is meant as a reusable tool, not a one-off pass).
