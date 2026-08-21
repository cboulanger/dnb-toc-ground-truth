# dnb-toc-ground-truth history

Full write-up for every superseded run and diagnosis behind this
project's ground-truth-generation pipeline -- "Current status" below is
expected to go stale and be rewritten as the pipeline changes or more of
the corpus gets ground truth; the sections beneath it hold the full
history so the reasoning and dead ends behind the current numbers aren't
lost.

## Current status

`dnb-toc-only` has **612 books with ground truth** out of ~1251 in the
manifest (~639 never yet attempted). Each passed either the automated
two-vision-model >=90%-agreement gate (`"source": "bulk_gate"`,
`"verified": false`) or direct human/Claude review against the real PDF
page images (`"source": "claude_arbitration"`, `"verified": true`).
Books currently used as the two vision-capable models:
`Qwen/Qwen3-Omni-30B-A3B-Instruct` (also cached under the older
KISSKI-era ID `qwen3-omni-30b-a3b-instruct` -- see the data gotcha
below) and `mistralai/Mistral-Small-3.2-24B-Instruct-2506`.

### Known model weaknesses

**Mistral-Small-3.2-24B-Instruct-2506:**

- Defaults `printed_page_number` to null far more often than it should,
  even when a real page number is clearly legible -- in both vision mode
  (fed page images) and OCR-text mode (fed OCR'd text). The gate's
  alignment logic never matches a null-page entry against a known-page
  one, so this alone tanks agreement rate even when the title-level
  reading is otherwise fine. When a Mistral-vs-Qwen disagreement shows
  many `?` pages on Mistral's side, suspect this first.
- Splits a bare author-name line (printed on its own line above/below a
  title, sharing that title's page number) into its own spurious,
  page-number-less entry instead of merging it into the following
  title's `authors` field -- the single most common structural defect.
- Can confidently misread an entirely different page as the real TOC,
  fabricating a plausible-looking but wrong entry list.
- Occasionally returns malformed JSON.
- In OCR-text mode, further degraded by the corpus's default
  `tessdata_fast` OCR quality (`tessdata_best` is not installed) --
  garbled diacritics (ü→ii, ß→fs), mangled non-Latin scripts, dropped or
  duplicated words.

**Qwen3-Omni-30B-A3B-Instruct** (stronger overall, not error-free):

- Fabricates a page number for a part/section divider by copying the
  following chapter's page number, instead of recognizing the divider
  has none of its own.
- Silently drops real content -- entries, occasionally whole chapters --
  with no error or warning.
- Pulls non-TOC page furniture (title-page imprint lines, running
  headers/folios) into the entry list as if they were real TOC entries.

**Shared -- the most important thing to remember when trusting any
agreement number:**

- Two independent reads, even from two genuinely different models, can
  confidently agree on the exact same wrong answer. 100% agreement is
  evidence, not proof -- open the real PDF page image even for a "clean"
  agreement, not just for disagreements.
- Two-line titles (main title + subtitle sharing one page number) get
  incorrectly split into two entries by either model, in either
  direction.
- The automated diff's alignment is greedy and order-preserving: one
  spurious extra/missing entry near the top of a book's list can make
  everything after it look like a disagreement even when the underlying
  readings mostly agree -- check the whole sequence, not just individual
  mismatched pairs, before concluding a book is a genuine mess.

### A data gotcha to watch for

`Qwen/Qwen3-Omni-30B-A3B-Instruct` (MPCDF-hosted) and
`qwen3-omni-30b-a3b-instruct` (older KISSKI-hosted) are cached under two
different literal model-ID strings but are the same underlying model.
The pipeline does not canonicalize this -- a book whose only two cached
"models" are these two labels has NOT been cross-model-verified, no
matter how well they agree with each other. Treat such a book as a
single-model book (verify directly against the real page image) rather
than trusting the apparent agreement.

## History


The subsection below is the first real smoke test's write-up, whose
root-cause diagnosis (a genuine editorial granularity difference between
the two models) was itself superseded by a more careful follow-up
investigation that found the real cause was `gemma-4-31b-it` silently
dropping content, not a deliberate judgment call -- see the next
subsection, "Model swap to qwen3.6 family", for the corrected diagnosis
and the fix.

### First real smoke test (2026-08-16) -- initial (incomplete) diagnosis

Per `docs/superpowers/specs/2026-08-16-dnb-toc-uniform-ocr-design.md` and
`docs/superpowers/plans/2026-08-16-dnb-toc-vision-extraction.md`,
`generate_dnb_toc_ground_truth.py` was migrated from a regex-heuristic +
text-LLM gate to a two-independent-vision-model gate (each model reads
the book's page images directly via `pdftoppm`, no OCR/text layer at
all). First real run against the live corpus and live KISSKI models,
after the migration and two follow-up robustness fixes (`max_tokens`
escalation on truncated responses; `_select_best_models` now takes
multiple candidates from one pattern before falling through):

```
uv run python evaluation/scripts/generate_dnb_toc_ground_truth.py --limit 15 --concurrency 4

Vision models used: qwen3-omni-30b-a3b-instruct, gemma-4-31b-it
6/15 books passed the gate and got .expected.json written.
  8 skipped: below_threshold
  1 skipped: error: JSONDecodeError
```

**40% pass rate is much lower than the near-perfect results the design
spec's own two-book prototype found** (18/18 and ~18/18 entries,
§2.1). Root-caused by comparing the two models' cached raw responses
directly for four `below_threshold` books:

| Book | Pages | qwen entries | gemma entries |
| --- | --- | --- | --- |
| `0745309941` | 2 | 8 | 2 |
| `3465016874` | 2 | 17 | 3 |
| `3492038174` | 7 | 135 | 24 |
| `3571092120` | 3 | 41 | 32 |

`gemma-4-31b-it` isn't truncating -- confirmed directly for
`3492038174` (the most extreme case): both models' entry lists end at
the exact same final item (page 313, "Zur Gründung einer »Stiftung
Weltethos«"), so gemma read every page and reached the true end of the
document. **The two models are making a genuinely different editorial
judgment about what counts as one "chapter" entry** on TOCs with deep
hierarchical nesting (numbered theses/aphorisms, sub-points under a
numbered heading): qwen extracts nearly every numbered sub-line as its
own entry, gemma collapses them into far fewer higher-level entries.
Where a TOC is flat (the design spec's two prototype books, and this
run's simpler passing books), both models agree closely and the gate
passes fine -- the mismatch is specific to densely-nested layouts.

**This diagnosis turned out to be incomplete** -- see the next
subsection, "Model swap to qwen3.6 family": comparing entry *page-number ranges* (not just
counts) across all 15 books showed gemma's range started dramatically
later than qwen's on 5 of 8 mismatched books, including flat, simple
TOCs where no granularity judgment call was plausible (a clean 8-entry
numbered list came back with only its last 2 entries) -- a real
reliability gap in `gemma-4-31b-it` on this task, not a considered
editorial choice. `3492038174`'s matching final entry was a coincidence
of that book happening to also have a genuine granularity disagreement
layered on top of the range problem, not evidence against it.

### Model swap to qwen3.6 family (2026-08-16) -- pass rate 60%, before the granularity-prompt fix

This subsection is the write-up of the run immediately after
`_VISION_MODEL_PATTERNS`' second pattern was swapped from `gemma-4-31b-it`
to the qwen3.6 family (fixing the content-dropping reliability gap above),
but before `_VISION_TOC_EXTRACTION_PROMPT` was clarified to handle
nested-TOC sub-points consistently -- that follow-up fix is what
superseded this run's numbers.

**Corrected root cause (of the first smoke test's 40%):** comparing entry
page-number *ranges* (not just counts) across all 15 books showed
`gemma-4-31b-it`'s range started dramatically later than `qwen3-omni`'s on
5 of 8 mismatched books -- including a clean, flat 8-entry numbered list
(`0745309941`) that came back with only its last 2 entries. This is a
reliability gap in `gemma-4-31b-it` on this task (silently dropping the
early portion of a multi-image request), not a considered editorial
choice about chapter granularity. Spot-checked `qwen3.6-27b` directly
(`vision_extract_toc_entries`, live KISSKI) against the same books: it
correctly covered the full page range every time, matching
`qwen3-omni`'s own range. `_VISION_MODEL_PATTERNS`' second pattern was
changed from `gemma-<N>-` to `qwen<N>.<M>-` accordingly.

**Re-run with the corrected model pair, same 15 books:**

```
uv run python evaluation/scripts/generate_dnb_toc_ground_truth.py --limit 15 --concurrency 4

Vision models used: qwen3-omni-30b-a3b-instruct, qwen3.5-122b-a10b
9/15 books passed the gate and got .expected.json written.
  4 skipped: below_threshold
  1 skipped: error: JSONDecodeError
  1 skipped: error: ValueError
```

**Pass rate improved from 40% to 60%, and the improvement is for the
right reason** -- confirmed by comparing page-number ranges again
across all 15 books: every single book now shows matching or
near-matching ranges between the two models, with zero "dropped early
content" cases remaining. The 4 remaining `below_threshold` books
(`3465016874`: 17 vs 14 entries; `3571092120`: 41 vs 33;
`9783842331976`: 57 vs 12; and the still-failing `3492038174`, see
below) all have matching ranges but differing entry *counts* -- this is
the genuine chapter-granularity disagreement on densely-nested TOCs
(numbered theses/sub-points under a numbered heading) originally
(mis)diagnosed in the first run. This is a narrower, better-understood
remaining problem than before: pipeline reliability is no longer in
question, only how consistently the two models segment deeply nested
TOC hierarchies into "one entry per chapter."

The `1 error: ValueError` is new in this run: `3492038174` (the
7-page, most deeply-nested book) got an *empty* response from
`qwen3.5-122b-a10b` (`"No JSON array found in LLM response: ''"`) --
not yet root-caused; may be specific to that model/book pair rather
than the family generally, since `_VISION_MODEL_PATTERNS`' second
pattern matches any `qwen<N>.<M>-` model and a busy-driven re-run could
pick a different specific model next time. The pre-existing
`1 error: JSONDecodeError` (`383050277X`) is unchanged from the first
run -- still not root-caused, still survives the `max_tokens`
escalation (so it's a genuinely malformed response shape, not
truncation).

**Open question, not yet resolved at the time:** whether to (a) tune the
prompt to make "chapter" granularity more explicit/consistent on
deeply-nested TOCs specifically, (b) accept a lower gate threshold for
such books, or (c) accept the current ~60% pass rate as-is. Resolved by
the granularity-prompt fix described in the next subsection,
"Granularity-prompt fix and re-run" -- see there for what was tried and
its effect.

### Granularity-prompt fix and re-run (2026-08-16) -- pass rate 53%, before the arbitration tool

This subsection is the write-up of the run
immediately after `_VISION_TOC_EXTRACTION_PROMPT` was clarified to fix
the nested-sub-point granularity problem, but before
`arbitrate_dnb_toc.py` existed to resolve the books that still didn't
clear the gate -- at this point in the investigation, a below-threshold
book was still simply discarded.

`_VISION_TOC_EXTRACTION_PROMPT` was clarified to explicitly call out
that indented/numbered/lettered sub-points each carry their own page
number and are their own entry, not to be collapsed into their parent
heading. Clean re-run, same 15 books, fresh cache:

```
uv run python evaluation/scripts/generate_dnb_toc_ground_truth.py --limit 15 --concurrency 4

Vision models used: qwen3-omni-30b-a3b-instruct, qwen3.6-35b-a3b
8/15 books passed the gate and got .expected.json written.
  5 skipped: below_threshold
  2 skipped: error: ValueError
```

**The fix worked exactly as intended on the case it targeted**:
`9783842331976` (the deeply-nested book previously diagnosed as 57 vs 12
entries) now matches 56 of 57 entries (rate 0.98, PASS) -- the nesting
instruction resolved that specific failure mode cleanly.

**But the aggregate pass rate did not improve (53% vs the prior run's
60%)**, because a different, previously-undiagnosed cluster of
disagreements dominates the remaining 5 `below_threshold` books.
Inspecting each below-threshold book's two entry lists side by side (not
just counts) shows this is NOT the nesting problem recurring -- it's a
mix of:

- **Genuine content omission, reliability not editorial choice**:
  `0745309941` -- `qwen3-omni` silently dropped one entire chapter
  ("Gender, Migration and Cross-Ethnic Coalition Building", p.48) that
  `qwen3.6` caught; a flat, simple 8-vs-9-entry book with no nesting at
  all. Note the direction is reversed from the earlier gemma finding --
  this time it's `qwen3-omni` that drops content, on a book unrelated to
  granularity.
- **Whether front/back matter should be its own entry at all**
  (`380061832X`: `qwen3.6` added "Vorwort" and "Autorenverzeichnis" that
  `qwen3-omni` correctly omitted per the "skip acknowledgements..."
  instruction; `3823350242`: `qwen3-omni` included a bibliography-like
  "Verzeichnis der Schriften von..." appendix entry that should have been
  skipped). This is the same "bulk vs eval tier target definition"
  question flagged as an open, undecided issue in the vision-extraction
  implementation's final code review -- not a new problem, but now
  visibly the dominant cause of gate failures.
- **Two-line TOC entries (a title line plus a subtitle/continuation
  line) being split into two entries by one model but correctly merged
  by the other** (`3779912511`, `9783515114868`): one model sometimes
  treats a part-header ("Geschichte der Pädagogik") and the chapter title
  that follows it as two separate entries (one with `printed_page_number:
  null`), while the other merges the header into the chapter's own title.
  This is the mirror image of the nesting problem the prompt fix just
  solved -- there, sub-points were wrongly merged into a parent; here,
  a title and its own continuation are wrongly split apart.

**The `2 error: ValueError` books both got an empty response** (`"No
JSON array found in LLM response: ''"`) from one model:
`qwen3.6-35b-a3b` on `3465016874`, and (no cache file written at all,
implying the failure happened before any content came back)
`qwen3.6-35b-a3b` on `3492038174` -- the same still-unresolved empty-
response failure mode as previous runs, now hitting a different specific
qwen3.6 sub-model (`_select_best_models` picks whichever qwen3.6 variant
is least busy at request time, so the exact model varies run to run).
The pre-existing `383050277X` `JSONDecodeError` from earlier runs did
NOT recur this time -- it happened to pass cleanly (rate 1.00) in this
run instead, consistent with it being a live-service flakiness case
rather than a deterministic per-book failure.

**Open question at the time:** whether to (a) fix the front/back-matter
prompt-adherence gap, (b) build a way to resolve below-threshold books
instead of discarding them, or (c) accept the current ~53% pass rate.
Resolved by building the arbitration tool
(`docs/superpowers/specs/2026-08-16-dnb-toc-arbitration-design.md`) --
see the next subsection, "Model-pairing search via MPCDF", for the
result.

### Model-pairing search via MPCDF: KISSKI, InternVL, Pixtral, GLM-4.5V, GLM-4.1V-9B-Thinking (2026-08-16 to 2026-08-20)

Per `docs/superpowers/specs/2026-08-16-dnb-toc-uniform-ocr-design.md` and
`docs/superpowers/plans/2026-08-16-dnb-toc-vision-extraction.md`,
`generate_dnb_toc_ground_truth.py` runs a two-independent-vision-model
gate (each model reads the book's page images directly via `pdftoppm`,
no OCR/text layer at all). Three smoke tests (40% with `gemma-4-31b-it`
as the second model; 60% after swapping to the qwen3.6 family; 53% after
a further granularity-prompt fix, whose aggregate rate didn't improve
because a different disagreement cluster then dominated) diagnosed and
fixed a content-dropping reliability gap and a nested-sub-point
granularity gap, and identified front/back-matter inclusion
disagreements as the next-largest remaining cause of gate failures --
see
"History" below for all three runs' full write-ups.

**Front/back-matter prompt fix, plus an arbitration tool for whatever
still doesn't clear the gate (2026-08-16):** `_VISION_TOC_EXTRACTION_PROMPT`
was made explicit that front matter, back matter, and part/section
dividers never get their own entry, and that a two-line title (main
title + subtitle sharing one page number) is a single entry, not two --
see `docs/superpowers/specs/2026-08-16-dnb-toc-arbitration-design.md`.
Rather than treat the gate as the final word, a new
`evaluation/scripts/arbitrate_dnb_toc.py` surfaces exactly what each
model extracted for any book that doesn't clear the gate (or where one
model returned nothing usable), so a Claude Code session can resolve it
by hand -- reading the diff, and opening the actual TOC page images when
the text alone doesn't settle it -- instead of the book being silently
discarded.

**Result on the same 15-book sample: 15/15 (100%) now have ground
truth**, up from 8/15 (53%) auto-gated alone:

```
uv run python evaluation/scripts/generate_dnb_toc_ground_truth.py --limit 15 --concurrency 4
# 8/15 passed the gate automatically ("source": "bulk_gate")

uv run python evaluation/scripts/arbitrate_dnb_toc.py
# surfaced the remaining 7 (5 below_threshold + 2 empty-response errors)
# each resolved by hand and written with "source": "claude_arbitration"
```

Arbitrating the 7 remaining books surfaced real extraction errors that
neither model's raw output alone would have caught, beyond the
already-diagnosed disagreement categories: an off-by-one page number on
6 entries in `3571092120` (the model read a preceding part-divider's own
page number instead of the actual chapter's, e.g. attributing page 79 --
the "Erkenntnistheorie des Rechts" section header's page -- to the
chapter that starts on page 80), a misspelled author name in two
different books (`3571092120`: "Jürgen Rödiger" for "Jürgen Rödig";
`9783515114868`: "Bodo V. Borries" vs. the correct "Bodo von Borries"
elsewhere in the same book's own author list), a spurious bibliography
entry in `9783842331976` ("Literaturverzeichnis", which the prompt
already says to skip but the model included anyway), and one book
(`3465016874`) where both models badly mishandled a 3-level nested
structure (part headers with roman-numeral subsections) badly enough
that it needed full hand-transcription from the page images rather than
reconciling either model's list.

**Still open, not blocking**: the empty-response failure mode on
`qwen3.6`'s side (2 of these 15 books hit it this run) remains
un-root-caused -- live-service flakiness is the leading hypothesis
(different specific qwen3.6 sub-model each run, and the same book
doesn't fail consistently across runs), but arbitration means it no
longer blocks ground-truth coverage, only adds arbitration work.

**Scaling generation to the rest of the corpus (2026-08-17):**
`generate_dnb_toc_ground_truth.py` was changed to skip books that already
have a `.expected.json` or are in `arbitration-rejected.json` (previously
`--limit N` always re-processed the same first-N books in manifest order,
which made repeated invocations useless for advancing through a large
corpus) -- see the function's own `_still_needs_a_decision` docstring.
Running it in successive `--limit 100` batches, each followed by
`arbitrate_dnb_toc.py`, took `dnb-toc-only` from 15 to 170 books with
ground truth. KISSKI rate-limited 30-60% of a typical 100-book batch even
at `--concurrency 4`; per-book errors are cheap to retry (a book without
`.expected.json` just gets re-attempted in the next batch, and any model
whose response was already cached is reused for free), so this cost wall
time, not correctness.

A real batch run did stall completely for several minutes with zero
throughput, though -- root cause: `_run_book`'s concurrency semaphore
wrapped the *entire* per-model retry sequence, including the backoff
`sleep()` between attempts. When enough books hit `RateLimitError` around
the same time, every concurrency slot ended up asleep in backoff
simultaneously, blocking all other pending books from even starting a new
attempt -- confirmed live via `ps` (the stalled process had accrued only
~8s of CPU time across ~3h of wall time) and `lsof` (zero open
connections, so it wasn't hung on a frozen socket, just sleeping). Fixed
by narrowing the semaphore to wrap only the individual API call inside
the retry closure, so it's released during backoff and other books can
proceed -- regression test:
`test_semaphore_is_released_during_backoff_sleep` in
`tests/test_generate_dnb_toc_ground_truth.py`. The same fix incidentally
un-broke a test-suite slowdown introduced alongside the rate-limit-aware
backoff (attempts 3->6, base delay 1s->2s): tests exercising `_run_book`'s
retry paths didn't inject a mock `sleep`, so they burned real wall time on
every real backoff -- `_run_book` now accepts an injectable `sleep` too.

A second, distinct stall showed up immediately after that fix, on the very
next batch: `lsof` on the running process showed 4 TCP connections to
KISSKI's real host stuck `ESTABLISHED` for 20+ minutes with the process
barely using any CPU -- not a client-side backoff sleep this time (no
connections would be open for that), but requests actually in flight and
not returning. Root cause: the `AsyncOpenAI` client was constructed with
no explicit `timeout`, so it fell back to the SDK's own default (600s read
timeout) -- one slow/stuck KISSKI response could occupy a concurrency
slot for up to 10 minutes per attempt, times up to 6 retry attempts, a
worst case over an hour for a single book. Fixed by passing `timeout=90.0`
to the `AsyncOpenAI(...)` call in `_generate` -- generous for a 1-4 page
TOC scan's vision call, but bounds the worst case to something a retry
loop can actually recover from within a batch's lifetime.

**Third stall, same session: a genuine daily quota, not a bug.** After
both fixes above, a fresh batch still made zero progress in its first
3 minutes, repeatedly -- but an isolated single call (no retry wrapper,
no concurrency) failed in 1.6s with `RateLimitError`, ruling out a hang.
The 429 response's own headers settled it precisely (`e.response.headers`
on the raised `RateLimitError`):
`x-ratelimit-limit-day: 1000` / `x-ratelimit-remaining-day: 0` /
`retry-after: 54179` (seconds) -- the account's daily quota (1000
requests) was fully spent by this session's batches 1-4, resetting at a
clean midnight UTC (minute/hour/month limits still had headroom, so day
was specifically the binding one). No further batches are worth
attempting until the daily reset; `_call_with_retry`'s backoff, however
long, cannot recover from a quota that's already at zero. Corpus stood at
206/1251 `dnb-toc-only` books with ground truth (up from 15) when this
was hit.

**Fourth exhaustion, `retry-after` header now confirms hour AND day are both
binding simultaneously (2026-08-18):** a fresh batch resumed once daily
quota reset, made real progress (206 -> 238 books, ~748 individual
model-calls cached), then hit a wall again. Direct header inspection at the
moment of failure: `x-ratelimit-limit-day: 1000` / `remaining-day: 0`,
`x-ratelimit-limit-hour: 200` / `remaining-hour: 0`,
`x-ratelimit-limit-minute: 30` / `remaining-minute: 30` (untouched),
`retry-after: 65597` (~18.2h, resetting at the same clean-midnight-UTC
pattern). Motivated a retry-scheduling fix:
`_call_with_retry` (`evaluation/scripts/generate_dnb_toc_ground_truth.py`)
now reads `retry-after`/`x-ratelimit-remaining-<window>` directly off a 429
response and sleeps the server's own reported delay for an inline-
recoverable "hour"/"minute" window, but gives up immediately (no further
attempts) when the binding window is "day" -- a day-scale reset cannot
happen within one script invocation, so the prior blind linear backoff (up
to ~5min/book x up to 6 attempts) burned real wall time re-discovering
that same fact one book at a time (the ~6.5h run above lost ~91% of its
attempted books this way). See `_binding_rate_limit_window`/
`_retry_after_seconds`'s own docstrings for the exact window-priority
logic; regression tests in `tests/test_generate_dnb_toc_ground_truth.py`
(`TestCallWithRetry`, `TestBindingRateLimitWindow`, `TestRetryAfterSeconds`).

**Spot-check of the bulk-tier gate's real precision (2026-08-19):** the
two-vision-model >=90%-agreement gate only measures whether the two models
*agree*, not whether they're both right -- raising the question of whether
a same-family model pairing (`qwen3-omni-30b-a3b-instruct` +
`qwen3.6-<N>`, see below) might share a correlated blind spot invisible to
the gate itself. Measured directly: 25 books randomly sampled from the 179
`"verified": false` bulk-tier books, each visually reviewed against its
real PDF scan (5 background Claude Code subagents, 5 books each, using the
`Read` tool's image rendering the same way `arbitrate_dnb_toc.py`'s
human-in-the-loop review already works) -- effectively running
`generate_dnb_toc_ground_truth.py --spot-check`'s Accept/Reject protocol
without needing a live terminal or new KISSKI calls.

Naive result: only 7/25 (28%) fully matched their scans. But 16 of the 25
sampled books turned out to still be pre-2026-08-17-schema files (no
`skip` field on any entry at all -- not yet reprocessed by the current
pipeline, purely a backlog/staleness artifact, see `_is_stale_bulk_gate_entry`),
and EVERY one of those 16 failed for the exact same, already-diagnosed,
already-fixed reason: front-matter/back-matter/part-divider lines silently
omitted rather than recorded with `skip: true`. Restricting to the 9
sampled books already on the current schema -- i.e. what the pipeline
actually produces today -- gives **7/9 (78%) precision**, a small but
real sample.

The two current-schema rejects are genuinely informative, and inspecting
both models' raw cached responses for `9783495485019` directly
(`evaluation/corpus/dnb-toc-only/llm-cache/v2/9783495485019.<model>.json`)
pins down the exact mechanism for that one -- **neither defect there is
actually a hallucination**:

- `9783495485019`'s "Anhang:" duplicate: both models independently read
  the exact same real text, both correctly with an unknown (`null`) page
  number -- not a disagreement at all. But `align_toc_entries` skips any
  pair where *either* side's `printed_page_number` is `None` before ever
  attempting a title comparison (`if entry_a.printed_page_number is None:
  continue`), so two entries that agree perfectly can never be recognized
  as a match; both survive into the merged output as unmatched
  singletons, producing a duplicate. A page-number-less-entry blind spot
  in the alignment algorithm, not a hallucination and not really an
  "unconfirmed singleton" either -- the second model DID confirm it, the
  merge logic just has no way to record that.
- `9783495485019`'s split heading: `qwen3.6` (side `b`) read
  "Einleitung: Endlichkeit und Verantwortung" correctly as one entry;
  `qwen3-omni` (side `a`) split it into "Einleitung:" and "Endlichkeit
  und Verantwortung". Because "Einleitung:" is a prefix of `b`'s full
  title, it fuzzy-matched above threshold and consumed that pairing --
  and `gate_book` always keeps side `a`'s title verbatim on a matched
  pair, never side `b`'s, so the objectively worse (split) reading won;
  `a`'s own leftover second half then had nothing left to match and
  became a stray singleton. Real text on both sides throughout; the
  defect is entirely `gate_book`'s "always trust `a`" merge policy.
- `0292746245`: an author name typo ("Irving Davis" for the correctly-
  printed "Irvine Davis"), plus an internally-inconsistent `skip`
  classification (its own "Index" entry marked `skip: false` despite the
  file correctly marking its own "Contents" entry `skip: true`) -- raw
  per-model cache not inspected for this one, so whether this specific
  typo came from one model or both is unconfirmed.

None of this looks like the two models independently making the *same*
mistake (contrast with the earlier `gemma-4-31b-it` content-dropping bug,
independently confirmed via page-range comparison against a third
reading) -- both `9783495485019` defects are `gate_book` merge-policy
gaps with real, model-agreed-upon text on both sides. This narrows (does
not eliminate) the "risk of trusting a singleton" concern that motivated
this whole spot-check: at least in this sample, a singleton's title text
itself was never fabricated -- the risk actually realized was a null-page
entry never getting matched at all (pure duplication) and a matched
pair's arbitrary side-`a`-wins tiebreak discarding a better available
reading, not invented content slipping through unchecked. Real
chapter-level content (titles/authors/page numbers for actual chapters,
as opposed to the divider/front/back-matter lines `skip` exists to mark)
was reliably accurate across nearly all 25 books, current- and
stale-schema alike; the handful of exceptions were isolated single-
character OCR-style typos (e.g. "Urteitskraft"/"Urteilskraft",
"Cotidianeidad"/"Cotidianidad"), not a systematic pattern.

**Conclusion**: the same-family model pairing is not obviously the
dominant risk here -- the measured 78% current-schema precision is
already explained by `gate_book`'s lenient merge policy (structural,
model-family-independent) plus the two isolated single-model errors above,
with no case found of both models independently producing the identical
wrong answer. The bulk of the naive 28% number is pipeline staleness
(pending regeneration once quota allows), not a correlated-bias finding.
Not yet acted on: two concrete, differently-costed fixes the raw-cache
inspection above points to directly. Cheap, low-risk: let
`align_toc_entries` also match two entries whose page numbers are BOTH
`None` when their titles agree closely (today it skips the pair check
entirely in that case) -- would have deduplicated `9783495485019`'s
"Anhang:" for free, no arbitration needed, since the two models already
agreed. More invasive: replacing `gate_book`'s unconditional
"matched pair keeps side `a`'s title" rule with something that flags a
pair for arbitration when `a` and `b`'s titles aren't near-identical
(only fuzzy-similar) rather than silently picking `a` -- would catch the
split-heading case, at the cost of routing more books to
`arbitrate_dnb_toc.py` instead of the fully-automatic bulk tier.

**Both fixes implemented (2026-08-19), plus three false-positive-driven
refinements found only by validating against real data:** `align_toc_entries`
now matches two `None`-page entries via a stricter title-only bar
(`_NEAR_EXACT_TITLE_THRESHOLD = 95.0`, deliberately NOT using the main
alignment score's OCR-noise-tolerant `partial_ratio`, since that would
score a truncated title ~100 against its own full version -- see
`_title_sort_score`'s docstring); `gate_book` now rejects the whole book
(routes to arbitration, same as a below-threshold book) when any matched
pair's titles aren't near-identical, rather than silently keeping side
`a`'s.

Before trusting either change, both were validated against every
already-cached real book pair on disk (85 books with exactly 2 cached
models) rather than just the synthetic unit tests -- a naive first
implementation (raw `token_sort_ratio` on unmodified titles) wrongly
flagged 51 of 85 (60%) previously-passing books for arbitration. Sampling
those showed the false positives weren't random: two systematic, high-
volume title-FORMATTING differences between the two models, unrelated to
correctness --

1. one model embedding the author name directly into the title text
   ("JOSEPH ROTH. Chapter Title", "Chapter Title—Jane Author", "Chapter
   Title (Jane Author)") while the other keeps title and author separate
   (title alone, author only in its own `authors` field) -- fixed by
   stripping each entry's OWN already-agreed-upon `authors` names out of
   its OWN title before comparing, rather than guessing at every possible
   name/separator shape by generic pattern;
2. one model including a leading chapter/section number in the title
   ("2 Decision-making", "Chapter 1: Un/thinking...", "1.4 Extraordinary
   revenues...") while the other reports the title alone -- by far the
   single biggest false-positive source measured, entire multi-chapter
   books (e.g. `9781107131101`, `9781405187268`, `9781433104503`) were
   failing almost purely on this; fixed by stripping a leading chapter-
   number pattern before comparing.

A third, smaller pass added German-style „low-9" opening quotation marks
to the existing curly-quote-to-ASCII normalization (only the English
curly style „“ → "" was originally covered) -- found on `3884796925`
(the Hölderlin Festschrift, 8 of 10 matched pairs affected: e.g.
"Mythologie der Vernunft" vs German „Mythologie der Vernunft“, otherwise
word-for-word identical) and `383050277X`.

After all three fixes: 37 of 85 books still newly fail -- but sampling
those directly shows most are NOT new false positives. Cross-checked
against the 7 books from the spot-check above that were directly,
visually confirmed correct against their real PDF scans (`9780190675684`,
`380061832X`, `3800181967`, `9783534245871`, `9783631725047`,
`9783823381891`, `9783890863603`): **zero overlap** with the newly-failing
set. Of the 37: most (`3823350242`, `3896697943`, `9783756030002`,
`9783779929369`, `9783825248512`, `9783531163789`, `9783948808211`, ...)
were ALREADY failing on agreement_rate alone, below 0.90, for reasons
entirely unrelated to the near-identical check (one model catastrophically
over- or under-segmenting the whole book) -- these were never going to
pass regardless of this change. A handful are genuine, previously-hidden
word-level misreadings the near-identical check newly surfaces now that
shared boilerplate (chapter numbers, author parentheticals) no longer
dilutes the comparison -- e.g. `9783556061497`: one model read a named
scale as "Leuvener Engagertheitsskala" (right name, misspelled second
word), the other as "Leumener Engagiertheitsskala" (misspelled name,
right second word) -- a genuine reading discrepancy on a real named term,
invisible before because the identical "2.2.1 ... (Dörte Weltzien)"
padding around it inflated the raw similarity score enough to clear the
old, unstripped threshold.

**One more widespread pattern found, deliberately left unfixed:** one
model capturing only a title's first clause/sentence, the other capturing
the FULL title+subtitle ("Zur Einführung" vs "Krise der Kritik? Zur
Einführung"; "Fortdauernder Sturm." vs "Fortdauernder Sturm. Einleitung
der Herausgeber") -- seen across many books (`9783835359208`: 11 of its
matched pairs; `9783837643671`: 7). Unlike the author-name/chapter-number
cases, the extra text here is NOT redundant metadata recoverable from
elsewhere on the entry -- it's the book's own genuine subtitle, present in
only one model's read. Silently normalizing this away (e.g. "always keep
the longer title") would reintroduce exactly the "incomplete training
target" risk `gate_book`'s whole design exists to avoid, just one level
down (title text instead of missing entries). Left as a real gate failure,
routing these books to arbitration -- more arbitration volume than either
fix alone would suggest, but arbitration is exactly where a human/Claude
judgment call between two genuinely different-content readings belongs.

Regression tests: `tests/test_dnb_toc_matching.py`'s `TestAlignTocEntries`
(4 new null-page cases), `TestGateBook` (2 new: the real split-heading
book and the real null-page-duplicate book), and
`TestTitleNearIdenticalNormalization` (9 cases covering every false-
positive pattern found above, plus two negative controls confirming a
genuine word-level misread and a genuine dropped subtitle still fail).

**MPCDF now available alongside KISSKI (2026-08-19):** both
`generate_dnb_toc_ground_truth.py` and `refresh_llm_cache.py` accept a
repeatable `--endpoint ALIAS` flag (see
`docs/superpowers/specs/2026-08-18-inference-endpoint-abstraction-design.md`)
that resolves `<ALIAS>_BASE_URL`/`_API_KEY`/`_MODEL` from the environment
and talks to any OpenAI-compatible endpoint, not just KISSKI's
discovery/selection path -- lets MPCDF's dedicated per-session vLLM
endpoints (`https://llm.mpcdf.mpg.de`, up to 8h, no shared-pool rate
limits) stand in for one or both sides of the two-vision-model gate.
Omitting `--endpoint` entirely is unchanged, still today's default
KISSKI auto-select behavior.

First real smoke test against two independent MPCDF sessions
(`Qwen/Qwen3-Omni-30B-A3B-Instruct` + `OpenGVLab/InternVL2_5-38B`,
`--endpoint MPCDF_A --endpoint MPCDF_B`, batches of 30 then 5): 6/35
books passed the gate automatically, the other 29 arbitrated by hand
against the real TOC page images. Arbitration also caught a real gate
bug independent of MPCDF itself -- `_TRAILING_PARENTHETICAL_RE` didn't
tolerate a trailing period after the closing paren, so one vision
model ending a title with "." right after `(...)` got an asymmetric
strip that failed an otherwise 100%-agreeing book
(`evaluation/dnb_toc_matching.py`, fixed with a regression test). Also
notable for future model-pairing choices: across the arbitrated books,
`Qwen3-Omni-30B` was right in nearly every individual disagreement,
while `InternVL2_5-38B` had frequent word-level misreadings and at
least two outright hallucinated strings -- this specific pairing is
lopsided enough that most of the 29 arbitrations were resolving
InternVL's own noise rather than genuine ambiguity, not yet swapped
for a stronger/more-independent second model.

**Arbitration backlog cleared, generation batch hit widespread connection
errors (2026-08-19):** `arbitrate_dnb_toc.py` surfaced 479 books with
cached model output but no decision -- 413 (86%) had only one model's
read cached (the second, a `qwen3.6-<N>` KISSKI call, never returned
anything), 66 (14%) had both models' reads but disagreed below the 0.90
gate. Single-model books aren't real arbitration candidates (nothing to
cross-check), so they were left for a `generate_dnb_toc_ground_truth.py`
re-run to fill in the missing second read via the existing per-model
cache (the already-cached side is free); the 66 genuine disagreements
were split across 8 parallel Claude subagents (~8-9 books each),
each following the documented arbitration workflow end to end --
**all 66 resolved, zero rejections**, root causes matching the
already-diagnosed patterns above (author-byline-split, front/back-matter
inclusion, two-line title splitting) plus one new one found repeatedly:
one model splitting each contributor's `AUTHOR NAME` line off as its own
spurious page-number-less TOC entry, confirmed against the real page
image every time.

The follow-up `--limit 100 --concurrency 4` generation batch (KISSKI
default two-model auto-discovery) made only 7 new bulk-tier passes and
regenerated 7 stale pre-2026-08-17-schema entries, but **46 of its 100
books hit `APIConnectionError`** (retries exhausted) -- a distinct
failure mode from the previously-diagnosed rate-limit/quota exhaustion
and empty-response flakiness, not yet root-caused (transient KISSKI-side
outage is the leading hypothesis; worth a straight retry before assuming
anything code-side broke). Net effect: the single-model-only backlog
barely moved (413 -> 412), so most of it is still open for a future
batch, ideally re-attempted when KISSKI connectivity is stable, or with
the second read routed to a live MPCDF session instead once one covers
an independent (non-`qwen3-omni`) model. The 3 books this batch *did*
newly push below the gate (both reads present, genuine disagreement)
were arbitrated directly, same standard as above -- also zero
rejections. `dnb-toc-only` ground truth stood at 246 books before this
session, 315 after.

**GLM-4.5V ruled out on startup cost, not capability; Pixtral-12B tried
as the InternVL replacement, prompt fix for verbatim numbering
(2026-08-19):** looking for a second MPCDF model that pairs better with
`Qwen3-Omni-30B-A3B-Instruct` than the noisy `InternVL2_5-38B` above,
`zai-org/GLM-4.5V` (`Glm4vMoeForConditionalGeneration`, ~108B total/~12B
active params) was spawned twice -- once on the default
`rocm7.0.0_vllm_0.11.2_*` image, once on a newer
`rocm7.13.0_gfx94X-dcgpu_..._vllm_0.19.1` image tried specifically to
clear other models' version gates (see `evaluation/hpc/llm-mpcdf.md`).
Both loaded the model and resolved its architecture correctly, but
neither reached a serving-ready `/v1/models` response within a 2h
session: weight loading alone took ~618-628s for the ~103GB checkpoint,
and CUDA/HIP graph capture across vLLM's 51 configured batch sizes for a
model this size appears to run well past that -- ruled out as a
practical second-model choice on startup cost alone, not quality (never
got far enough to evaluate output). Also found on the newer image: a
stale `.overlay_<jobid>_0.img` left behind by an earlier crashed/killed
session caused a `FATAL: EXT3 overlay image ... already exists` crash on
a later spawn attempt -- a new MPCDF gotcha, written up in
`evaluation/hpc/llm-mpcdf.md`. And confirmed the dashboard status label
is unreliable in *both* directions, not just "Running-but-not-ready" as
found earlier: a session that already logged the model fully loaded
still showed "building," so `/v1/models` is the only real signal either
way.

Switched to `mistralai/Pixtral-12B-2409` instead -- dense 12B, Mistral
lineage (independent from both Qwen and InternVL), loads in a fraction
of GLM-4.5V's time. First 10-book smoke test: 0/10 passed the gate (8
below-threshold, 2 `APIConnectionError`). Diffing the 8 comparable
books' raw cached output against Qwen's found a systematic pattern, not
noise: Pixtral stripped every printed leading section number from titles
(0/8 books had any numbered Pixtral title, including one book where Qwen
kept numbering on 36/43 entries) -- a real deviation from the project's
verbatim-per-line extraction standard, confirmed by a direct before/after
test against the two most-affected books, then fixed with an explicit
instruction in `_VISION_TOC_EXTRACTION_PROMPT`
(`evaluation/dnb_toc_vision.py`, commit `9bea40a`).

Re-running the same 8 books (stale Pixtral cache cleared first) confirms
the fix works but isn't sufficient on its own: agreement rate rose above
the 0.90 threshold on 2 of the 8 (`9783407254917`: 0.977; `9782875623980`:
0.926), but neither actually passed -- both still trip the 2026-08-19
any-mismatched-pair rule, now on genuine word-level OCR disagreements
between the two models ("Leseverstehen" vs "Leseverständnis") rather than
a numbering artifact. The other 6 have structural disagreements unrelated
to numbering: Pixtral merging part-headers into chapter titles and
dropping subtitle-continuation lines on one book (rate 0.408), Qwen
dropping author-byline lines Pixtral caught correctly on another (rate
0.571), plain entry-count mismatches on the rest. Net read: with the
numbering fix in place, Pixtral behaves like a genuinely independent
second model -- its own distinct OCR noise, not systematically weaker or
stronger than Qwen the way InternVL was -- but this small sample shows no
clean automatic-pass rate yet; still needs arbitration like any other
pairing.

**Pixtral ruled out for real: confident hallucination on dense TOCs
(2026-08-20).** A follow-up 20-book batch made the picture much worse: 1/20
passed, 16 below-threshold, 3 errored (1 `BadRequestError` -- a book whose
combined prompt+image token count exceeded the 65536 max_model_len; 2
`JSONDecodeError`). Of the 16 below-threshold books, most were re-resolved
by the concurrent KISSKI session's own bulk-gate pass before arbitration
was needed, leaving 4 genuine Qwen-vs-Pixtral disagreements
(`9783648200056`, `9783837840452`, `3825220494`, `9783110373912`) --
all four with a *high* raw agreement rate (0.96-0.98) dragged below
threshold by a large number (8-48) of near-identical-but-not-exact title
mismatches, a different shape of disagreement than the earlier 8-book
batch. Spot-checking `9783648200056` ("KI-Masterclass") against the real
page image confirmed Qwen correct on every single checked line and
Pixtral wrong on all of them -- not OCR noise but fluent, plausible-sounding
fabrication: printed "1.3 Rezession als Stresstest, nicht als Ursache"
came back from Pixtral as "1.3 Rekursion als Stressfaktor, nicht als
Ursache"; printed "Selbstberuhigung und Realitätstest" came back
"Selbstabrufung und Realitätstest"; printed "Der operative »Grind« als
eigentlicher Engpass" came back "Der operative Grid: als eigentlicher
Engpass". All three affected books share dense, small-font, multi-level
numbered TOCs (technical/business or academic-handbook style) -- Pixtral's
failure mode is specific to that document class, not a general weakness
(it did fine on the flatter TOCs in the first smoke test). Verdict:
Pixtral is not a usable second model for this corpus as a whole --
dropped.

**GLM-4.1V-9B-Thinking tried next: architecturally incompatible with the
pipeline's JSON extraction, not just weaker (2026-08-20).** Spawned as
a fast, genuinely-independent (Zhipu/GLM lineage), small (9B dense, 1
GPU) candidate specifically to fix the two problems above -- on the newer
`rocm7.13.0_gfx94X-dcgpu_..._vllm_0.19.1` image, since
`Glm4vForConditionalGeneration` needs vLLM >=0.12.0 (see
`evaluation/hpc/llm-mpcdf.md`). A 20-book batch scored 0/20: 3
below-threshold, 1 `BadRequestError` (same context-length-overflow
failure as Pixtral's), and **15 `JSONDecodeError`**. Root cause: this
model always answers in a fixed `<think>...reasoning...</think><answer>...`
format, and `parse_json_array` (`src/chapter_segmentation/_llm_json.py`)
naively takes the first `[` and last `]` in the *whole* response --
every model tried before this one answered with no preamble, so the
assumption had never been tested. The reasoning block's own brackets get
caught by that naive span, producing malformed JSON on most books; one
book's raw response showed the reasoning trace visibly running out of its
token budget and degenerating into a multi-thousand-character run of
repeated periods before ever reaching an answer.

Confirmed there is no request-level escape hatch: passing
`extra_body={"chat_template_kwargs": {"enable_thinking": False}}` (the
convention some Qwen3-style hybrid-reasoning chat templates honor) had
zero effect on a direct test call -- the `<think>` block is structurally
fixed into this model's template, not a runtime toggle. A direct
before/after test (stripping the `<think>...</think>` span before
`parse_json_array`, `max_tokens` raised to 16000) got a real book to parse
cleanly (`9783779932819`, 21 entries) but was **not reliably
reproducible even at temperature=0.0** -- an identical repeat call
against a second book (`3518006940`) failed to parse on the first try and
only succeeded on a second identical call. Reasoning alone consumed
6,600-11,600 completion tokens on these two *small* (8-21 entry) books,
against this session's 65536-token context ceiling shared with the
prompt+images -- the same ceiling that already produced a
`BadRequestError` on a dense book with no reasoning overhead at all,
meaning the corpus's denser TOCs (150+ entries, the exact class Pixtral
just failed on) are likely to blow the budget outright even with a code
fix. Not pursued further as a production pairing: the fix is real but
partial, and the remaining risk concentrates on precisely the books this
whole search was trying to make more reliable.

**Why a second model keeps failing for a different reason each time
(2026-08-20):** across every candidate tried this session
(`InternVL2_5-38B`, `Pixtral-12B-2409`, `GLM-4.5V`, `GLM-4.1V-9B-Thinking`,
plus `Qwen/Qwen3.6-35B-A3B` and `deepseek-ai/deepseek-vl2` which never
even got a quality test), the failures cluster into three independent
constraints, not one recurring flaw: (1) **architectural independence
from Qwen is rarer than expected** -- a large share of the open VLM
ecosystem, InternVL included, is built on a Qwen2/2.5/3 LLM backbone,
since it's currently the strongest freely-available option to pair a
vision encoder with, so "not Qwen-derived" and "strong at dense document
OCR" are anti-correlated in practice; (2) **the labs that are genuinely
independent mostly aren't optimized for this task** -- Pixtral and
Llama-Vision are general vision-chat models, not document specialists,
and it showed as confident fabrication rather than garbled OCR; GLM's
answer to competing on quality was a reasoning model, which solved
neither problem and introduced a new output-format incompatibility of its
own; (3) **pure MPCDF-launcher infrastructure friction, uncorrelated with
model quality** -- the default image's old vLLM rejects some
architectures outright (`Qwen3.6-35B-A3B`, `GLM-4.1V` without the newer
image), the CLI-arguments field's blanket quote-stripping makes
`deepseek-vl2`'s required `--hf-overrides` unusable regardless of image,
and a large/MoE checkpoint's graph-capture time (`GLM-4.5V`) can consume
most of a session before quality is ever testable. Each candidate so far
has failed a *different* one of these three, which is why the search has
felt like it keeps finding new problems rather than converging --
narrowing on any future candidate should check all three before spending
a batch on it, not just the one that sank the previous attempt.
