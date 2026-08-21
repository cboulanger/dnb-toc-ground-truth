# Arbitrating below-gate books

`cli/generate_ground_truth.py`'s agreement gate discards a book outright
when no pair of its resolved endpoints' reads agrees well enough (below
0.90 agreement) or fewer than two endpoints produce usable output -- but
it never deletes any endpoint's cached raw extraction
(`data/corpus/pilot/llm-cache/<key>.<model>.json`). Rather than
re-running the whole book from scratch or leaving it discarded, walk
through the following after a generation run leaves books below the
gate (design spec `docs/superpowers/specs/2026-08-16-dnb-toc-arbitration-design.md`):

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
   (`{"entries": [...], "verified": true, "source": "claude_arbitration"}`,
   each entry via `dnb_toc_ground_truth.matching.toc_entry_to_gt_dict`),
   but with `"verified": true` rather than `false`: unlike the bulk-tier
   gate's own output, this went through direct scrutiny (including the
   images, when needed) -- excluded from `_spot_check`'s sampling pool
   going forward. The `"source": "claude_arbitration"` field (vs. the
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
