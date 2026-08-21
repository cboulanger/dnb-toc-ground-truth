"""Whole-book agreement gate for dnb-toc-only ground truth -- see design
spec docs/superpowers/specs/2026-08-15-dnb-toc-ground-truth-generation-design.md
section 4, and docs/superpowers/specs/2026-08-16-dnb-toc-uniform-ocr-design.md
for the follow-up redesign. Pure functions over TocEntry lists produced by
two independent vision_extract_toc_entries calls
(vision.py), which already return the identical
list[TocEntry] shape."""

import re
from dataclasses import replace
from typing import Optional

from rapidfuzz import fuzz

from dnb_toc_ground_truth.toc_entry import TocEntry, _parse_toc_page_number

# Typographic Unicode punctuation variants that mean the same thing as
# their ASCII counterpart for comparison purposes -- one model reading a
# page's curly quotes as "smart" Unicode and the other normalizing to
# ASCII (or vice versa) is a rendering/tokenization difference, not a
# content disagreement. rapidfuzz's char-level scoring doesn't know that,
# so _title_sort_score normalizes both sides before comparing.
_PUNCTUATION_NORMALIZATION = str.maketrans({
    "“": '"', "”": '"', "„": '"',  # German-style low-9 opening quote too
    "‘": "'", "’": "'", "‚": "'",
    "–": "-", "—": "-", "…": "...",
})

# A trailing "(...)" group with no recognizable author name inside it --
# fallback for the rare case _normalize_title_for_near_identical_check's
# own-authors stripping below doesn't already cover (e.g. a translator/
# illustrator credit vision extraction didn't put in `authors`). Applied
# only before the strict near-identical comparison (see _title_sort_score)
# -- the main alignment score (_title_score) doesn't need this, since
# partial_ratio already scores the shared prefix highly enough to align
# regardless. The optional trailing "." (found 2026-08-18, book
# 9783161636820) matters: one vision model ending its title with a period
# right after the closing paren otherwise defeats the "$" anchor, so only
# the OTHER, period-less side gets its parenthetical stripped -- an
# asymmetric strip that scores an otherwise-identical pair below threshold.
_TRAILING_PARENTHETICAL_RE = re.compile(r"\s*\([^()]*\)\.?\s*$")

# A leading chapter/section number one model includes in the title text
# ("2 Decision-making", "1.4 Extraordinary revenues", "Chapter 1:
# Un/thinking...") while the other reports the title alone -- see
# _normalize_title_for_near_identical_check's own docstring for how
# common this turned out to be. Symmetric on both sides of a comparison,
# so an identical real leading number on both sides still compares equal
# after stripping; the narrow risk is a genuine chapter MIS-numbering
# error (entry_a says "3", entry_b says "4", same title text otherwise)
# being masked rather than flagged -- accepted, since it's a much rarer
# failure mode than the formatting difference this fixes.
_LEADING_CHAPTER_NUMBER_RE = re.compile(r"^\s*(?:chapter\s+)?\d+(?:\.\d+)*\s*[.:]?\s+", re.IGNORECASE)


def _normalize_title_for_near_identical_check(entry: TocEntry) -> str:
    """Normalizes one entry's title for the strict near-identical
    comparison (_title_sort_score/_title_near_identical) -- collapses
    away two title-FORMATTING differences that measured against all 85
    already-cached real dnb-toc-only book pairs turned out to be extremely
    common (2026-08-19; see _title_sort_score's own docstring for the
    measurement), neither of which is an actual content disagreement:

    - typographic Unicode punctuation (curly quotes -- including the
      German „low-9" opening style -- em/en dashes, ellipsis) normalized
      to its ASCII equivalent;
    - a leading chapter/section number one model includes in the title
      text while the other doesn't ("2 Decision-making" vs
      "Decision-making", "Chapter 1: Un/thinking..." vs "Un/thinking...")
      -- by far the single most common false-positive source measured
      (see _LEADING_CHAPTER_NUMBER_RE's own docstring for the risk this
      accepts in exchange);
    - the entry's OWN `authors` names stripped out of its OWN title text,
      wherever one model folds an author into the title itself
      ("JOSEPH ROTH. Chapter Title", "Chapter Title—Jane Author",
      "Chapter Title (Jane Author)") while the other keeps title and
      author cleanly separated (title alone, author only in `authors`).
      Both models were found to already agree on the actual author name
      in their own `authors` field in every sampled case -- using each
      entry's OWN reported author to strip is a targeted, grounded
      alternative to guessing at every possible name/separator shape by
      generic punctuation pattern, and self-consistent since it can never
      strip something the entry's own extraction didn't already claim was
      the author.

    Does NOT consult TocEntry.title_variants the way _title_score's
    _candidate_titles does -- title_variants is only ever populated by
    the regex/heuristic extraction path (never vision extraction, this
    module's only real input, per _candidate_titles' own docstring), so
    there is nothing for it to add here."""
    title = entry.title.translate(_PUNCTUATION_NORMALIZATION)
    title = _LEADING_CHAPTER_NUMBER_RE.sub("", title)
    for author in entry.authors:
        escaped = re.escape(author)
        title = re.sub(rf"^\s*{escaped}\s*[.:-]\s*", "", title, flags=re.IGNORECASE)
        title = re.sub(rf"\s*[.:-]\s*{escaped}\s*$", "", title, flags=re.IGNORECASE)
        title = re.sub(rf"\s*\(\s*{escaped}\s*\)\s*$", "", title, flags=re.IGNORECASE)
    title = _TRAILING_PARENTHETICAL_RE.sub("", title)
    return title.strip()

_ALIGN_SCORE_THRESHOLD = 70.0

# Stricter bar used wherever there's no printed_page_number to independently
# confirm two titles refer to the same physical line -- see
# _title_near_identical's own docstring for why this can't reuse
# _ALIGN_SCORE_THRESHOLD's scoring.
_NEAR_EXACT_TITLE_THRESHOLD = 95.0


def _pages_equivalent(a: str | None, b: str | None) -> bool:
    """True when two entries' printed_page_number values represent the
    same page. None never matches None (or anything else) -- "unknown"
    on either side means there is nothing to compare, same policy the
    old -1-sentinel-skip already enforced. Otherwise: exact string match
    first (handles a shared alternate-scheme marker like "R42" directly,
    with no numeric parsing needed at all); then numeric equality via
    _parse_toc_page_number (handles a case difference in a roman numeral,
    "VII" vs "vii", or a leading zero, "07" vs "7"); then a
    case-insensitive string match (handles a case difference in a
    non-roman marker, "R42" vs "r42").

    Known limitation: the numeric tier does not consult printed_roman,
    so a roman marker can numerically collide with an unrelated arabic
    marker of the same value ("L" vs "50", both 50) -- accepted rather
    than threading printed_roman through this function and every
    align_toc_entries call site, since align_toc_entries' own
    title-similarity gate already makes a real false-positive rare (it
    additionally requires the two entries' titles to score >=
    _ALIGN_SCORE_THRESHOLD)."""
    if a is None or b is None:
        return False
    if a == b:
        return True
    parsed_a, parsed_b = _parse_toc_page_number(a), _parse_toc_page_number(b)
    if parsed_a is not None and parsed_a == parsed_b:
        return True
    return a.casefold() == b.casefold()


def _candidate_titles(entry: TocEntry) -> tuple[str, ...]:
    """Every title reading worth trying for a fuzzy match: the primary
    title, plus any longer wrapped-title variants (TocEntry.title_variants
    -- see its own docstring in segmentation.py). An entry whose title
    wrapped across multiple TOC-page lines can have its FULL title only in
    title_variants, not title itself (a line-based capture may record just
    the last line, where the page number sits) -- comparing only .title
    against the other side's full, whole-read title would systematically
    under-score a real match. Found empirically: a real book's wrapped
    page-33 title matched its own title_variants near-verbatim but scored
    well below threshold against .title alone (2026-08-15 smoke test, book
    9783899718188). Note that of this module's current callers, only
    find_toc_candidates' regex path ever populates title_variants --
    _toc_items_to_entries (shared by both the text-LLM and vision-LLM
    paths) never does, so for two vision-extraction inputs this tuple is
    just (entry.title,) on both sides; the mechanism stays in place for
    whichever future extractor populates it."""
    return (entry.title,) + entry.title_variants


def _title_score(entry_a: TocEntry, entry_b: TocEntry) -> float:
    """Best fuzzy title-similarity score between two entries, trying every
    candidate title reading on each side (see _candidate_titles) -- the
    better of rapidfuzz's token_sort_ratio and partial_ratio (the latter
    added 2026-08-16 to tolerate a trailing noise run on one side, e.g.
    garbled OCR tokens following an otherwise-exact-match title, without
    inflating false positives; see design spec
    docs/superpowers/specs/2026-08-16-dnb-toc-uniform-ocr-design.md
    section 1d/3.2 for the measurements behind this)."""
    return max(
        max(
            fuzz.token_sort_ratio(title_a.lower(), title_b.lower()),
            fuzz.partial_ratio(title_a.lower(), title_b.lower()),
        )
        for title_a in _candidate_titles(entry_a)
        for title_b in _candidate_titles(entry_b)
    )


def _title_sort_score(entry_a: TocEntry, entry_b: TocEntry) -> float:
    """token_sort_ratio between two entries' titles, each first run through
    _normalize_title_for_near_identical_check -- deliberately does NOT
    blend in partial_ratio the way _title_score does: partial_ratio scores
    a strict prefix/substring match as ~100 regardless of how much extra
    content the longer side has, which is exactly what makes it good at
    tolerating trailing OCR noise (see _title_score's own docstring) but
    useless at telling a genuinely truncated/split title apart from a
    merely noisy one -- e.g. "Einleitung:" scores 100 by partial_ratio
    against "Einleitung: Endlichkeit und Verantwortung" (a real
    split-heading defect found via docs/history.md's
    bulk-gate spot-check, 2026-08-19) despite being a completely different,
    incomplete title. token_sort_ratio alone correctly scores that pair
    ~42.

    The normalization step matters just as much as the metric choice here.
    Measured directly against all 85 already-cached real dnb-toc-only book
    pairs (2026-08-19): scoring RAW titles wrongly failed 51 of them (60%)
    that were previously passing the gate -- root-caused to two title-
    FORMATTING differences with no bearing on correctness (curly vs.
    straight quotes; one model folding an author name into the title text
    while the other keeps title/author separate, in several different
    shapes -- leading "NAME. Title"/"Name: Title", trailing
    "Title—Name"/"Title (Name)"). Normalizing punctuation and stripping
    each entry's own already-agreed-upon `authors` names out of its own
    title (see _normalize_title_for_near_identical_check) cut that down to
    still real, differently-shaped defects (this function's own module,
    dnb_toc_matching.py, has no further false positives once real-world-
    verified on that same 85-book set) while still scoring the actual
    split-heading defect above at ~42 (well below threshold)."""
    return fuzz.token_sort_ratio(
        _normalize_title_for_near_identical_check(entry_a).lower(),
        _normalize_title_for_near_identical_check(entry_b).lower(),
    )


def _title_near_identical(entry_a: TocEntry, entry_b: TocEntry) -> bool:
    """True when two entries' titles are close enough to trust as the same
    line WITHOUT independent page-number confirmation -- see
    _title_sort_score's own docstring for why this uses token_sort_ratio
    alone rather than _title_score's more permissive blend."""
    return _title_sort_score(entry_a, entry_b) >= _NEAR_EXACT_TITLE_THRESHOLD


def align_toc_entries(a: list[TocEntry], b: list[TocEntry]) -> list[tuple[int, int]]:
    """Greedy, order-preserving alignment between two independently-
    produced TocEntry lists for the same TOC scan. A pair (i, j) counts as
    a match when EITHER:

    - both sides have a KNOWN printed_page_number (neither is None) that's
      equivalent per _pages_equivalent (exact string match, numeric match,
      or case-insensitive string match), AND their titles score
      >= _ALIGN_SCORE_THRESHOLD per _title_score; or
    - both sides have an UNKNOWN printed_page_number (both None) and their
      titles are near-identical per _title_near_identical -- added
      2026-08-19 after a real case (docs/history.md's
      bulk-gate spot-check) where two models independently read the exact same
      page-number-less divider ("Anhang:") but the previous blanket "either
      side None -> never match" rule meant two IDENTICAL readings could
      never be recognized as agreeing, silently duplicating the entry in
      gate_book's merged output. The near-identical bar (rather than
      _ALIGN_SCORE_THRESHOLD) is deliberate: with no page number to confirm
      the pair refers to the same physical line, a looser bar risks merging
      two genuinely different unrelated null-page entries (e.g. two
      different section dividers) into one. A None-page entry_a never
      matches a known-page entry_b or vice versa.

    Alignment is page-number-first, then title, using a greedy scan from
    the last matched b-index ("TOC order is book order"), but returns
    index PAIRS rather than a bare count, since the whole-book gate below
    needs to know exactly which entries agreed."""
    pairs: list[tuple[int, int]] = []
    last_j = -1
    for i, entry_a in enumerate(a):
        page_known = entry_a.printed_page_number is not None
        best_j = None
        best_score = _ALIGN_SCORE_THRESHOLD if page_known else _NEAR_EXACT_TITLE_THRESHOLD
        for j in range(last_j + 1, len(b)):
            entry_b = b[j]
            if page_known:
                if not _pages_equivalent(entry_a.printed_page_number, entry_b.printed_page_number):
                    continue
                score = _title_score(entry_a, entry_b)
            else:
                if entry_b.printed_page_number is not None:
                    continue
                score = _title_sort_score(entry_a, entry_b)
            if score >= best_score:
                best_score = score
                best_j = j
        if best_j is not None:
            pairs.append((i, best_j))
            last_j = best_j
    return pairs


def diff_toc_entries(
    a: list[TocEntry], b: list[TocEntry],
) -> tuple[list[tuple[TocEntry, TocEntry]], list[TocEntry], list[TocEntry]]:
    """Aligns a and b via align_toc_entries and returns (matched_pairs,
    only_in_a, only_in_b) -- matched_pairs holds the actual TocEntry
    objects (not indices) from each side for each matched line,
    only_in_a/only_in_b hold every entry from that side with no match on
    the other. Same underlying alignment gate_book uses to decide
    pass/fail; this exposes the full breakdown for a human (or Claude,
    arbitrating a below-threshold book) to review the actual
    disagreement -- see
    docs/superpowers/specs/2026-08-16-dnb-toc-arbitration-design.md
    section 4.1."""
    pairs = align_toc_entries(a, b)
    matched_a = {i for i, _ in pairs}
    matched_b = {j for _, j in pairs}
    matched_pairs = [(a[i], b[j]) for i, j in pairs]
    only_in_a = [entry for i, entry in enumerate(a) if i not in matched_a]
    only_in_b = [entry for j, entry in enumerate(b) if j not in matched_b]
    return matched_pairs, only_in_a, only_in_b


def gate_book(
    a: list[TocEntry], b: list[TocEntry], threshold: float = 0.90,
) -> tuple[bool, list[TocEntry]]:
    """Whole-book agreement gate (design spec section 4.2 of the original
    2026-08-15 design; the two inputs are now two independent vision-model
    extractions rather than a regex heuristic and a text-LLM pass -- see
    docs/superpowers/specs/2026-08-16-dnb-toc-uniform-ocr-design.md section
    3.1). agreement_rate = matched-pair count / max(len(a), len(b)).
    Below `threshold`, the book is rejected outright (passed=False,
    entries=[]) rather than trimmed down to just the agreeing entries -- a
    partially-agreeing book is exactly the case this design distrusts
    most, and a caller must not silently write a partial/incomplete
    result for it.

    Also rejected outright, even at/above `threshold`: any matched pair
    whose titles aren't near-identical per _title_near_identical -- added
    2026-08-19 after a real case (docs/history.md's
    bulk-gate spot-check) where a fuzzy-but-not-exact matched pair (one model split
    a heading the other read whole) silently kept the WORSE of the two
    readings, since the merge below has no way to tell which side is
    actually correct. Both sides having independently produced a real
    reading of the same line is exactly the case worth a closer look, not
    a silent pick -- the design's own preference here is to fail the whole
    book (routing it to arbitrate_dnb_toc.py, same as a below-threshold
    book) rather than resolve the ambiguity by fiat.

    At or above `threshold` (and with every matched pair near-identical),
    `entries` is the UNION of matched pairs (`a`'s title kept -- an
    arbitrary but deterministic choice between two equally-produced,
    now-confirmed-equivalent extractions -- falling back to `b`'s authors
    when `a`'s own are empty, in case one model dropped them) plus every
    singleton entry either side found alone, ordered by
    printed_page_number (an unknown, i.e. None, value sorts last). This is
    deliberate: once a book clears the trust bar, a line only one side
    caught is far likelier a real entry the other missed than a
    hallucination -- trimming it out would silently understate the page's
    real content, which is exactly the "incomplete training target"
    failure mode this design exists to avoid."""
    if not a and not b:
        return False, []
    matched_pairs, only_in_a, only_in_b = diff_toc_entries(a, b)
    agreement_rate = len(matched_pairs) / max(len(a), len(b))
    if agreement_rate < threshold:
        return False, []
    if any(not _title_near_identical(entry_a, entry_b) for entry_a, entry_b in matched_pairs):
        return False, []
    merged = [
        replace(entry_a, authors=entry_a.authors or entry_b.authors)
        for entry_a, entry_b in matched_pairs
    ]
    merged += only_in_a
    merged += only_in_b

    def _page_sort_key(entry: TocEntry) -> tuple:
        value = _parse_toc_page_number(entry.printed_page_number) if entry.printed_page_number else None
        return (entry.printed_page_number is None, value if value is not None else 0, entry.printed_page_number or "")

    merged.sort(key=_page_sort_key)
    return True, merged


def toc_entry_to_gt_dict(entry: TocEntry) -> dict:
    """Serializes one TocEntry to this corpus's <id>.expected.json entry
    shape (design spec section 2) -- printed_page_number is already
    str | None on TocEntry (see docs/superpowers/specs/2026-08-17-printed-page-number-string-design.md),
    so it's passed through as-is, and mirrors how citation_pages is
    already stored as a string elsewhere in this project's ground truth.

    "skip" (added 2026-08-17, see TocEntry.skip's own docstring) records
    the extraction's own hint about whether this entry is a real chapter
    (from either vision_extract_toc_entries or, since 2026-08-20,
    text_extract_toc_entries -- both produce the same TocEntry shape), but
    is NOT authoritative -- a downstream consumer that wants "which of
    these are real chapters" should be free to reclassify from the
    verbatim title/page data without needing new model calls."""
    return {
        "title": entry.title,
        "authors": list(entry.authors),
        "printed_page_number": entry.printed_page_number,
        "skip": entry.skip,
    }


def gate_books(
    entry_lists: list[list[TocEntry]], threshold: float = 0.90,
) -> tuple[bool, list[TocEntry], Optional[tuple[int, int]]]:
    """Extends gate_book to N >= 2 independently-produced TocEntry lists
    -- "best pair wins" (design spec
    docs/superpowers/specs/2026-08-21-dnb-toc-ground-truth-extraction-design.md
    "N-way gating"): every pair among entry_lists is checked with the SAME
    pairwise alignment + agreement-rate + near-identical-title gate
    gate_book already implements, unchanged. The book passes if at least
    one pair clears `threshold`; among passing pairs, the one with the
    HIGHEST agreement rate is used, and its merged output (gate_book's own
    merge -- union of matched pairs plus each side's singletons) becomes
    the result. Returns (passed, merged_entries, winning_pair_indices) --
    winning_pair_indices names which two positions in entry_lists produced
    the result (for logging which two models actually agreed), None if
    every pair failed.

    Deliberately does NOT attempt a multi-way consensus merge across more
    than one passing pair -- see the design spec section this implements
    for why: reusing gate_book's proven, heavily-tested two-list merge
    logic verbatim was chosen over a new N-way merge algorithm."""
    if len(entry_lists) < 2:
        raise ValueError(f"gate_books needs at least 2 entry lists to gate, got {len(entry_lists)}")
    best_rate = -1.0
    best_pair: Optional[tuple[int, int]] = None
    for i in range(len(entry_lists)):
        for j in range(i + 1, len(entry_lists)):
            a, b = entry_lists[i], entry_lists[j]
            if not a and not b:
                continue
            matched_pairs, _, _ = diff_toc_entries(a, b)
            rate = len(matched_pairs) / max(len(a), len(b))
            if rate < threshold or rate <= best_rate:
                continue
            if any(not _title_near_identical(entry_a, entry_b) for entry_a, entry_b in matched_pairs):
                continue
            best_rate = rate
            best_pair = (i, j)
    if best_pair is None:
        return False, [], None
    i, j = best_pair
    passed, merged = gate_book(entry_lists[i], entry_lists[j], threshold=threshold)
    assert passed  # guaranteed by the loop above having already checked this exact pair
    return True, merged, best_pair
