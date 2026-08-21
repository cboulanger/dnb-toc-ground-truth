"""Crossref book-DOI and chapter-list lookup by ISBN -- ported from
chapter-segmentation's
src/chapter_segmentation/evidence/crossref_strategy.py (async
httpx.AsyncClient, ChapterCandidate output) to this repo's sync
httpx.Client convention (matching cli/fetch_corpus.py's own style) and
TocEntry output, so it composes directly with matching.py's diff logic.
See design spec
docs/superpowers/specs/2026-08-21-crossref-cross-validation-design.md.

One GET .../works?filter=isbn:{isbn} serves both this design's needs at
once -- Crossref returns every work type registered under an ISBN in one
response, not just book-chapter, so the book-level DOI (first item typed
as an actual book, not one of its chapters) and the chapter list (every
book-chapter-typed item) both come from a single call."""

import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx

from dnb_toc_ground_truth.toc_entry import TocEntry

_CROSSREF_BASE_URL = "https://api.crossref.org/works"
_MAX_RETRIES = 3
_DEFAULT_RETRY_DELAY_SECONDS = 1.0

# Preferred book-level Crossref work types, checked in the order Crossref
# returns items (not reordered by preference among these) -- a narrower
# whitelist than "anything not book-chapter", since a stray "book-part" or
# "book-set" item registered under the same ISBN is not itself the book's
# own DOI.
_BOOK_TYPES = ("book", "monograph", "edited-book", "reference-book")


def normalize_isbn(raw: str) -> Optional[str]:
    """Extracts the first ISBN-13 (13 digits after stripping separators)
    from a manifest ISBN field, falling back to the first ISBN-10 (10
    digits, last may be 'X'), or None if nothing usable is present."""
    if not raw:
        return None
    candidates = re.split(r"[;\s]+", raw.strip())
    stripped = [re.sub(r"[-\s]", "", c) for c in candidates if c.strip()]
    for c in stripped:
        if len(c) == 13 and c.isdigit():
            return c
    for c in stripped:
        if len(c) == 10 and c[:-1].isdigit() and (c[-1].isdigit() or c[-1].upper() == "X"):
            return c[:-1] + c[-1].upper()
    return None


@dataclass(frozen=True)
class CrossrefBookData:
    isbn: str
    doi: Optional[str]
    chapters: tuple[TocEntry, ...]
    fetched_at: str
