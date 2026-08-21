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


def _cache_path(cache_dir: Path, isbn: str) -> Path:
    return cache_dir / f"{isbn}.crossref.json"


def _load_cache(cache_dir: Path, isbn: str) -> Optional[CrossrefBookData]:
    path = _cache_path(cache_dir, isbn)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        chapters = tuple(
            TocEntry(
                title=c["title"], authors=tuple(c["authors"]),
                printed_page_number=c["printed_page_number"], source_page_index=-1, skip=False,
            )
            for c in data["chapters"]
        )
        return CrossrefBookData(isbn=isbn, doi=data.get("doi"), chapters=chapters, fetched_at=data["fetched_at"])
    except Exception as exc:
        print(f"  [warn] corrupted crossref cache for {isbn}, refetching: {exc}")
        return None


def _save_cache(cache_dir: Path, data: CrossrefBookData) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "isbn": data.isbn,
        "doi": data.doi,
        "fetched_at": data.fetched_at,
        "chapters": [
            {"title": c.title, "authors": list(c.authors), "printed_page_number": c.printed_page_number}
            for c in data.chapters
        ],
    }
    _cache_path(cache_dir, data.isbn).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
    )


def _first_page(page_range: Optional[str]) -> Optional[str]:
    if not page_range:
        return None
    first = page_range.split("-")[0].strip()
    return first or None


def _parse_chapter_item(item: dict) -> Optional[TocEntry]:
    if item.get("type") != "book-chapter":
        return None
    titles = item.get("title") or []
    if not titles:
        return None
    # Crossref splits a chapter's real printed heading into separate
    # title/subtitle fields -- see crossref_strategy.py's
    # _parse_crossref_item for the full rationale this ports.
    subtitles = item.get("subtitle") or []
    title = f"{titles[0]} {subtitles[0]}" if subtitles else titles[0]
    authors = tuple(
        f"{a.get('given', '')} {a.get('family', '')}".strip()
        for a in item.get("author", []) if a.get("family")
    )
    return TocEntry(
        title=title, authors=authors, printed_page_number=_first_page(item.get("page")),
        source_page_index=-1, skip=False,
    )


def _book_doi(items: list[dict]) -> Optional[str]:
    for item in items:
        if item.get("type") in _BOOK_TYPES:
            return item.get("DOI")
    return None


def _query_crossref(isbn: str, client: httpx.Client, contact_email: Optional[str]) -> Optional[list[dict]]:
    """Returns the raw items list on a confirmed successful response
    (including a confirmed-empty one), or None on any network/HTTP-
    status/JSON-shape failure -- the None/empty-list distinction is what
    lets fetch_crossref_book below decide whether the result is safe to
    cache."""
    params: dict[str, str | int] = {
        "filter": f"isbn:{isbn}",
        "select": "DOI,title,subtitle,author,page,type",
        "rows": 100,
    }
    if contact_email:
        params["mailto"] = contact_email

    response = None
    for _attempt in range(_MAX_RETRIES):
        try:
            response = client.get(_CROSSREF_BASE_URL, params=params, timeout=10.0)
        except httpx.HTTPError as exc:
            print(f"  [warn] network error fetching Crossref data for {isbn}: {exc}")
            return None
        if response.status_code != 429:
            break
        retry_after = response.headers.get("Retry-After")
        delay = float(retry_after) if retry_after and retry_after.isdigit() else _DEFAULT_RETRY_DELAY_SECONDS
        time.sleep(delay)
    else:
        print(f"  [warn] exhausted retries (429) fetching Crossref data for {isbn}")
        return None

    try:
        response.raise_for_status()
        return response.json()["message"]["items"]
    except Exception as exc:
        print(f"  [warn] bad Crossref response for {isbn}: {exc}")
        return None


def fetch_crossref_book(
    isbn: str,
    client: httpx.Client,
    contact_email: Optional[str],
    cache_dir: Path,
    force: bool = False,
) -> CrossrefBookData:
    """The book-level DOI plus its Crossref-registered chapter list,
    fetched from one GET .../works?filter=isbn:{isbn} and cached on disk
    at <cache_dir>/<isbn>.crossref.json. Cached even when doi is None and
    chapters is empty (a confirmed miss, never re-queried on repeat
    runs) -- but NOT cached when the request itself failed (network
    error, exhausted 429 retries, malformed response), so a transient
    failure gets retried on the next run instead of being mistaken for a
    confirmed miss forever."""
    if not force:
        cached = _load_cache(cache_dir, isbn)
        if cached is not None:
            return cached

    items = _query_crossref(isbn, client, contact_email)
    if items is None:
        return CrossrefBookData(isbn=isbn, doi=None, chapters=(), fetched_at="")

    doi = _book_doi(items)
    chapters = tuple(c for item in items if (c := _parse_chapter_item(item)) is not None)
    data = CrossrefBookData(
        isbn=isbn, doi=doi, chapters=chapters, fetched_at=datetime.now(timezone.utc).isoformat(),
    )
    _save_cache(cache_dir, data)
    return data
