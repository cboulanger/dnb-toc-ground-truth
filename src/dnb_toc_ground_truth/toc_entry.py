"""TOC-entry data type and parsing helpers -- vendored from
chapter-segmentation's src/chapter_segmentation/segmentation.py
(TocEntry, _parse_toc_page_number, _normalize_printed_page_number,
_toc_items_to_entries) and src/chapter_segmentation/_llm_json.py
(parse_json_array), copied rather than imported so this repo has zero
dependency on the chapter_segmentation package -- see design spec
docs/superpowers/specs/2026-08-21-dnb-toc-ground-truth-extraction-design.md.
Kept byte-for-byte identical to its origin except for this module's own
docstring and any DNB-toc-only-specific mentions that no longer need a
disclaimer, since this whole repo IS dnb-toc-only-specific now."""

import json
import math
import re
from dataclasses import dataclass

_STRICT_ROMAN_RE = re.compile(r"^m{0,3}(cm|cd|d?c{0,3})(xc|xl|l?x{0,3})(ix|iv|v?i{0,3})$")
_ROMAN_VALUES = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}
_ROMAN_PAGE_MAX_VALUE = 50


def _parse_toc_page_number(raw: str) -> int | None:
    """The integer value of a TOC line's captured page field, or None if
    the field is not a plausible page number (an invalid or implausibly
    large roman numeral)."""
    if raw.isdigit():
        return int(raw)
    lowered = raw.lower()
    if not _STRICT_ROMAN_RE.match(lowered) or not lowered:
        return None
    total = 0
    for ch, nxt in zip(lowered, lowered[1:] + " "):
        value = _ROMAN_VALUES[ch]
        total += -value if nxt != " " and _ROMAN_VALUES.get(nxt, 0) > value else value
    return total if total <= _ROMAN_PAGE_MAX_VALUE else None


def _normalize_printed_page_number(value: object) -> str | None:
    """Coerces any of TocEntry.printed_page_number's legal input shapes
    into the canonical str | None form."""
    if isinstance(value, str):
        text = value.strip()
        return None if not text or text.lower() == "null" else text
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        value = int(value)
        return None if value == -1 else str(value)
    return None


@dataclass(frozen=True)
class TocEntry:
    title: str
    printed_page_number: str | None
    source_page_index: int  # which page (0-based) the TOC entry itself was found on
    authors: tuple[str, ...] = ()
    printed_roman: bool = False
    title_variants: tuple[str, ...] = ()
    skip: bool = False  # True when this entry is not itself a real chapter --
    # a part/section divider, or front/back matter (preface, bibliography,
    # index, ...). Set by vision.py/ocr.py's extraction: both deliberately
    # extract EVERY printed TOC line verbatim rather than omitting
    # non-chapter lines outright, so a two-model disagreement over what to
    # extract can never cause an editorial-judgment mismatch to fail the
    # whole-book agreement gate (matching.py) -- only a genuine reading
    # mismatch can.

    def __post_init__(self) -> None:
        object.__setattr__(self, "printed_page_number", _normalize_printed_page_number(self.printed_page_number))


def _toc_items_to_entries(items: list) -> list[TocEntry]:
    """Converts a parsed JSON array of {"title", "authors",
    "printed_page_number"} dicts -- the shape both vision.py's and
    ocr.py's prompts ask an LLM to return -- into TocEntry objects,
    tolerating the malformed-response shapes a real model occasionally
    produces. An optional "skip" key becomes TocEntry.skip (default
    False when absent)."""
    entries: list[TocEntry] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        if len(title) < 3:
            continue
        raw_authors = item.get("authors")
        authors = tuple(str(a).strip() for a in raw_authors if str(a).strip()) if isinstance(raw_authors, list) else ()
        printed_page_number = _normalize_printed_page_number(item.get("printed_page_number"))
        printed_roman = (
            printed_page_number is not None
            and not printed_page_number.isdigit()
            and _parse_toc_page_number(printed_page_number) is not None
        )
        entries.append(TocEntry(
            title=title, printed_page_number=printed_page_number, source_page_index=-1,
            authors=authors, printed_roman=printed_roman, skip=bool(item.get("skip", False)),
        ))
    return entries


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(line for line in lines if not line.startswith("```")).strip()
    return text


def parse_json_array(text: str) -> list:
    """Extract and parse the first JSON array ([...]) found in *text*."""
    text = _strip_code_fence(text)
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON array found in LLM response: {text!r}")
    return json.loads(text[start: end + 1])
