"""Vision-LLM TOC extraction for dnb-toc-only -- reads each book's page
images directly (no OCR, no text layer at all), per design spec
docs/superpowers/specs/2026-08-16-dnb-toc-uniform-ocr-design.md section 3.
dnb-toc-only's PDFs are pre-filtered to just their TOC pages during
acquisition (1-3 pages typically), so rendering every page unconditionally
is cheap and bounded -- this does NOT generalize to whole-book PDFs.

Also owns caching vision_extract_toc_entries's own per-(book, model)
results (cache_path/load_cached_llm_entries/write_cached_llm_entries) --
kept here rather than in a calling script so more than one script
(cli/generate_ground_truth.py, cli/arbitrate.py) can read the
same cache without importing one script from another."""

import base64
import json
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

from pypdf import PdfReader

from dnb_toc_ground_truth.toc_entry import TocEntry, _toc_items_to_entries, parse_json_array


# Bump whenever vision_extract_toc_entries' output shape or the prompt's
# extraction standard changes in a way that makes a previously-cached
# result stale (not just wrong on some books, but built to a different
# spec) -- 2026-08-17: switched from "omit non-chapter lines entirely" to
# "include every printed line verbatim, flag non-chapters via
# TocEntry.skip instead" (see TocEntry.skip's own docstring for why). Old
# cache under a previous version is deliberately left on disk rather than
# deleted -- some of it is already git-committed -- but lives in its own
# subdirectory so a version bump here can never be mistaken for fresh data;
# only load_cached_llm_entries/write_cached_llm_entries/cache_path need to
# know about this constant, since every caller already goes through them
# rather than computing cache paths itself (cli/arbitrate.py's own
# globbing uses the versioned_cache_dir helper below for the same reason).
_CACHE_SCHEMA_VERSION = "v2"


def versioned_cache_dir(cache_directory: Path) -> Path:
    """The actual directory this schema version's cache files live in --
    cache_directory itself may also hold one or more OLDER versions'
    files, never read by current code. Exposed (not `_`-prefixed) so
    cli/arbitrate.py's own cache-directory globbing stays in sync
    without duplicating the version constant."""
    return cache_directory / _CACHE_SCHEMA_VERSION


def cache_path(cache_directory: Path, key: str, model: str) -> Path:
    """Where one (book, model) pair's cached vision_extract_toc_entries
    result lives -- <key>.<model>.json, so two independent models never
    collide for the same book. `model` is sanitized ("/" -> "__") before
    interpolation -- some OpenAI-compatible endpoints (e.g. MPCDF) use a
    real HuggingFace repo id as their model id (e.g.
    "Qwen/Qwen2.5-VL-7B-Instruct"), and an un-sanitized "/" would be
    interpreted as a path separator, turning this into a nested directory
    instead of a flat file. A no-op for any endpoint whose model id
    already has no slash."""
    safe_model = model.replace("/", "__")
    return versioned_cache_dir(cache_directory) / f"{key}.{safe_model}.json"


def load_cached_llm_entries(cache_directory: Path, key: str, model: str) -> Optional[list[TocEntry]]:
    """Returns the cached result for (key, model), or None on a cache
    miss (never seen this pair, or the call that would have populated it
    failed/returned empty -- write_cached_llm_entries only caches
    non-empty results, see its own docstring)."""
    path = cache_path(cache_directory, key, model)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        TocEntry(
            title=e["title"], printed_page_number=e["printed_page_number"],
            source_page_index=e["source_page_index"], authors=tuple(e["authors"]),
            printed_roman=e["printed_roman"], skip=e.get("skip", False),
        )
        for e in data["entries"]
    ]


def write_cached_llm_entries(
    cache_directory: Path, key: str, model: str, entries: list[TocEntry], *, kind: str = "vision",
) -> None:
    """Caches entries for (key, model). Callers should only call this
    with a non-empty entries list -- an empty result could be a genuine
    "no TOC content" or a transient failure, and caching it either way
    would make a later re-run trust a possibly-transient empty result
    forever instead of retrying. `kind` ("vision" or "text", default
    "vision") records which extraction path produced these entries --
    see load_cached_kind's own docstring for how a caller reads it back."""
    path = cache_path(cache_directory, key, model)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "generated_at": time.time(),
        "kind": kind,
        "entries": [
            {
                "title": e.title, "printed_page_number": e.printed_page_number,
                "source_page_index": e.source_page_index, "authors": list(e.authors),
                "printed_roman": e.printed_roman, "skip": e.skip,
            }
            for e in entries
        ],
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_cached_kind(cache_directory: Path, key: str, model: str) -> str:
    """Which extraction path produced (key, model)'s cached entries --
    "vision" or "text". Reads the "kind" field write_cached_llm_entries
    writes; absent on any cache file written before this field existed
    (every cache file to date came from vision_extract_toc_entries), which
    is treated as "vision" for full backward compatibility -- see design
    spec docs/superpowers/specs/2026-08-20-dnb-toc-vision-text-pairing-
    design.md section 4. Also returns "vision" if the cache file doesn't
    exist at all -- callers (cli/arbitrate.py) only ever call this for
    a (key, model) pair they already know has a cache file, but this keeps
    the function total rather than raising on a caller's bug."""
    path = cache_path(cache_directory, key, model)
    if not path.exists():
        return "vision"
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("kind", "vision")


_VISION_TOC_EXTRACTION_PROMPT = """\
You are reading photographed/scanned page images of a book's table of \
contents. Some layouts use simple dotted leaders ("Title ..... 12"), \
others print the author's name on its own line above or below the title, \
or right-align the page number with no leader at all -- read the images \
directly rather than assuming one fixed layout.

A heading can have indented, numbered, or lettered sub-points listed \
below it (e.g. "I.", "II.", "1.", "2.") that each carry their OWN page \
number -- each such sub-point is its own separate entry too, exactly as \
printed, not merged into its parent heading. Do not collapse or omit a \
sub-point just because it is indented under a larger heading.

A single chapter's title sometimes spans two printed lines -- a short \
main title followed by a longer explanatory subtitle right below it \
(or vice versa) -- with only ONE page number for the pair. That is ONE \
entry, not two: join both lines into a single title string. Do not \
create a separate entry for the subtitle line, and do not create a \
separate entry with no page number just because a line of text sits \
above a chapter's title.

Return ONLY a JSON array, one entry for EVERY line printed on the page \
that names a titled section and (usually) a page number -- transcribe \
what is actually printed, do not decide which lines matter. This \
includes lines you might not think of as a "real chapter": a \
part/section divider (e.g. "Teil 1", "I. Historische Grundlagen", an \
unnumbered section-title line that groups several chapters under it), \
front matter (preface, foreword, acknowledgements, list of \
contributors/authors), and back matter (bibliography, index, an \
appendix listing an author's or honoree's own prior publications) all \
get their own entry too, exactly like any other line, even when they \
carry no page number of their own. Mark each entry "skip": true if it is \
one of these non-chapter lines (a divider, front matter, or back \
matter) and "skip": false if it is an actual chapter -- but include the \
entry either way; never omit a printed line because of what "skip" \
value it would get:
[{"title": "...", "authors": ["First Last", ...], "printed_page_number": "12", "skip": false}]

printed_page_number is the page number exactly AS PRINTED on the page -- \
copy it verbatim, including roman numerals for front-matter chapters \
(e.g. "vii", not 7). If a line's printed page number is not visible, use \
null for printed_page_number -- never leave the line out just because it \
has no page number. If authors are not identifiable, use an empty list. \
Almost every chapter/section line DOES have a page number printed \
somewhere on it, even when there is no dot leader connecting it to the \
title -- look along the whole line, including its far right edge, before \
writing null; null should be rare, reserved for lines that genuinely show \
no number anywhere (most commonly a divider or heading with no page of \
its own), not a default answer.

If a title is printed with a leading number, letter, or label (e.g. "1 ", \
"2.3 ", "I. ", "a) "), that label is part of the title -- include it \
verbatim as the start of the title string. Do not strip, renumber, or \
omit any such printed label."""

# Rendered image count this corpus's PDFs never exceed today (1-3 pages,
# per the acquisition pipeline's own TOC-only filtering) -- guards against
# silently building an arbitrarily large multi-image request if a
# mis-filtered outlier ever slips through (design spec section 5).
_MAX_VISION_PAGES = 20

# Mirrors segmentation.py's _LLM_TOC_RETRY_MAX_TOKENS escalation: a dense
# multi-page edited-volume TOC (60-80 entries, long German titles, author
# lists) can plausibly overrun the first attempt's budget. A truncated JSON
# array reliably fails parse_json_array (no closing "]") regardless of
# cause, so JSON-parseability alone is a sufficient, client-agnostic retry
# trigger -- without this, _call_with_retry's outer wrapper would reissue an
# IDENTICAL request at temperature=0.0 and fail the same way every time.
_VISION_MAX_TOKENS = 4096
_VISION_MAX_TOKENS_RETRY = 8192


def render_pages_to_images(pdf_path: Path, dpi: int = 200, pdftoppm_bin: str = "pdftoppm") -> list[bytes]:
    """Rasterizes every page of pdf_path to PNG bytes, in page order, via
    pdftoppm -- no OCR, no text extraction. Follows the same
    resolve_binary-able-external-tool and glob-then-sort conventions
    evaluation/scripts/clean_scanned_pdf.py already uses for pdftoppm."""
    with tempfile.TemporaryDirectory() as tmp:
        prefix = Path(tmp) / "page"
        result = subprocess.run(
            [pdftoppm_bin, "-r", str(dpi), "-png", str(pdf_path), str(prefix)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"pdftoppm failed on {pdf_path}: {result.stderr}")
        matches = sorted(Path(tmp).glob("page-*.png"))
        if not matches:
            raise RuntimeError(f"pdftoppm produced no output for {pdf_path}")
        return [p.read_bytes() for p in matches]


async def vision_extract_toc_entries(pdf_path: Path, model: str, client: Any, *, pdftoppm_bin: str = "pdftoppm") -> list[TocEntry]:
    """Renders every page of pdf_path and asks a vision-capable model
    (via an already-constructed openai.AsyncOpenAI-shaped `client`, model
    id `model`) to read the table of contents directly from the images.
    Same return shape as llm_extract_toc_entries, sharing its item-parsing
    tolerance logic (_toc_items_to_entries).

    Deliberately RAISES on any failure (network error, malformed JSON)
    rather than catching and returning [] the way llm_extract_toc_entries
    does -- that swallowing made the text pipeline's _call_with_retry
    wrapper dead code (llm_extract_toc_entries never actually raised to
    it). Here, the caller's retry wrapper does real work.

    Escalates max_tokens once (_VISION_MAX_TOKENS -> _VISION_MAX_TOKENS_RETRY)
    if the first response doesn't parse as a complete JSON array, mirroring
    segmentation.py's _extract_with_retry -- otherwise a truncated response
    would fail identically on every one of the caller's retry attempts
    (same images, same prompt, temperature=0.0)."""
    page_count = len(PdfReader(str(pdf_path)).pages)
    if page_count > _MAX_VISION_PAGES:
        raise ValueError(f"{pdf_path}: {page_count} pages exceeds vision-extraction cap of {_MAX_VISION_PAGES}")
    images = render_pages_to_images(pdf_path, pdftoppm_bin=pdftoppm_bin)
    content: list[dict] = [{"type": "text", "text": _VISION_TOC_EXTRACTION_PROMPT}]
    for image_bytes in images:
        b64 = base64.b64encode(image_bytes).decode("ascii")
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
    messages = [{"role": "user", "content": content}]

    last_error: Exception | None = None
    for max_tokens in (_VISION_MAX_TOKENS, _VISION_MAX_TOKENS_RETRY):
        response = await client.chat.completions.create(
            model=model, messages=messages, max_tokens=max_tokens, temperature=0.0,
        )
        raw = response.choices[0].message.content or ""
        try:
            items = parse_json_array(raw)
            return _toc_items_to_entries(items)
        except Exception as exc:  # noqa: BLE001 -- any parse failure triggers the escalation retry
            last_error = exc
    raise last_error
