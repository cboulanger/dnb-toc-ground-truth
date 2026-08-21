"""OCR'd-text TOC extraction for dnb-toc-only's vision+text-model pairing
-- feeds a text-only LLM the book's TOC pages reconstructed as plain text
via ocrmypdf + pdfalto, instead of vision_extract_toc_entries' page
images. See design spec
docs/superpowers/specs/2026-08-20-dnb-toc-vision-text-pairing-design.md
section 3. Structurally parallel to vision.py, reusing
its cache_path/load_cached_llm_entries/write_cached_llm_entries directly
(both extraction paths share one cache, keyed by (book, model) -- see
that module's own "kind" field for how a cached entry records which
extraction path produced it)."""

import os
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from dnb_toc_ground_truth.toc_entry import TocEntry, _toc_items_to_entries, parse_json_array
from dnb_toc_ground_truth.inference import OpenAICompatibleLLMClient
from dnb_toc_ground_truth import pdfalto_runner

_ALTO_NS = "{http://www.loc.gov/standards/alto/ns-v3#}"

# VPOS tolerance (ALTO points/px) for clustering <String> tokens into one
# reconstructed row -- see design spec section 3 and
# docs/superpowers/specs/2026-08-16-dnb-toc-uniform-ocr-design.md section
# 1b, which found pdfalto's own <TextBlock>/<TextLine> segmentation groups
# a dot-leader TOC's title column and page-number column into SEPARATE
# blocks (all titles, then all page numbers) rather than one block per
# printed line -- exactly the failure this raw-<String>-position
# reclustering avoids by ignoring ALTO's own TextBlock/TextLine nesting
# entirely and re-deriving rows purely from token geometry.
_ROW_VPOS_TOLERANCE = 8.0


def _rows_from_alto_xml(alto_path: Path) -> list[str]:
    """One reading-order-reconstructed text block per ALTO <Page>, in page
    order. Clusters every <String> token on a page by VPOS (tolerance
    _ROW_VPOS_TOLERANCE) into rows, sorts each row's tokens by HPOS, and
    joins them with a single space -- deliberately ignores ALTO's own
    <TextBlock>/<TextLine> nesting (see _ROW_VPOS_TOLERANCE's docstring for
    why trusting it doesn't work for this corpus's dot-leader TOCs). A page
    with no <String> tokens at all (e.g. a blank page pdfalto still emits
    an empty <Page> element for) produces an empty string, not an error --
    ocr_pages_to_rows's caller needs one entry per page to keep its
    per-page-list shape aligned with render_pages_to_images'."""
    root = ET.parse(alto_path).getroot()
    page_rows: list[str] = []
    for page in root.iter(_ALTO_NS + "Page"):
        tokens = sorted(
            (
                (float(string.get("VPOS", "0")), float(string.get("HPOS", "0")), string.get("CONTENT", ""))
                for string in page.iter(_ALTO_NS + "String")
                if string.get("CONTENT")
            ),
            key=lambda token: (token[0], token[1]),
        )
        clusters: list[list[tuple[float, float, str]]] = []
        for token in tokens:
            if clusters and token[0] - clusters[-1][0][0] <= _ROW_VPOS_TOLERANCE:
                clusters[-1].append(token)
            else:
                clusters.append([token])
        rows = [
            " ".join(content for _, _, content in sorted(cluster, key=lambda t: t[1]))
            for cluster in clusters
        ]
        page_rows.append("\n".join(rows))
    return page_rows


_TESSDATA_BEST_DIR_ENV_VAR = "TESSDATA_BEST_DIR"


def _resolve_tessdata_best_env(languages: tuple[str, ...] = ("deu", "eng")) -> dict[str, str] | None:
    """Resolves an optional subprocess environment override pointing
    ocrmypdf/tesseract at a tessdata_best directory instead of whatever
    ships by default (Homebrew's tesseract-lang formula ships
    tessdata_fast only -- there is no Homebrew formula for tessdata_best,
    so this is opt-in via the TESSDATA_BEST_DIR environment variable after
    a manual per-language download from
    https://github.com/tesseract-ocr/tessdata_best). Returns
    None (no env override -- ocrmypdf uses whatever tessdata is already on
    PATH/its default location) when the variable isn't set at all; purely
    opt-in, no default guessed. When it IS set, validates the directory
    actually contains every requested language's .traineddata file and
    raises RuntimeError naming exactly what's missing if not -- a
    misconfigured explicit request should fail loudly with an actionable
    message, not silently fall back to the default or surface as a
    cryptic tesseract error deep inside a subprocess (same
    raise-naming-what's-wrong convention
    inference.py's resolve_model_endpoints already
    established for a similar env-var-driven setup step)."""
    directory = os.environ.get(_TESSDATA_BEST_DIR_ENV_VAR)
    if not directory:
        return None
    if not Path(directory).is_dir():
        raise RuntimeError(f"{_TESSDATA_BEST_DIR_ENV_VAR}={directory} does not exist or is not a directory")
    missing = [lang for lang in languages if not (Path(directory) / f"{lang}.traineddata").exists()]
    if missing:
        raise RuntimeError(
            f"{_TESSDATA_BEST_DIR_ENV_VAR}={directory} is missing traineddata for: {', '.join(missing)} -- "
            f"download from https://github.com/tesseract-ocr/tessdata_best"
        )
    return {**os.environ, "TESSDATA_PREFIX": directory}


def ocr_pages_to_rows(pdf_path: Path, *, pdfalto_bin: str | None = None) -> list[str]:
    """Forces fresh OCR on pdf_path (ocrmypdf --force-ocr, unconditionally
    -- this corpus's PDFs are pre-filtered to 1-3 TOC pages, so re-OCRing
    even an already-text-layered PDF is cheap and keeps behavior uniform
    regardless of the source PDF's own text layer quality), then runs
    pdfalto and reconstructs reading order via _rows_from_alto_xml. Returns
    one string per page, in page order -- the same per-page-list shape
    render_pages_to_images (vision.py) returns for
    images, so the vision and text extraction paths stay visually parallel
    in any calling code. `pdfalto_bin` is passed straight through to
    pdfalto_runner.resolve_pdfalto_binary -- None (the default) resolves
    via the PDFALTO_BIN environment variable, then a bare "pdfalto" on
    PATH; pdfalto is a sibling checkout, not on PATH by default (see
    this repo's README.md "Setup" section for the pdfalto
    sibling-checkout notes). Raises
    RuntimeError if ocrmypdf exits non-zero -- propagates to the caller
    exactly like any other extraction failure, no special-casing."""
    resolved_pdfalto_bin = pdfalto_runner.resolve_pdfalto_binary(pdfalto_bin)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        ocr_pdf_path = tmp_dir / f"{pdf_path.stem}.ocr.pdf"
        result = subprocess.run(
            ["ocrmypdf", "--force-ocr", "-l", "deu+eng", str(pdf_path), str(ocr_pdf_path)],
            capture_output=True, text=True, env=_resolve_tessdata_best_env(),
        )
        if result.returncode != 0:
            raise RuntimeError(f"ocrmypdf failed on {pdf_path}: {result.stderr}")
        alto_path = pdfalto_runner.ensure_alto_xml(ocr_pdf_path, tmp_dir, resolved_pdfalto_bin)
        return _rows_from_alto_xml(alto_path)


_TEXT_TOC_EXTRACTION_PROMPT = """\
You are reading a machine-OCR'd transcription of a book's table of \
contents pages, with each printed line's reading order already \
reconstructed for you. The OCR process can still introduce scanning \
artifacts: misrecognized characters, a run of garbled tokens where a \
printed dot leader ("....") was misread as text, or an occasional \
dropped or duplicated word. Read past such artifacts and report the \
actual title/page-number content a human would recognize the line as \
saying -- do not transcribe OCR noise literally as if it were real text.

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

Return ONLY a JSON array, one entry for EVERY line in the transcription \
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
Almost every chapter/section entry DOES have a page number in the \
transcription, even when it is not visually separated from the title by \
a dot leader -- the OCR reconstruction frequently glues it directly onto \
the end of the entry's own last line with nothing but a single space and \
no punctuation (e.g. a block ending "...in die\\nThematik 9" means the \
title text ends at "Thematik" and "9" is that entry's \
printed_page_number, NOT part of the title). Treat a standalone number \
appearing at the very end of an entry's text this way. Before writing \
null, re-check the end of the entry's own text for a trailing number \
like this; null should be rare, reserved for lines that genuinely show no \
number anywhere (most commonly a divider or heading with no page of its \
own), not a default answer.

If a title is printed with a leading number, letter, or label (e.g. "1 ", \
"2.3 ", "I. ", "a) "), that label is part of the title -- include it \
verbatim as the start of the title string. Do not strip, renumber, or \
omit any such printed label."""

# Same escalation shape as vision.py's
# _VISION_MAX_TOKENS/_VISION_MAX_TOKENS_RETRY -- a truncated JSON array
# reliably fails parse_json_array regardless of cause, so JSON-parseability
# alone is a sufficient retry trigger. A text response has no image tokens
# inflating the prompt, so budget pressure here is milder than vision, but
# the escalation costs nothing to keep for consistency.
_TEXT_MAX_TOKENS = 4096
_TEXT_MAX_TOKENS_RETRY = 8192

# Same guard and reasoning as vision.py's
# _MAX_VISION_PAGES -- this corpus's PDFs never exceed 1-3 pages today (the
# acquisition pipeline's own TOC-only filtering), so this only matters if a
# mis-filtered outlier ever slips through; a text prompt doesn't inflate
# per-page cost the way a multi-image vision request does, but an unbounded
# page count could still silently build an arbitrarily long OCR'd-text
# prompt (and pay for an unbounded ocrmypdf/pdfalto run) for such an
# outlier, so the same cap applies here too rather than omitting it.
_MAX_TEXT_PAGES = 20


async def text_extract_toc_entries(
    pdf_path: Path, model: str, client: Any, *, pdfalto_bin: str | None = None,
) -> list[TocEntry]:
    """OCRs pdf_path (ocr_pages_to_rows) and asks a text-only model (via an
    already-constructed openai.AsyncOpenAI-shaped `client`, model id
    `model`) to extract the table of contents from the reconstructed page
    text. Same return shape as vision_extract_toc_entries, sharing its
    item-parsing tolerance logic (_toc_items_to_entries) and its
    raises-on-failure/max_tokens-escalation contract -- see that
    function's own docstring in vision.py for why
    swallowing failures internally would be wrong here too. Does not catch
    exceptions from ocr_pages_to_rows -- an OCR failure propagates exactly
    like any other extraction failure, no special-casing (design spec
    section "Error handling"). Raises ValueError before any OCR/network
    call if pdf_path has more than _MAX_TEXT_PAGES pages -- same guard as
    vision_extract_toc_entries's _MAX_VISION_PAGES check, see that
    constant's own docstring."""
    page_count = len(PdfReader(str(pdf_path)).pages)
    if page_count > _MAX_TEXT_PAGES:
        raise ValueError(f"{pdf_path}: {page_count} pages exceeds text-extraction cap of {_MAX_TEXT_PAGES}")
    page_texts = ocr_pages_to_rows(pdf_path, pdfalto_bin=pdfalto_bin)
    pages_block = "\n\n".join(f"--- Page {i + 1} ---\n{text}" for i, text in enumerate(page_texts))
    prompt = f"{_TEXT_TOC_EXTRACTION_PROMPT}\n\n{pages_block}"
    llm_client = OpenAICompatibleLLMClient(model=model, client=client)

    last_error: Exception | None = None
    for max_tokens in (_TEXT_MAX_TOKENS, _TEXT_MAX_TOKENS_RETRY):
        raw = await llm_client.generate(prompt, max_tokens=max_tokens, temperature=0.0)
        try:
            items = parse_json_array(raw)
            return _toc_items_to_entries(items)
        except Exception as exc:  # noqa: BLE001 -- any parse failure triggers the escalation retry
            last_error = exc
    raise last_error
