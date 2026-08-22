"""NuExtract-family (numind/NuExtract3 today, a finetuned NuExtract2
later) template-mode TOC extraction -- an alternative to vision.py's/
ocr.py's free-text-prompt extraction path, for endpoints that declare
"extraction_api": "nuextract" in .endpoints (see inference.py). Ad hoc
testing against a live numind/NuExtract3 endpoint found that a free-text
prompt sent as ordinary chat content is effectively ignored by this
model family -- it only follows its own bespoke chat_template_kwargs
convention (an explicit JSON "template" plus an optional prose
"instructions" string), not a prompt embedded in the message content.
Calls client.chat.completions.create directly rather than through
inference.py's OpenAICompatibleLLMClient wrapper, since that wrapper's
generate() has no way to pass extra_body. See design spec
docs/superpowers/specs/2026-08-22-nuextract-template-mode-integration-design.md.

Text-mode quality is noticeably weaker than vision-mode on this
project's OCR'd text (dropped divider labels, one missed page number, an
author name duplicated into its title, in ad hoc testing against a real
book) -- documented here as a known limitation, not engineered around."""

import base64
import json
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from dnb_toc_ground_truth.toc_entry import TocEntry, _toc_items_to_entries, parse_json_array
from dnb_toc_ground_truth.vision import render_pages_to_images

_TEMPLATE = {
    "entries": [
        {
            "title": "verbatim-string",
            "authors": ["string"],
            "printed_page_number": "verbatim-string",
            "skip": "boolean",
        }
    ]
}

# Ported from vision.py's/ocr.py's free-text prompt rules, minus the
# "return ONLY a JSON array..." boilerplate -- template mode already
# governs output shape, so only the extraction RULES need spelling out
# here.
_INSTRUCTIONS = """\
Include EVERY printed line that names a titled section, even lines you \
might not think of as a real chapter: part/section dividers (e.g. "Part \
I", "Teil 1"), front matter (preface, list of contributors), and back \
matter (bibliography, index). Mark `skip` true for these non-chapter \
lines and false for actual chapters -- but always include the entry \
either way, never omit it.

A single chapter's title sometimes spans two printed lines (a short \
main title plus a longer subtitle right below it) with only ONE page \
number for the pair -- that is ONE entry, not two; join both lines into \
a single title string.

An indented, numbered, or lettered sub-point under a heading (e.g. \
"I.", "1.") that carries its OWN page number is its own separate entry \
too, not merged into its parent heading.

If a title is printed with a leading number, letter, or label (e.g. \
"1 ", "2.3 ", "I. ", "a) "), that label is part of the title -- include \
it verbatim as the start of the title string.

`printed_page_number` must be copied exactly as printed, including \
roman numerals (e.g. "vii", not 7). Use null only if no page number is \
visible anywhere on the line."""

_VISION_PROMPT_TEXT = "Extract every table-of-contents entry from this page."

# Same guard/escalation shape as vision.py's _MAX_VISION_PAGES/
# _VISION_MAX_TOKENS/_VISION_MAX_TOKENS_RETRY and ocr.py's equivalents --
# see those constants' own docstrings for the reasoning, unchanged here.
_MAX_PAGES = 20
_MAX_TOKENS = 4096
_MAX_TOKENS_RETRY = 8192


def _extra_body(use_instructions: bool) -> dict:
    kwargs: dict = {"template": json.dumps(_TEMPLATE), "enable_thinking": False}
    if use_instructions:
        kwargs["instructions"] = _INSTRUCTIONS
    return {"chat_template_kwargs": kwargs}


async def nuextract_vision_extract_toc_entries(
    pdf_path: Path, model: str, client: Any, *, use_instructions: bool = True, pdftoppm_bin: str = "pdftoppm",
) -> list[TocEntry]:
    """Template-mode counterpart to vision.py's vision_extract_toc_entries
    -- same page-rendering, page-count cap, max-tokens escalation, and
    raises-on-failure contract, but sends NuExtract's own
    chat_template_kwargs (template + optional instructions) instead of a
    free-text prompt, per this module's own docstring."""
    page_count = len(PdfReader(str(pdf_path)).pages)
    if page_count > _MAX_PAGES:
        raise ValueError(f"{pdf_path}: {page_count} pages exceeds vision-extraction cap of {_MAX_PAGES}")
    images = render_pages_to_images(pdf_path, pdftoppm_bin=pdftoppm_bin)
    content: list[dict] = [{"type": "text", "text": _VISION_PROMPT_TEXT}]
    for image_bytes in images:
        b64 = base64.b64encode(image_bytes).decode("ascii")
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
    messages = [{"role": "user", "content": content}]
    extra_body = _extra_body(use_instructions)

    last_error: Exception | None = None
    for max_tokens in (_MAX_TOKENS, _MAX_TOKENS_RETRY):
        response = await client.chat.completions.create(
            model=model, messages=messages, max_tokens=max_tokens, temperature=0.0, extra_body=extra_body,
        )
        raw = response.choices[0].message.content or ""
        try:
            items = parse_json_array(raw)
            return _toc_items_to_entries(items)
        except Exception as exc:  # noqa: BLE001 -- any parse failure triggers the escalation retry
            last_error = exc
    raise last_error
