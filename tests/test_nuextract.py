"""Unit tests for nuextract.py -- NuExtract-family template-mode TOC
extraction, see design spec
docs/superpowers/specs/2026-08-22-nuextract-template-mode-integration-design.md.
Both extraction functions are tested with a mocked OpenAI-shaped client,
no real network call; nuextract_text_extract_toc_entries mocks
ocr_pages_to_rows the same way test_ocr.py's text_extract_toc_entries
tests do."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from pypdf import PdfWriter

from dnb_toc_ground_truth.nuextract import _MAX_PAGES, nuextract_vision_extract_toc_entries


def _make_pdf(path: Path, page_count: int) -> Path:
    writer = PdfWriter()
    for _ in range(page_count):
        writer.add_blank_page(width=200, height=200)
    with open(path, "wb") as f:
        writer.write(f)
    return path


def _fake_response(response_text: str):
    message = MagicMock()
    message.content = response_text
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


def _fake_client(response_text: str):
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=_fake_response(response_text))
    return client


def _fake_client_sequence(*response_texts: str):
    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=[_fake_response(t) for t in response_texts])
    return client


_NUEXTRACT_RESPONSE = json.dumps({
    "entries": [
        {"title": "Einleitung", "authors": [], "printed_page_number": "9", "skip": False},
        {"title": "Bibliographie", "authors": [], "printed_page_number": "200", "skip": True},
    ]
})


class TestNuextractVisionExtractTocEntries(unittest.IsolatedAsyncioTestCase):
    async def test_parses_the_entries_wrapper_shape_into_toc_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = _make_pdf(Path(tmp) / "book.pdf", 1)
            client = _fake_client(_NUEXTRACT_RESPONSE)
            entries = await nuextract_vision_extract_toc_entries(pdf_path, "numind/NuExtract3", client)
            self.assertEqual(len(entries), 2)
            self.assertEqual(entries[0].title, "Einleitung")
            self.assertFalse(entries[0].skip)
            self.assertTrue(entries[1].skip)

    async def test_sends_template_and_instructions_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = _make_pdf(Path(tmp) / "book.pdf", 1)
            client = _fake_client(_NUEXTRACT_RESPONSE)
            await nuextract_vision_extract_toc_entries(pdf_path, "numind/NuExtract3", client)
            extra_body = client.chat.completions.create.call_args.kwargs["extra_body"]
            self.assertIn("template", extra_body["chat_template_kwargs"])
            self.assertIn("instructions", extra_body["chat_template_kwargs"])

    async def test_omits_instructions_when_use_instructions_is_false(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = _make_pdf(Path(tmp) / "book.pdf", 1)
            client = _fake_client(_NUEXTRACT_RESPONSE)
            await nuextract_vision_extract_toc_entries(pdf_path, "numind/NuExtract3", client, use_instructions=False)
            extra_body = client.chat.completions.create.call_args.kwargs["extra_body"]
            self.assertNotIn("instructions", extra_body["chat_template_kwargs"])

    async def test_sends_one_image_content_block_per_page(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = _make_pdf(Path(tmp) / "book.pdf", 2)
            client = _fake_client(json.dumps({"entries": []}))
            await nuextract_vision_extract_toc_entries(pdf_path, "numind/NuExtract3", client)
            messages = client.chat.completions.create.call_args.kwargs["messages"]
            content = messages[0]["content"]
            image_blocks = [c for c in content if c["type"] == "image_url"]
            self.assertEqual(len(image_blocks), 2)

    async def test_raises_on_malformed_json_instead_of_swallowing(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = _make_pdf(Path(tmp) / "book.pdf", 1)
            client = _fake_client("not json at all")
            with self.assertRaises(Exception):
                await nuextract_vision_extract_toc_entries(pdf_path, "numind/NuExtract3", client)

    async def test_raises_before_any_network_call_when_page_count_exceeds_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = _make_pdf(Path(tmp) / "book.pdf", _MAX_PAGES + 1)
            client = _fake_client(json.dumps({"entries": []}))
            with self.assertRaises(ValueError):
                await nuextract_vision_extract_toc_entries(pdf_path, "numind/NuExtract3", client)
            client.chat.completions.create.assert_not_called()

    async def test_escalates_max_tokens_and_recovers_from_a_truncated_first_response(self):
        truncated = '{"entries": [{"title": "Einleitung"'
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = _make_pdf(Path(tmp) / "book.pdf", 1)
            client = _fake_client_sequence(truncated, _NUEXTRACT_RESPONSE)
            entries = await nuextract_vision_extract_toc_entries(pdf_path, "numind/NuExtract3", client)
            self.assertEqual(len(entries), 2)
            self.assertEqual(client.chat.completions.create.await_count, 2)
            first_call_kwargs = client.chat.completions.create.call_args_list[0].kwargs
            second_call_kwargs = client.chat.completions.create.call_args_list[1].kwargs
            self.assertEqual(first_call_kwargs["max_tokens"], 4096)
            self.assertEqual(second_call_kwargs["max_tokens"], 8192)
