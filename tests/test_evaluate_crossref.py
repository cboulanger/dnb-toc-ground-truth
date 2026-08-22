"""Unit tests for cli/evaluate_crossref.py's --backfill CLI wiring. The
actual backfill logic (extraction, cache writing) is tested at the
library level in tests/test_crossref_evaluation.py -- this only tests
that _run_backfill resolves --model against --endpoints-file correctly
and fails loudly when it can't, mirroring
tests/test_generate_ground_truth.py's TestResolveEndpoints convention of
testing the internal helper directly rather than main() via sys.argv."""

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from pypdf import PdfWriter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cli"))

from evaluate_crossref import _run_backfill

from dnb_toc_ground_truth import corpus, crossref_evaluation, vision


def _args(**overrides) -> argparse.Namespace:
    defaults = dict(model=None, endpoints_file=None)
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _write_evaluation_json(key: str, entries: list[dict]) -> None:
    corpus.evaluation_dir().mkdir(parents=True, exist_ok=True)
    corpus.evaluation_json_path(key).write_text(
        json.dumps({"entries": entries, "source": "crossref", "fetched_at": ""}), encoding="utf-8",
    )


def _make_pdf(path: Path) -> Path:
    writer = PdfWriter()
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


class TestRunBackfill(unittest.TestCase):
    def test_raises_without_a_matching_endpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            endpoints_path = Path(tmp) / ".endpoints"
            endpoints_path.write_text(json.dumps([
                {"url": "https://x.invalid/a", "key": "k", "model": "some-other-model"},
            ]), encoding="utf-8")
            args = _args(model=["nonexistent/model"], endpoints_file=endpoints_path)
            with self.assertRaises(ValueError):
                _run_backfill(args, {})

    def test_requires_at_least_one_model(self):
        args = _args(model=None)
        with self.assertRaises(SystemExit):
            _run_backfill(args, {})

    def test_a_successful_resolution_calls_backfill_model_cache_with_the_resolved_endpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            endpoints_path = Path(tmp) / ".endpoints"
            endpoints_path.write_text(json.dumps([
                {"url": "https://x.invalid/a", "key": "k", "model": "some/model"},
            ]), encoding="utf-8")
            args = _args(model=["some/model"], endpoints_file=endpoints_path)
            with patch(
                "evaluate_crossref.backfill_model_cache",
                new=AsyncMock(return_value=(["some-key"], [])),
            ) as mock_backfill:
                _run_backfill(args, {})

            mock_backfill.assert_awaited_once()
            called_model, called_endpoint, called_cache_dir = mock_backfill.await_args.args
            self.assertEqual(called_model, "some/model")
            self.assertEqual(called_endpoint.model_id, "some/model")
            self.assertEqual(called_cache_dir, corpus.llm_cache_dir())

    def test_a_real_endpoints_file_and_backfill_flow_actually_writes_and_is_scored(self):
        # Join-point test: every layer above is well-covered in isolation
        # (this class's other tests mock backfill_model_cache itself;
        # TestBackfillModelCache in test_crossref_evaluation.py hand-builds
        # ModelEndpoint objects rather than routing through real
        # .endpoints-file resolution), but nothing drives the FULL chain --
        # a real .endpoints file, through the real
        # load_endpoint_entries/resolve_model_endpoints, into the real
        # backfill_model_cache, a real cache write, and then a real
        # evaluate_model_corpus read of what was just written. This closes
        # that gap.
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp) / "corpus"):
            tmp_path = Path(tmp)
            key = "9783899718188"
            model = "some/model"

            manifest_path = corpus.manifest_path()
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                json.dumps({"toc_only": True, "books": [{"filename": f"{key}.pdf", "doi": "10.1/x"}]}),
                encoding="utf-8",
            )
            corpus.pdf_dir().mkdir(parents=True, exist_ok=True)
            _make_pdf(corpus.pdf_path(key))
            _write_evaluation_json(key, [
                {"title": "Introduction", "authors": [], "printed_page_number": "1", "skip": False},
            ])

            endpoints_path = tmp_path / ".endpoints"
            endpoints_path.write_text(json.dumps([
                {"url": "https://x.invalid/a", "key": "k1", "model": model},
            ]), encoding="utf-8")
            args = _args(model=[model], endpoints_file=endpoints_path)

            client = _fake_client(
                '[{"title": "1. Introduction", "authors": [], "printed_page_number": "1", "skip": false}]'
            )
            # resolve_model_endpoints builds a real AsyncOpenAI client from
            # the .endpoints file's url/key -- swap the class it uses to
            # construct that client so no real network call happens, while
            # leaving load_endpoint_entries/resolve_model_endpoints'
            # parsing and matching logic, and _run_backfill's own
            # resolution + dispatch to the real backfill_model_cache,
            # completely real.
            with patch("dnb_toc_ground_truth.inference.AsyncOpenAI", return_value=client):
                _run_backfill(args, {})

            cached = vision.load_cached_llm_entries(corpus.llm_cache_dir(), key, model)
            self.assertIsNotNone(cached)
            self.assertEqual(cached[0].title, "1. Introduction")

            results, no_cache = crossref_evaluation.evaluate_model_corpus(model)
            self.assertEqual(no_cache, [])
            self.assertEqual([r.key for r in results], [key])


if __name__ == "__main__":
    unittest.main()
