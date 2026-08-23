"""Smoke test for cli/compare_models.py -- the actual metrics are
tested at the library level in tests/test_model_agreement.py; this only
checks main() runs end-to-end against a tiny corpus without crashing
and prints the expected section headers."""

import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cli"))

from compare_models import main

from dnb_toc_ground_truth import corpus, vision
from dnb_toc_ground_truth.toc_entry import TocEntry


class TestMain(unittest.TestCase):
    def test_runs_end_to_end_and_prints_all_three_sections(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            corpus.manifest_path().write_text(
                json.dumps({"toc_only": True, "books": [{"filename": "111.pdf", "doi": None}]}),
                encoding="utf-8",
            )
            corpus.ground_truth_dir().mkdir(parents=True, exist_ok=True)
            corpus.expected_json_path("111").write_text(
                json.dumps({
                    "entries": [{"title": "1. Introduction", "authors": [], "printed_page_number": "1", "skip": False}],
                    "verified": True, "source": "agent_arbitration",
                }),
                encoding="utf-8",
            )
            entries = [TocEntry(title="1. Introduction", printed_page_number="1", source_page_index=0)]
            vision.write_cached_llm_entries(corpus.llm_cache_dir(), "111", "model/a", entries)
            vision.write_cached_llm_entries(corpus.llm_cache_dir(), "111", "model/b", entries)

            out = StringIO()
            with patch("sys.argv", ["compare_models.py", "--corpus", Path(tmp).name]), redirect_stdout(out):
                with patch.object(corpus, "_CORPUS_ROOT", Path(tmp).parent):
                    exit_code = main()

            self.assertEqual(exit_code, 0)
            output = out.getvalue()
            self.assertIn("Pairwise agreement", output)
            self.assertIn("Per-model accuracy", output)
            self.assertIn("Candidate pair ranking", output)

    def test_reports_no_cached_models_without_crashing(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            corpus.manifest_path().write_text(
                json.dumps({"toc_only": True, "books": []}), encoding="utf-8",
            )
            out = StringIO()
            with patch("sys.argv", ["compare_models.py", "--corpus", Path(tmp).name]), redirect_stdout(out):
                with patch.object(corpus, "_CORPUS_ROOT", Path(tmp).parent):
                    exit_code = main()
            self.assertEqual(exit_code, 0)
            self.assertIn("No cached model readings found", out.getvalue())


if __name__ == "__main__":
    unittest.main()
