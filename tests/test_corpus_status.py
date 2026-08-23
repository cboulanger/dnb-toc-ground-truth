"""Unit tests for cli/corpus_status.py's pure logic. No real corpus
paths -- all fixtures live in tempdirs, with
dnb_toc_ground_truth.corpus.CORPUS_DIR patched per test so the module's
path helpers (which read the module-level constant fresh at call time)
resolve into isolated per-test directories."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dnb_toc_ground_truth import corpus
from dnb_toc_ground_truth.toc_entry import TocEntry
from dnb_toc_ground_truth.vision import write_cached_llm_entries

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cli"))

import corpus_status
from corpus_status import _MARKER_END, _MARKER_START, build_status_table, update_readme


def _entry(title: str, page: int) -> TocEntry:
    return TocEntry(title=title, printed_page_number=page, source_page_index=0)


def _write_manifest(filenames: list[str]) -> None:
    corpus.corpus_dir().mkdir(parents=True, exist_ok=True)
    corpus.manifest_path().write_text(
        json.dumps({"books": [{"filename": f} for f in filenames]}), encoding="utf-8"
    )


def _write_ground_truth(key: str, source: str) -> None:
    corpus.ground_truth_dir().mkdir(parents=True, exist_ok=True)
    corpus.expected_json_path(key).write_text(
        json.dumps({"entries": [{"title": "X", "authors": [], "printed_page_number": "1", "skip": False}],
                    "verified": source == "agent_arbitration", "source": source}),
        encoding="utf-8",
    )


class TestBuildStatusTable(unittest.TestCase):
    def test_counts_every_metric_from_a_small_fixture_corpus(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(corpus, "CORPUS_DIR", Path(tmp) / "corpus"),
            patch.object(corpus_status, "_MIN_MODEL_READINGS", 1),
        ):
            _write_manifest(["book1.pdf", "book2.pdf", "book3.pdf", "book4.pdf"])

            write_cached_llm_entries(corpus.llm_cache_dir(), "book1", "model-a", [_entry("X", 1)])
            write_cached_llm_entries(corpus.llm_cache_dir(), "book2", "model-a", [_entry("X", 1)])
            write_cached_llm_entries(corpus.llm_cache_dir(), "book2", "model-b", [_entry("X", 1)])
            write_cached_llm_entries(corpus.llm_cache_dir(), "book3", "model-a", [_entry("X", 1)])

            _write_ground_truth("book1", "bulk_gate")
            _write_ground_truth("book2", "agent_arbitration")
            # book3 has a cache entry but no ground truth yet -> awaiting arbitration
            # book4 has neither -> not counted anywhere but the manifest total

            corpus.arbitration_rejected_path().write_text(
                json.dumps({"rejected": [{"key": "book5", "reason": "unrecoverable", "rejected_at": "2026-08-16"}]}),
                encoding="utf-8",
            )
            corpus.eval_tier_ids_path().write_text(json.dumps(["book4"]), encoding="utf-8")
            corpus.evaluation_dir().mkdir(parents=True, exist_ok=True)
            corpus.evaluation_json_path("book2").write_text(
                json.dumps({"entries": [], "verified": False, "source": "crossref"}), encoding="utf-8"
            )

            table = build_status_table()

            self.assertIn("| Manifest books | 4 |", table)
            self.assertIn("| Books with ground truth | 2 |", table)
            self.assertIn("| — via two-model gate (`bulk_gate`) | 1 |", table)
            self.assertIn("| — via arbitration (`agent_arbitration`) | 1 |", table)
            self.assertIn("| Books with a `model-a` reading | 3 |", table)
            self.assertIn("| Books with a `model-b` reading | 1 |", table)
            self.assertIn("| Books awaiting arbitration | 1 |", table)
            self.assertIn("| Books permanently rejected (unrecoverable) | 1 |", table)
            self.assertIn("| Held-out eval-tier sample | 1 |", table)
            self.assertIn("| Crossref evaluation-corpus entries | 1 |", table)

    def test_omits_a_model_below_the_minimum_reading_threshold(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(corpus, "CORPUS_DIR", Path(tmp) / "corpus"),
            patch.object(corpus_status, "_MIN_MODEL_READINGS", 3),
        ):
            _write_manifest(["book1.pdf", "book2.pdf", "book3.pdf"])
            for i, key in enumerate(["book1", "book2", "book3"]):
                write_cached_llm_entries(corpus.llm_cache_dir(), key, "frequent-model", [_entry("X", 1)])
            write_cached_llm_entries(corpus.llm_cache_dir(), "book1", "rare-model", [_entry("X", 1)])

            table = build_status_table()

            self.assertIn("| Books with a `frequent-model` reading | 3 |", table)
            self.assertNotIn("rare-model", table)

    def test_handles_an_empty_corpus_without_crashing(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp) / "corpus"):
            _write_manifest([])

            table = build_status_table()

            self.assertIn("| Manifest books | 0 |", table)
            self.assertIn("| Books with ground truth | 0 |", table)


class TestUpdateReadme(unittest.TestCase):
    def test_replaces_text_between_markers(self):
        with tempfile.TemporaryDirectory() as tmp:
            readme_path = Path(tmp) / "README.md"
            readme_path.write_text(f"# Title\n\n{_MARKER_START}\nstale\n{_MARKER_END}\n\n## Next\n", encoding="utf-8")

            changed = update_readme("fresh table", readme_path)

            self.assertTrue(changed)
            text = readme_path.read_text(encoding="utf-8")
            self.assertIn("fresh table", text)
            self.assertNotIn("stale", text)
            self.assertIn("## Next", text)

    def test_returns_false_when_content_is_already_current(self):
        with tempfile.TemporaryDirectory() as tmp:
            readme_path = Path(tmp) / "README.md"
            readme_path.write_text(f"{_MARKER_START}\n\nfresh table\n\n{_MARKER_END}\n", encoding="utf-8")

            self.assertFalse(update_readme("fresh table", readme_path))

    def test_raises_when_markers_are_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            readme_path = Path(tmp) / "README.md"
            readme_path.write_text("# Title\n\nno markers here\n", encoding="utf-8")

            with self.assertRaises(SystemExit):
                update_readme("fresh table", readme_path)


if __name__ == "__main__":
    unittest.main()
