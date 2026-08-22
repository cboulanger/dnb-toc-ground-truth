"""Unit tests for dnb_toc_ground_truth.evaluation_site -- builds the
static GitHub Pages site presenting crossref-evaluation scores. See
cli/generate_evaluation_site.py for the CLI wrapper and
.github/workflows/pages.yml for how the site is deployed."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dnb_toc_ground_truth import corpus, vision
from dnb_toc_ground_truth.crossref_evaluation import BookMetrics
from dnb_toc_ground_truth.evaluation_site import (
    CorpusData,
    SourceScores,
    collect_corpus_data,
    render_corpus_html,
    render_index_html,
    write_site,
)
from dnb_toc_ground_truth.toc_entry import TocEntry


class TestCollectCorpusData(unittest.TestCase):
    def test_gathers_ground_truth_and_every_cached_model(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            corpus.manifest_path().write_text(
                json.dumps({"toc_only": True, "books": [
                    {"filename": "9783899718188.pdf", "doi": "10.1/x", "title": "Some Book"},
                ]}),
                encoding="utf-8",
            )
            corpus.ground_truth_dir().mkdir(parents=True, exist_ok=True)
            corpus.expected_json_path("9783899718188").write_text(
                json.dumps({"entries": [
                    {"title": "1. Introduction", "authors": [], "printed_page_number": "1", "skip": False},
                ]}),
                encoding="utf-8",
            )
            corpus.evaluation_dir().mkdir(parents=True, exist_ok=True)
            corpus.evaluation_json_path("9783899718188").write_text(
                json.dumps({
                    "entries": [{"title": "Introduction", "authors": [], "printed_page_number": "1", "skip": False}],
                    "source": "crossref", "fetched_at": "",
                }),
                encoding="utf-8",
            )
            vision.write_cached_llm_entries(corpus.llm_cache_dir(), "9783899718188", "some/model", [
                TocEntry(title="Introduction", printed_page_number="1", source_page_index=-1, skip=False),
            ])

            data = collect_corpus_data()

            self.assertEqual(data.titles["9783899718188"], "Some Book")
            self.assertEqual(len(data.sources), 2)
            gt_source, model_source = data.sources
            self.assertTrue(gt_source.is_ground_truth)
            self.assertEqual(gt_source.label, "Ground truth")
            self.assertEqual(len(gt_source.results), 1)
            self.assertFalse(model_source.is_ground_truth)
            self.assertEqual(model_source.label, "some__model")
            self.assertEqual(len(model_source.results), 1)


class TestRenderIndexHtml(unittest.TestCase):
    def test_lists_each_corpus_with_a_link(self):
        html = render_index_html(["pilot"])
        self.assertIn('href="pilot.html"', html)
        self.assertIn(">pilot<", html)

    def test_mentions_sample_bias_and_publisher_data_caveats(self):
        html = render_index_html(["pilot"])
        lowered = html.lower()
        self.assertIn("small", lowered)
        self.assertIn("publisher", lowered)
        self.assertIn("macro-averaged", lowered)


class TestRenderCorpusHtml(unittest.TestCase):
    def _sample_data(self) -> CorpusData:
        return CorpusData(
            name="pilot",
            titles={"9783899718188": "Some Book"},
            sources=[
                SourceScores(
                    label="Ground truth", is_ground_truth=True,
                    results=[BookMetrics(key="9783899718188", tp=1, fp=0, fn=0, precision=1.0, recall=1.0, f1=1.0)],
                    uncovered_count=0,
                ),
                SourceScores(
                    label="some__model", is_ground_truth=False,
                    results=[BookMetrics(key="9783899718188", tp=1, fp=0, fn=1, precision=1.0, recall=0.5, f1=2 / 3)],
                    uncovered_count=3,
                ),
            ],
        )

    def test_renders_ground_truth_and_model_sections_with_scores(self):
        html = render_corpus_html(self._sample_data())
        self.assertIn("Some Book", html)
        self.assertIn("Ground truth", html)
        self.assertIn("some__model", html)
        self.assertIn("100%", html)
        self.assertIn("50%", html)
        self.assertIn("Mean", html)
        self.assertIn("3 crossref-sample book(s)", html)

    def test_ground_truth_section_is_visually_distinguished(self):
        # The design calls for ground truth to be bolded/highlighted --
        # rendered via a dedicated CSS class rather than inline styling
        # so the styling lives once in the shared stylesheet.
        html = render_corpus_html(self._sample_data())
        self.assertIn('class="ground-truth"', html)

    def test_source_with_no_results_still_renders_without_crashing(self):
        data = CorpusData(
            name="pilot", titles={},
            sources=[SourceScores(label="Ground truth", is_ground_truth=True, results=[], uncovered_count=5)],
        )
        html = render_corpus_html(data)
        self.assertIn("No crossref-sample books had data for this source.", html)


def _init_corpus(root: Path, name: str, books: list[dict]) -> None:
    corpus_dir = root / name
    corpus_dir.mkdir(parents=True, exist_ok=True)
    (corpus_dir / "manifest.json").write_text(json.dumps({"toc_only": True, "books": books}), encoding="utf-8")


class TestWriteSite(unittest.TestCase):
    def test_writes_index_and_one_page_per_corpus(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _init_corpus(root, "pilot", [])
            _init_corpus(root, "other", [])

            with patch.object(corpus, "_CORPUS_ROOT", root), patch.object(corpus, "CORPUS_DIR", root / "pilot"):
                with tempfile.TemporaryDirectory() as out:
                    output_dir = Path(out) / "site"
                    write_site(output_dir)

                    self.assertTrue((output_dir / "index.html").exists())
                    self.assertTrue((output_dir / "pilot.html").exists())
                    self.assertTrue((output_dir / "other.html").exists())
                    self.assertTrue((output_dir / ".nojekyll").exists())
                    index_html = (output_dir / "index.html").read_text(encoding="utf-8")
                    self.assertIn('href="pilot.html"', index_html)
                    self.assertIn('href="other.html"', index_html)

    def test_restores_the_previously_selected_corpus_afterward(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _init_corpus(root, "pilot", [])
            _init_corpus(root, "other", [])

            with patch.object(corpus, "_CORPUS_ROOT", root), patch.object(corpus, "CORPUS_DIR", root / "other"):
                with tempfile.TemporaryDirectory() as out:
                    write_site(Path(out) / "site")
                    self.assertEqual(corpus.CORPUS_DIR, root / "other")

    def test_each_corpus_page_reflects_that_corpus_own_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _init_corpus(root, "pilot", [{"filename": "9783899718188.pdf", "doi": None, "title": "Pilot Book"}])
            _init_corpus(root, "other", [{"filename": "1111111111.pdf", "doi": None, "title": "Other Book"}])

            with patch.object(corpus, "_CORPUS_ROOT", root), patch.object(corpus, "CORPUS_DIR", root / "pilot"):
                with tempfile.TemporaryDirectory() as out:
                    output_dir = Path(out) / "site"
                    write_site(output_dir)

                    pilot_html = (output_dir / "pilot.html").read_text(encoding="utf-8")
                    other_html = (output_dir / "other.html").read_text(encoding="utf-8")
                    self.assertIn(">pilot<", pilot_html)
                    self.assertIn(">other<", other_html)


if __name__ == "__main__":
    unittest.main()
