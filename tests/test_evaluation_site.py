"""Unit tests for dnb_toc_ground_truth.evaluation_site -- builds the
static GitHub Pages site presenting crossref-evaluation scores. See
cli/generate_evaluation_site.py for the CLI wrapper and
.github/workflows/pages.yml for how the site is deployed."""

import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dnb_toc_ground_truth import corpus, model_agreement, vision
from dnb_toc_ground_truth.crossref_evaluation import BookMetrics
from dnb_toc_ground_truth.evaluation_site import (
    CorpusData,
    ModelComparisonData,
    SourceScores,
    _model_comparison_filename,
    collect_corpus_data,
    collect_model_comparison_data,
    render_corpus_html,
    render_index_html,
    render_model_comparison_html,
    write_site,
)
from dnb_toc_ground_truth.model_agreement import (
    ModelGroundTruthMetrics,
    PairAgreement,
    PairCandidateScore,
)
from dnb_toc_ground_truth.toc_entry import TocEntry


class TestCollectCorpusData(unittest.TestCase):
    def test_gathers_ground_truth_and_every_cached_model(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            corpus.manifest_path().write_text(
                json.dumps({"toc_only": True, "books": [
                    {
                        "filename": "9783899718188.pdf", "doi": "10.1/x", "title": "Some Book",
                        "toc_download_url": "https://example.org/toc.pdf",
                    },
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
            self.assertEqual(data.toc_urls["9783899718188"], "https://example.org/toc.pdf")
            self.assertEqual(len(data.sources), 2)
            gt_source, model_source = data.sources
            self.assertTrue(gt_source.is_ground_truth)
            self.assertEqual(gt_source.label, "Ground truth")
            self.assertIsNone(gt_source.model)
            self.assertEqual(len(gt_source.results), 1)
            self.assertFalse(model_source.is_ground_truth)
            self.assertEqual(model_source.label, "some__model")
            self.assertEqual(model_source.model, "some__model")
            self.assertEqual(len(model_source.results), 1)


class TestRenderIndexHtml(unittest.TestCase):
    def test_lists_each_corpus_with_a_link(self):
        html = render_index_html(["pilot"])
        self.assertIn('href="pilot.html"', html)
        self.assertIn(">pilot<", html)
        self.assertIn('href="pilot-model-comparison.html"', html)

    def test_corpus_links_stay_in_the_same_tab(self):
        html = render_index_html(["pilot"])
        self.assertIn('<a href="pilot.html">pilot</a>', html)

    def test_crossref_link_opens_in_a_new_tab(self):
        html = render_index_html(["pilot"])
        self.assertIn(
            '<a href="https://www.crossref.org/" target="_blank" rel="noopener noreferrer">Crossref</a>', html,
        )

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
            toc_urls={"9783899718188": "https://example.org/toc.pdf"},
            sources=[
                SourceScores(
                    label="Ground truth", is_ground_truth=True,
                    results=[BookMetrics(key="9783899718188", tp=1, fp=0, fn=0, precision=1.0, recall=1.0, f1=1.0)],
                    uncovered_count=0,
                ),
                SourceScores(
                    label="some__model", is_ground_truth=False,
                    results=[BookMetrics(key="9783899718188", tp=1, fp=0, fn=1, precision=1.0, recall=0.5, f1=2 / 3)],
                    uncovered_count=3, model="some__model",
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
            name="pilot", titles={}, toc_urls={},
            sources=[SourceScores(label="Ground truth", is_ground_truth=True, results=[], uncovered_count=5)],
        )
        html = render_corpus_html(data)
        self.assertIn("No crossref-sample books had data for this source.", html)

    def test_book_rows_are_sorted_by_descending_f1(self):
        data = CorpusData(
            name="pilot",
            titles={"a": "Low Book", "b": "High Book", "c": "Mid Book"},
            toc_urls={},
            sources=[SourceScores(
                label="Ground truth", is_ground_truth=True,
                results=[
                    BookMetrics(key="a", tp=1, fp=9, fn=9, precision=0.1, recall=0.1, f1=0.10),
                    BookMetrics(key="b", tp=9, fp=1, fn=1, precision=0.9, recall=0.9, f1=0.90),
                    BookMetrics(key="c", tp=5, fp=5, fn=5, precision=0.5, recall=0.5, f1=0.50),
                ],
                uncovered_count=0,
            )],
        )
        html = render_corpus_html(data)
        first = html.index("High Book")
        second = html.index("Mid Book")
        third = html.index("Low Book")
        self.assertLess(first, second)
        self.assertLess(second, third)

    def test_book_title_links_to_its_toc_download_url(self):
        html = render_corpus_html(self._sample_data())
        self.assertIn(
            '<td><a href="https://example.org/toc.pdf" target="_blank" rel="noopener noreferrer">Some Book</a></td>',
            html,
        )

    def test_book_without_a_toc_url_renders_plain(self):
        data = CorpusData(
            name="pilot", titles={"1111111111": "No URL Book"}, toc_urls={},
            sources=[SourceScores(
                label="Ground truth", is_ground_truth=True,
                results=[BookMetrics(key="1111111111", tp=1, fp=0, fn=0, precision=1.0, recall=1.0, f1=1.0)],
                uncovered_count=0,
            )],
        )
        html = render_corpus_html(data)
        # Not wrapped in a link -- if it were, the tag would immediately
        # follow "<td>" instead of the title text itself.
        self.assertIn("<td>No URL Book</td>", html)

    def test_ground_truth_row_links_to_the_expected_json_blob_url(self):
        html = render_corpus_html(self._sample_data())
        expected = corpus.expected_json_path("9783899718188").relative_to(corpus.repo_root())
        self.assertIn(
            f'<a href="https://github.com/cboulanger/dnb-toc-ground-truth/blob/main/{expected}" '
            'target="_blank" rel="noopener noreferrer">JSON</a>', html,
        )

    def test_model_row_links_to_its_llm_cache_blob_url(self):
        html = render_corpus_html(self._sample_data())
        cache_path = vision.cache_path(corpus.llm_cache_dir(), "9783899718188", "some__model")
        expected = cache_path.relative_to(corpus.repo_root())
        self.assertIn(
            f'<a href="https://github.com/cboulanger/dnb-toc-ground-truth/blob/main/{expected}" '
            'target="_blank" rel="noopener noreferrer">JSON</a>', html,
        )

    def test_every_off_site_link_opens_in_a_new_tab(self):
        # Standing invariant for every link that takes the reader off
        # this site's own pages (TOC scan host, GitHub blob) -- any
        # future one that skips the shared _link() helper's new_tab
        # default should fail here rather than silently opening in the
        # same tab.
        html = render_corpus_html(self._sample_data())
        anchors = re.findall(r"<a\b[^>]*>", html)
        off_site = [
            a for a in anchors
            if 'href="index.html"' not in a
            and 'href="pilot.html"' not in a
            and 'href="pilot-model-comparison.html"' not in a
        ]
        self.assertTrue(off_site)
        for anchor in off_site:
            self.assertIn('target="_blank"', anchor)
            self.assertIn('rel="noopener noreferrer"', anchor)

    def test_back_to_corpora_link_stays_in_the_same_tab(self):
        html = render_corpus_html(self._sample_data())
        self.assertIn('<a href="index.html">&larr; Corpora</a>', html)

    def test_metric_headers_carry_explanatory_tooltips(self):
        html = render_corpus_html(self._sample_data())
        for label in ("Precision", "Recall", "F1", "TP", "FP", "FN"):
            self.assertRegex(html, rf'<th[^>]*title="[^"]+"[^>]*>{label}</th>')

    def test_overview_table_has_one_mean_row_per_source_with_book_count(self):
        html = render_corpus_html(self._sample_data())
        self.assertIn("Overview", html)
        self.assertIn("Ground truth (1 book(s))", html)
        self.assertIn("some__model (1 book(s))", html)

    def test_overview_table_appears_before_the_per_source_sections(self):
        html = render_corpus_html(self._sample_data())
        self.assertLess(html.index("Overview"), html.index('<section class="ground-truth"'))

    def test_overview_table_omits_sources_with_no_results(self):
        data = CorpusData(
            name="pilot", titles={}, toc_urls={},
            sources=[
                SourceScores(label="Ground truth", is_ground_truth=True, results=[], uncovered_count=5),
                SourceScores(
                    label="some__model", is_ground_truth=False,
                    results=[BookMetrics(key="a", tp=1, fp=0, fn=0, precision=1.0, recall=1.0, f1=1.0)],
                    uncovered_count=0, model="some__model",
                ),
            ],
        )
        html = render_corpus_html(data)
        self.assertNotIn("Ground truth (0 book(s))", html)
        self.assertIn("some__model (1 book(s))", html)


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


class TestWriteSiteModelComparison(unittest.TestCase):
    def test_writes_a_model_comparison_page_per_corpus(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _init_corpus(root, "pilot", [])
            with patch.object(corpus, "_CORPUS_ROOT", root), patch.object(corpus, "CORPUS_DIR", root / "pilot"):
                with tempfile.TemporaryDirectory() as out:
                    output_dir = Path(out) / "site"
                    write_site(output_dir)
                    self.assertTrue((output_dir / "pilot-model-comparison.html").exists())


class TestModelComparisonFilenameConsistency(unittest.TestCase):
    """render_index_html, render_corpus_html, and write_site each build a
    corpus's model-comparison filename independently -- this pins all
    three to _model_comparison_filename() so a future edit to one call
    site's convention (e.g. renaming the suffix) fails here instead of
    silently producing a broken link."""

    def test_index_link_matches_the_shared_filename_convention(self):
        html = render_index_html(["pilot"])
        self.assertIn(f'href="{_model_comparison_filename("pilot")}"', html)

    def test_corpus_page_link_matches_the_shared_filename_convention(self):
        data = CorpusData(
            name="pilot", titles={}, toc_urls={},
            sources=[SourceScores(label="Ground truth", is_ground_truth=True, results=[], uncovered_count=0)],
        )
        html = render_corpus_html(data)
        self.assertIn(f'href="{_model_comparison_filename("pilot")}"', html)

    def test_write_site_writes_the_file_the_shared_filename_convention_predicts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _init_corpus(root, "pilot", [])
            with patch.object(corpus, "_CORPUS_ROOT", root), patch.object(corpus, "CORPUS_DIR", root / "pilot"):
                with tempfile.TemporaryDirectory() as out:
                    output_dir = Path(out) / "site"
                    write_site(output_dir)
                    self.assertTrue((output_dir / _model_comparison_filename("pilot")).exists())


class TestCollectModelComparisonData(unittest.TestCase):
    def test_gathers_agreement_gt_metrics_and_crossref_for_every_cached_model(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp) / "pilot"), \
                patch.object(model_agreement, "_MIN_MODEL_READINGS", 1):
            corpus.corpus_dir().mkdir(parents=True, exist_ok=True)
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

            data = collect_model_comparison_data()

            self.assertEqual(data.name, "pilot")
            self.assertEqual(data.models, ["model__a", "model__b"])
            self.assertEqual(len(data.agreements), 1)
            self.assertEqual(len(data.gt_metrics), 2)
            self.assertIn("model__a", data.crossref_results)
            self.assertEqual(len(data.scored_pairs), 1)
            self.assertEqual(data.unscored_pairs, [])

    def test_no_cached_models_yields_empty_data_without_crashing(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            corpus.manifest_path().write_text(json.dumps({"toc_only": True, "books": []}), encoding="utf-8")
            data = collect_model_comparison_data()
            self.assertEqual(data.models, [])
            self.assertEqual(data.agreements, [])


class TestRenderModelComparisonHtml(unittest.TestCase):
    def _sample_data(self) -> ModelComparisonData:
        return ModelComparisonData(
            name="pilot",
            models=["model__a", "model__b"],
            agreements=[PairAgreement(model_a="model__a", model_b="model__b", mean_agreement=0.9, n_books=5)],
            gt_metrics=[
                ModelGroundTruthMetrics(model="model__a", precision=0.8, recall=0.8, f1=0.8, n_books=10),
                ModelGroundTruthMetrics(model="model__b", precision=0.7, recall=0.7, f1=0.7, n_books=8),
            ],
            crossref_results={"model__a": ([], []), "model__b": ([], [])},
            scored_pairs=[PairCandidateScore(
                model_a="model__a", model_b="model__b", f1_a=0.8, f1_b=0.7,
                observed_agreement=0.9, expected_agreement=0.62, kappa=0.74, score=-0.04, n_books=5,
            )],
            unscored_pairs=[],
        )

    def test_renders_all_three_sections(self):
        html = render_model_comparison_html(self._sample_data())
        self.assertIn("Pairwise agreement", html)
        self.assertIn("Per-model accuracy", html)
        self.assertIn("Candidate pair ranking", html)
        self.assertIn("model__a", html)
        self.assertIn("model__b", html)

    def test_heatmap_cell_shows_rate_and_n(self):
        html = render_model_comparison_html(self._sample_data())
        self.assertIn("90% (n=5)", html)

    def test_no_models_renders_a_placeholder_without_crashing(self):
        data = ModelComparisonData(
            name="pilot", models=[], agreements=[], gt_metrics=[], crossref_results={},
            scored_pairs=[], unscored_pairs=[],
        )
        html = render_model_comparison_html(data)
        self.assertIn("No cached model readings found", html)

    def test_unscored_pairs_are_listed_separately(self):
        data = ModelComparisonData(
            name="pilot", models=["model__a", "model__b"],
            agreements=[PairAgreement(model_a="model__a", model_b="model__b", mean_agreement=0.9, n_books=5)],
            gt_metrics=[], crossref_results={}, scored_pairs=[],
            unscored_pairs=[("model__a", "model__b")],
        )
        html = render_model_comparison_html(data)
        self.assertIn("Unscored pairs", html)
        self.assertIn("model__a + model__b", html)

    def test_back_links_to_corpora_and_corpus_page(self):
        html = render_model_comparison_html(self._sample_data())
        self.assertIn('<a href="index.html">&larr; Corpora</a>', html)
        self.assertIn('<a href="pilot.html">&larr; pilot</a>', html)

    def test_model_with_no_arbitration_gt_coverage_sorts_last_and_shows_dashes(self):
        data = ModelComparisonData(
            name="pilot", models=["model__a", "model__b"],
            agreements=[PairAgreement(model_a="model__a", model_b="model__b", mean_agreement=0.9, n_books=5)],
            gt_metrics=[
                ModelGroundTruthMetrics(model="model__a", precision=0.8, recall=0.8, f1=0.8, n_books=10),
            ],  # model__b has NO arbitration-GT coverage at all
            crossref_results={"model__a": ([], []), "model__b": ([], [])},
            scored_pairs=[], unscored_pairs=[("model__a", "model__b")],
        )
        html = render_model_comparison_html(data)
        table = html[html.index("Per-model accuracy"):html.index("Candidate pair ranking")]
        # model__a (has coverage) must appear before model__b (no coverage) in the accuracy table.
        self.assertLess(table.index("model__a"), table.index("model__b"))
        row_a = table[table.index("<tr><td>model__a"):table.index("<tr><td>model__b")]
        row_b = table[table.index("<tr><td>model__b"):]
        # model__a's real F1 shows as a proportional bar.
        self.assertIn('<div style="width:80%; background:#2e7d32; height:0.5rem;"></div>', row_a)
        # model__b's row shows dashes for its missing arbitration-GT columns...
        self.assertIn('<td class="num">-</td><td class="num">-</td><td class="num">-</td><td class="num">-</td>', row_b)
        # ...and its bar cell is empty, not a fabricated 0%-width div.
        self.assertIn("<td></td>", row_b)
        self.assertNotIn("width:0%", row_b)


if __name__ == "__main__":
    unittest.main()
