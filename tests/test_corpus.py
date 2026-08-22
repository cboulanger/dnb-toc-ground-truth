import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dnb_toc_ground_truth import corpus


class TestCorpusPaths(unittest.TestCase):
    def test_pdf_path_lives_under_pdf_subdir(self):
        self.assertEqual(corpus.pdf_path("9783899718188").name, "9783899718188.pdf")
        self.assertEqual(corpus.pdf_path("9783899718188").parent, corpus.pdf_dir())

    def test_expected_json_path_lives_under_ground_truth_subdir(self):
        path = corpus.expected_json_path("9783899718188")
        self.assertEqual(path.name, "9783899718188.expected.json")
        self.assertEqual(path.parent, corpus.ground_truth_dir())

    def test_crossref_cache_dir(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            self.assertEqual(corpus.crossref_cache_dir(), Path(tmp) / ".crossref-cache")

    def test_evaluation_dir(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            self.assertEqual(corpus.evaluation_dir(), Path(tmp) / "evaluation")

    def test_evaluation_json_path(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            path = corpus.evaluation_json_path("9783899718188")
            self.assertEqual(path.name, "9783899718188.expected.json")
            self.assertEqual(path.parent, corpus.evaluation_dir())


class TestListCorpora(unittest.TestCase):
    def test_discovers_directories_with_a_manifest_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pilot").mkdir()
            (root / "pilot" / "manifest.json").write_text("{}", encoding="utf-8")
            (root / "other").mkdir()
            (root / "other" / "manifest.json").write_text("{}", encoding="utf-8")
            # A directory with no manifest.json isn't a corpus -- must be ignored.
            (root / "not-a-corpus").mkdir()
            with patch.object(corpus, "_CORPUS_ROOT", root):
                self.assertEqual(corpus.list_corpora(), ["other", "pilot"])

    def test_empty_when_corpus_root_does_not_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "does-not-exist"
            with patch.object(corpus, "_CORPUS_ROOT", missing):
                self.assertEqual(corpus.list_corpora(), [])


class TestSetCorpus(unittest.TestCase):
    def test_switches_corpus_dir_to_the_named_corpus(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "other").mkdir()
            (root / "other" / "manifest.json").write_text("{}", encoding="utf-8")
            original = corpus.CORPUS_DIR
            try:
                with patch.object(corpus, "_CORPUS_ROOT", root):
                    corpus.set_corpus("other")
                    self.assertEqual(corpus.CORPUS_DIR, root / "other")
            finally:
                corpus.CORPUS_DIR = original

    def test_raises_with_available_corpora_for_an_unknown_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pilot").mkdir()
            (root / "pilot" / "manifest.json").write_text("{}", encoding="utf-8")
            with patch.object(corpus, "_CORPUS_ROOT", root):
                with self.assertRaises(ValueError) as ctx:
                    corpus.set_corpus("nonexistent")
                self.assertIn("nonexistent", str(ctx.exception))
                self.assertIn("pilot", str(ctx.exception))


class TestManifestKey(unittest.TestCase):
    def test_strips_pdf_extension(self):
        self.assertEqual(corpus.manifest_key({"filename": "9783899718188.pdf"}), "9783899718188")


class TestLoadManifestBooks(unittest.TestCase):
    def test_reads_books_list_from_manifest_json(self):
        with patch.object(corpus, "manifest_path", return_value=Path("/tmp/does-not-matter")):
            with patch.object(Path, "read_text", return_value=json.dumps({"books": [{"filename": "a.pdf"}]})):
                self.assertEqual(corpus.load_manifest_books(), [{"filename": "a.pdf"}])


if __name__ == "__main__":
    unittest.main()
