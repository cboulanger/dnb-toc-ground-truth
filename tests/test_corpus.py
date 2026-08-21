import json
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
