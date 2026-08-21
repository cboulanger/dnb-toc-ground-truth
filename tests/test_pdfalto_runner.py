"""Unit tests for pdfalto_runner.py's pure resolve_pdfalto_binary logic --
no real pdfalto binary needed (ensure_alto_xml shells out and is exercised
manually, matching chapter-segmentation's own convention for this
module)."""

import os
import unittest
from unittest.mock import patch

from dnb_toc_ground_truth.pdfalto_runner import resolve_pdfalto_binary


class TestResolvePdfaltoBinary(unittest.TestCase):
    def test_explicit_cli_arg_wins(self):
        with patch.dict(os.environ, {"PDFALTO_BIN": "/env/pdfalto"}, clear=False):
            self.assertEqual(resolve_pdfalto_binary("/explicit/pdfalto"), "/explicit/pdfalto")

    def test_falls_back_to_env_var(self):
        with patch.dict(os.environ, {"PDFALTO_BIN": "/env/pdfalto"}, clear=False):
            self.assertEqual(resolve_pdfalto_binary(None), "/env/pdfalto")

    def test_falls_back_to_bare_command(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(resolve_pdfalto_binary(None), "pdfalto")


if __name__ == "__main__":
    unittest.main()
