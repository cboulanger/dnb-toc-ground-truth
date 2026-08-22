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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cli"))

from evaluate_crossref import _run_backfill


def _args(**overrides) -> argparse.Namespace:
    defaults = dict(model=None, endpoints_file=None)
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


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


if __name__ == "__main__":
    unittest.main()
