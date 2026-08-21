"""Unit tests for inference.py -- endpoints-file parsing (both formats)
and model-id resolution. No real network calls; AsyncOpenAI client
construction is exercised for real (it doesn't connect until a call is
made) so tests can assert on endpoint.client.base_url."""

import json
import tempfile
import unittest
from pathlib import Path

from dnb_toc_ground_truth.inference import (
    ModelEndpoint, load_config, load_endpoint_entries, resolve_model_endpoints,
)


def _write(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


class TestLoadEndpointEntriesJson(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_parses_model_field_directly(self):
        path = _write(self.tmp_path, ".endpoints", json.dumps([
            {"url": "https://example.invalid/session-a", "key": "secret-a", "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct"},
        ]))
        entries = load_endpoint_entries(path)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].model, "Qwen/Qwen3-Omni-30B-A3B-Instruct")
        self.assertEqual(entries[0].base_url, "https://example.invalid/session-a/v1")

    def test_parses_model_from_framework_args_when_model_key_absent(self):
        path = _write(self.tmp_path, ".endpoints", json.dumps([
            {
                "url": "https://example.invalid/session-b/v1", "key": "secret-b",
                "framework_args": "--model=mistralai/Mistral-Small-3.2-24B-Instruct-2506 --tensor-parallel-size=2",
            },
        ]))
        entries = load_endpoint_entries(path)
        self.assertEqual(entries[0].model, "mistralai/Mistral-Small-3.2-24B-Instruct-2506")

    def test_skips_entry_missing_required_fields(self):
        path = _write(self.tmp_path, ".endpoints", json.dumps([
            {"url": "https://example.invalid/x", "key": "secret"},  # no model, no framework_args
        ]))
        with self.assertRaises(ValueError):
            load_endpoint_entries(path)

    def test_carries_status_field(self):
        path = _write(self.tmp_path, ".endpoints", json.dumps([
            {"url": "https://example.invalid/x", "key": "secret", "model": "model-a", "status": "Running"},
        ]))
        entries = load_endpoint_entries(path)
        self.assertEqual(entries[0].status, "Running")


class TestLoadEndpointEntriesPlainText(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_parses_pasted_session_table(self):
        path = _write(self.tmp_path, ".endpoints", (
            "framework\tvLLM\n"
            "framework_args\t--model=Qwen/Qwen3-Omni-30B-A3B-Instruct --tensor-parallel-size=2\n"
            "key\tsecret-a\n"
            "url\thttps://llm.mpcdf.mpg.de/abc123\n"
        ))
        entries = load_endpoint_entries(path)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].model, "Qwen/Qwen3-Omni-30B-A3B-Instruct")
        self.assertEqual(entries[0].base_url, "https://llm.mpcdf.mpg.de/abc123/v1")
        self.assertEqual(entries[0].status, "")

    def test_parses_multiple_blocks_separated_by_blank_line(self):
        path = _write(self.tmp_path, ".endpoints", (
            "framework_args\t--model=model-a\nkey\tkey-a\nurl\thttps://x.invalid/a/v1\n"
            "\n"
            "framework_args\t--model=model-b\nkey\tkey-b\nurl\thttps://x.invalid/b/v1\n"
        ))
        entries = load_endpoint_entries(path)
        self.assertEqual([e.model for e in entries], ["model-a", "model-b"])

    def test_raises_on_missing_file(self):
        with self.assertRaises(ValueError):
            load_endpoint_entries(self.tmp_path / "nonexistent")


class TestResolveModelEndpoints(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _entries(self, rows):
        path = _write(self.tmp_path, ".endpoints", json.dumps(rows))
        return load_endpoint_entries(path)

    def test_resolves_exact_model_match(self):
        entries = self._entries([{"url": "https://x.invalid/a", "key": "k", "model": "model-a"}])
        resolved = resolve_model_endpoints(["model-a"], "vision", entries)
        self.assertEqual(len(resolved), 1)
        self.assertIsInstance(resolved[0], ModelEndpoint)
        self.assertEqual(resolved[0].model_id, "model-a")
        self.assertEqual(resolved[0].kind, "vision")

    def test_raises_when_model_not_found(self):
        entries = self._entries([{"url": "https://x.invalid/a", "key": "k", "model": "model-a"}])
        with self.assertRaises(ValueError):
            resolve_model_endpoints(["model-missing"], "vision", entries)

    def test_disambiguates_duplicate_model_by_running_status(self):
        entries = self._entries([
            {"url": "https://x.invalid/a", "key": "k1", "model": "model-a", "status": "Stopped"},
            {"url": "https://x.invalid/b", "key": "k2", "model": "model-a", "status": "Running"},
        ])
        resolved = resolve_model_endpoints(["model-a"], "vision", entries)
        # AsyncOpenAI normalizes base_url with a trailing slash, so assert
        # containment rather than exact equality (matches
        # chapter-segmentation's own inference_endpoints test convention).
        self.assertIn("https://x.invalid/b/v1", str(resolved[0].client.base_url))

    def test_raises_on_ambiguous_duplicate_with_no_running_tiebreak(self):
        entries = self._entries([
            {"url": "https://x.invalid/a", "key": "k1", "model": "model-a"},
            {"url": "https://x.invalid/b", "key": "k2", "model": "model-a"},
        ])
        with self.assertRaises(ValueError):
            resolve_model_endpoints(["model-a"], "vision", entries)

    def test_resolves_same_model_id_twice_for_two_independent_reads(self):
        entries = self._entries([{"url": "https://x.invalid/a", "key": "k", "model": "model-a"}])
        resolved = resolve_model_endpoints(["model-a", "model-a"], "vision", entries)
        self.assertEqual(len(resolved), 2)


class TestLoadConfig(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_returns_empty_dict_when_missing(self):
        self.assertEqual(load_config(self.tmp_path / "nonexistent"), {})

    def test_parses_json_config(self):
        path = _write(self.tmp_path, ".config", json.dumps({"use_vision": ["model-a", "model-b"], "concurrency": 2}))
        self.assertEqual(load_config(path), {"use_vision": ["model-a", "model-b"], "concurrency": 2})


if __name__ == "__main__":
    unittest.main()
