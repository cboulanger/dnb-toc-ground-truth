"""Unit tests for cli/fetch_corpus.py's pure logic (record filtering,
field extraction, streaming JSON-Lines decode) against mocked httpx
responses and an in-memory gzip stream -- no live network. The real
network-calling main()/_run_isbns_file()/_run_from_dump() orchestration is
exercised manually, matching the project's existing convention of no
pytest coverage for its own network-calling entry point."""

import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import httpx

from dnb_toc_ground_truth import corpus, crossref

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cli"))

from fetch_corpus import (
    _ChunkStreamReader,
    _acquire_record,
    _append_book,
    _ensure_manifest_shell,
    _iter_dump_records_from_chunks,
    _load_existing_keys,
    _read_isbns_file,
    _record_api_url,
    _record_key,
    _record_language,
    _record_matches,
    _scan_and_acquire,
    _search_by_isbn,
    _toc_download_url,
    manifest_entry_from_record,
)

# Trimmed to the fields this script actually reads, sourced from a real
# lobid-resources record (isbn:9783899718188, confirmed live 2026-08-14 --
# see the plan's header).
_SAMPLE_RECORD = {
    "id": "http://lobid.org/resources/990183806670206441#!",
    "type": ["BibliographicResource", "EditedVolume", "Book"],
    "title": "Systemtheorie in den Fachwissenschaften",
    "isbn": ["9783899718188", "3899718186"],
    "language": [{"id": "http://id.loc.gov/vocabulary/iso639-2/ger", "label": "Deutsch"}],
    "tableOfContents": [
        {
            "label": "Inhaltsverzeichnis",
            "id": "https://digitale-objekte.hbz-nrw.de/storage/2011/03/19/file_10/4104671.pdf",
        }
    ],
}


def _json_response(payload: dict) -> Mock:
    response = Mock()
    response.status_code = 200
    response.json.return_value = payload
    response.raise_for_status = Mock()
    return response


class TestRecordMatches(unittest.TestCase):
    def test_matches_book_with_toc(self):
        self.assertTrue(_record_matches(_SAMPLE_RECORD))

    def test_rejects_wrong_type(self):
        record = {**_SAMPLE_RECORD, "type": ["BibliographicResource", "Article"]}
        self.assertFalse(_record_matches(record))

    def test_rejects_missing_toc(self):
        record = {**_SAMPLE_RECORD, "tableOfContents": []}
        self.assertFalse(_record_matches(record))

    def test_rejects_absent_toc_key(self):
        record = {k: v for k, v in _SAMPLE_RECORD.items() if k != "tableOfContents"}
        self.assertFalse(_record_matches(record))

    def test_rejects_plain_book_without_edited_volume(self):
        # Confirmed live 2026-08-15 (isbn:9783868674095): a Lehrbuch
        # typed just ["BibliographicResource", "Book"] -- no
        # "EditedVolume" -- must NOT match even though it has a TOC and
        # "Book" is technically in its type list.
        record = {**_SAMPLE_RECORD, "type": ["BibliographicResource", "Book"]}
        self.assertFalse(_record_matches(record))


class TestTocDownloadUrl(unittest.TestCase):
    def test_returns_first_entry_id(self):
        self.assertEqual(
            _toc_download_url(_SAMPLE_RECORD),
            "https://digitale-objekte.hbz-nrw.de/storage/2011/03/19/file_10/4104671.pdf",
        )

    def test_returns_none_when_absent(self):
        self.assertIsNone(_toc_download_url({}))

    def test_returns_none_for_schemeless_url(self):
        # A real lobid-resources record (found 2026-08-15) had a
        # tableOfContents[].id missing its scheme entirely
        # ("/www.folkwang-uni.de/.../HT016325725.pdf") -- httpx raises
        # ValueError (not httpx.HTTPError) for such a URL, which
        # _acquire_record's retry handling doesn't catch, so this must be
        # filtered out here rather than passed through.
        record = {"tableOfContents": [{"id": "/www.example.edu/fileadmin/toc.pdf"}]}
        self.assertIsNone(_toc_download_url(record))


class TestRecordKey(unittest.TestCase):
    def test_prefers_isbn(self):
        self.assertEqual(_record_key(_SAMPLE_RECORD), "9783899718188")

    def test_falls_back_to_record_id(self):
        record = {k: v for k, v in _SAMPLE_RECORD.items() if k != "isbn"}
        self.assertEqual(_record_key(record), "990183806670206441")


class TestRecordLanguage(unittest.TestCase):
    def test_maps_iso639_2_to_iso639_1(self):
        self.assertEqual(_record_language(_SAMPLE_RECORD), "de")

    def test_falls_back_to_raw_code_when_unmapped(self):
        record = {**_SAMPLE_RECORD, "language": [{"id": ".../vocabulary/iso639-2/wen"}]}
        self.assertEqual(_record_language(record), "wen")

    def test_none_when_absent(self):
        self.assertIsNone(_record_language({}))


class TestRecordApiUrl(unittest.TestCase):
    def test_strips_jsonld_fragment_and_adds_format(self):
        self.assertEqual(
            _record_api_url(_SAMPLE_RECORD),
            "http://lobid.org/resources/990183806670206441?format=json",
        )

    def test_none_when_id_absent(self):
        self.assertIsNone(_record_api_url({}))


class TestManifestEntryFromRecord(unittest.TestCase):
    def test_builds_expected_shape(self):
        entry = manifest_entry_from_record(_SAMPLE_RECORD, "9783899718188.pdf")
        self.assertEqual(entry["filename"], "9783899718188.pdf")
        self.assertEqual(entry["title"], "Systemtheorie in den Fachwissenschaften")
        self.assertEqual(entry["language"], "de")
        self.assertIsNone(entry["doi"])
        self.assertEqual(
            entry["toc_download_url"],
            "https://digitale-objekte.hbz-nrw.de/storage/2011/03/19/file_10/4104671.pdf",
        )
        self.assertEqual(entry["license"], "CC0-1.0")
        self.assertEqual(entry["license_source"], "dnb")
        self.assertEqual(
            entry["lobid_url"],
            "http://lobid.org/resources/990183806670206441?format=json",
        )
        self.assertNotIn("lobid_record", entry)


class TestSearchByIsbn(unittest.TestCase):
    def test_returns_first_member(self):
        client = Mock()
        client.get.return_value = _json_response({"member": [_SAMPLE_RECORD]})
        self.assertEqual(_search_by_isbn("9783899718188", client), _SAMPLE_RECORD)

    def test_returns_none_when_no_member(self):
        client = Mock()
        client.get.return_value = _json_response({"member": []})
        self.assertIsNone(_search_by_isbn("0000000000000", client))


class TestAcquireRecord(unittest.TestCase):
    """rate_limit_seconds=0 everywhere below to avoid a real time.sleep()
    per test -- simplest option since it's already a parameter. Each test
    patches dnb_toc_ground_truth.corpus.CORPUS_DIR to a tempdir so
    _acquire_record's corpus.pdf_dir()/corpus.lobid_cache_dir() calls
    (which read the module-level constant fresh at call time) resolve
    into isolated per-test directories rather than the real corpus."""

    def _setup(self, tmp):
        manifest_path = corpus.manifest_path()
        _ensure_manifest_shell(manifest_path)
        return manifest_path

    def test_acquires_new_matching_record(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            manifest_path = self._setup(tmp)
            client = Mock()
            response = Mock()
            response.raise_for_status = Mock()
            response.content = b"%PDF-1.4 fake toc bytes"
            client.get.return_value = response
            seen_keys = set()

            result = _acquire_record(_SAMPLE_RECORD, manifest_path, client, 0, seen_keys)

            self.assertIsNone(result)
            pdf_path = corpus.pdf_path("9783899718188")
            self.assertTrue(pdf_path.exists())
            self.assertEqual(pdf_path.read_bytes(), b"%PDF-1.4 fake toc bytes")
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(len(data["books"]), 1)
            self.assertEqual(data["books"][0]["filename"], "9783899718188.pdf")
            self.assertIn("9783899718188", seen_keys)
            lobid_path = corpus.lobid_cache_dir() / "9783899718188.lobid.json"
            self.assertTrue(lobid_path.exists())
            self.assertEqual(json.loads(lobid_path.read_text(encoding="utf-8")), _SAMPLE_RECORD)

    def test_skips_already_acquired_key(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            manifest_path = self._setup(tmp)
            client = Mock()
            seen_keys = {"9783899718188"}

            result = _acquire_record(_SAMPLE_RECORD, manifest_path, client, 0, seen_keys)

            self.assertIsNotNone(result)
            client.get.assert_not_called()
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(data["books"], [])

    def test_skips_non_matching_record_without_any_network_call(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            manifest_path = self._setup(tmp)
            client = Mock()
            record = {**_SAMPLE_RECORD, "type": ["BibliographicResource", "Article"]}
            seen_keys = set()

            result = _acquire_record(record, manifest_path, client, 0, seen_keys)

            self.assertIsNotNone(result)
            client.get.assert_not_called()
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(data["books"], [])
            self.assertEqual(seen_keys, set())

    def test_download_http_error_is_caught_and_skipped(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            manifest_path = self._setup(tmp)
            client = Mock()
            response = Mock()
            response.raise_for_status = Mock(side_effect=httpx.HTTPError("boom"))
            client.get.return_value = response
            seen_keys = set()

            result = _acquire_record(_SAMPLE_RECORD, manifest_path, client, 0, seen_keys)

            self.assertIsNotNone(result)
            self.assertIn("boom", result)
            self.assertFalse(corpus.pdf_path("9783899718188").exists())
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(data["books"], [])
            self.assertEqual(seen_keys, set())


class TestAcquireRecordCrossref(unittest.TestCase):
    """_acquire_record's Crossref hook -- see TestAcquireRecord's own
    docstring for the tempdir-isolation pattern this reuses."""

    def _setup(self, tmp):
        manifest_path = corpus.manifest_path()
        _ensure_manifest_shell(manifest_path)
        return manifest_path

    def test_writes_doi_found_via_crossref(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            manifest_path = self._setup(tmp)
            pdf_response = Mock()
            pdf_response.raise_for_status = Mock()
            pdf_response.content = b"%PDF-1.4 fake toc bytes"
            crossref_response = _json_response(
                {"message": {"items": [{"type": "book", "DOI": "10.1515/book-doi", "title": ["X"]}]}}
            )
            client = Mock()
            client.get.side_effect = [pdf_response, crossref_response]
            seen_keys = set()

            result = _acquire_record(
                _SAMPLE_RECORD, manifest_path, client, 0, seen_keys, contact_email="me@example.org",
            )

            self.assertIsNone(result)
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(data["books"][0]["doi"], "10.1515/book-doi")
            cache_path = corpus.crossref_cache_dir() / "9783899718188.crossref.json"
            self.assertTrue(cache_path.exists())

    def test_crossref_failure_leaves_doi_null_and_still_acquires(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            manifest_path = self._setup(tmp)
            pdf_response = Mock()
            pdf_response.raise_for_status = Mock()
            pdf_response.content = b"%PDF-1.4 fake toc bytes"
            client = Mock()
            client.get.side_effect = [pdf_response, httpx.HTTPError("boom")]
            seen_keys = set()

            result = _acquire_record(_SAMPLE_RECORD, manifest_path, client, 0, seen_keys)

            self.assertIsNone(result)
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertIsNone(data["books"][0]["doi"])

    def test_non_isbn_key_skips_crossref_lookup_entirely(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            manifest_path = self._setup(tmp)
            record = {k: v for k, v in _SAMPLE_RECORD.items() if k != "isbn"}
            pdf_response = Mock()
            pdf_response.raise_for_status = Mock()
            pdf_response.content = b"%PDF-1.4 fake toc bytes"
            client = Mock()
            client.get.return_value = pdf_response
            seen_keys = set()

            result = _acquire_record(record, manifest_path, client, 0, seen_keys)

            self.assertIsNone(result)
            client.get.assert_called_once()  # only the PDF download, no Crossref lookup


class TestScanAndAcquire(unittest.TestCase):
    def test_stops_early_once_cumulative_limit_reached(self):
        # Three matching records, but acquired_so_far=1 and limit=2, so
        # only one more should be acquired before the loop stops -- and
        # it must stop consuming the iterator immediately, not drain it.
        records = [
            {**_SAMPLE_RECORD, "isbn": ["1111111111111"]},
            {**_SAMPLE_RECORD, "isbn": ["2222222222222"]},
            {**_SAMPLE_RECORD, "isbn": ["3333333333333"]},
        ]

        def _record_stream():
            yield from records
            self.fail("iterator was drained past the limit")

        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            manifest_path = corpus.manifest_path()
            _ensure_manifest_shell(manifest_path)
            client = Mock()
            client.get.return_value = _json_response({})  # PDF download response; content below
            client.get.return_value.content = b"%PDF-fake"
            scanned, newly_acquired = _scan_and_acquire(
                _record_stream(), manifest_path, client,
                rate_limit_seconds=0, limit=2, seen_keys=set(), acquired_so_far=1,
            )
        self.assertEqual(newly_acquired, 1)
        self.assertEqual(scanned, 2)

    def test_consumes_whole_iterator_when_no_limit(self):
        records = [
            {**_SAMPLE_RECORD, "isbn": ["1111111111111"]},
            {**_SAMPLE_RECORD, "isbn": ["2222222222222"]},
        ]
        with tempfile.TemporaryDirectory() as tmp, patch.object(corpus, "CORPUS_DIR", Path(tmp)):
            manifest_path = corpus.manifest_path()
            _ensure_manifest_shell(manifest_path)
            client = Mock()
            client.get.return_value = _json_response({})
            client.get.return_value.content = b"%PDF-fake"
            scanned, newly_acquired = _scan_and_acquire(
                iter(records), manifest_path, client,
                rate_limit_seconds=0, limit=None, seen_keys=set(), acquired_so_far=0,
            )
        self.assertEqual(newly_acquired, 2)
        self.assertEqual(scanned, 2)


class TestChunkStreamReader(unittest.TestCase):
    def test_read_reassembles_chunks_of_any_requested_size(self):
        reader = _ChunkStreamReader(iter([b"ab", b"cde", b"f"]))
        self.assertEqual(reader.read(4), b"abcd")
        self.assertEqual(reader.read(2), b"ef")
        self.assertEqual(reader.read(10), b"")

    def test_supports_gzip_decompression_through_small_reads(self):
        original = b"line one\nline two\nline three\n"
        compressed = gzip.compress(original)
        # Force many small reads to exercise the buffering logic.
        chunks = [compressed[i:i + 3] for i in range(0, len(compressed), 3)]
        reader = _ChunkStreamReader(iter(chunks))
        with gzip.GzipFile(fileobj=reader) as gz:
            self.assertEqual(gz.read(), original)


class TestIterDumpRecordsFromChunks(unittest.TestCase):
    def test_decodes_gzipped_jsonl_stream(self):
        lines = [json.dumps({"n": 1}), json.dumps({"n": 2}), ""]
        compressed = gzip.compress("\n".join(lines).encode("utf-8"))
        chunks = [compressed[i:i + 5] for i in range(0, len(compressed), 5)]
        records = list(_iter_dump_records_from_chunks(iter(chunks)))
        self.assertEqual(records, [{"n": 1}, {"n": 2}])


class TestReadIsbnsFile(unittest.TestCase):
    def test_ignores_blank_lines_and_comments(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "isbns.txt"
            path.write_text("9783899718188\n\n# a comment\n9781234567897\n", encoding="utf-8")
            self.assertEqual(_read_isbns_file(path), ["9783899718188", "9781234567897"])


class TestManifestFileHelpers(unittest.TestCase):
    def test_ensure_manifest_shell_creates_toc_only_shell(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "sub" / "manifest.json"
            _ensure_manifest_shell(manifest_path)
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(data, {"toc_only": True, "books": []})

    def test_ensure_manifest_shell_is_a_noop_when_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text('{"toc_only": true, "books": [{"filename": "x.pdf"}]}', encoding="utf-8")
            _ensure_manifest_shell(manifest_path)
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(len(data["books"]), 1)

    def test_append_book_adds_to_existing_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text('{"toc_only": true, "books": []}', encoding="utf-8")
            _append_book(manifest_path, {"filename": "a.pdf"})
            _append_book(manifest_path, {"filename": "b.pdf"})
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual([b["filename"] for b in data["books"]], ["a.pdf", "b.pdf"])

    def test_load_existing_keys_returns_filename_stems(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(
                json.dumps({"toc_only": True, "books": [{"filename": "9783899718188.pdf"}]}),
                encoding="utf-8",
            )
            self.assertEqual(_load_existing_keys(manifest_path), {"9783899718188"})

    def test_load_existing_keys_empty_when_manifest_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(_load_existing_keys(Path(tmp) / "manifest.json"), set())


if __name__ == "__main__":
    unittest.main()
