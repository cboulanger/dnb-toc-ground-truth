# DNB/lobid digitized-TOC corpus acquisition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `evaluation/scripts/fetch_dnb_toc_corpus.py` (acquires real DNB-scanned table-of-contents PDFs via the `lobid-resources` API into a new `evaluation/corpus/dnb-toc-only/` corpus) and `evaluation/scripts/measure_dnb_scan_noise_stats.py` (measures real font-size contrast/dispersion from that corpus's ALTO to calibrate `alto_scan_noise.py`'s synthetic constants), per `docs/superpowers/specs/2026-08-14-dnb-toc-corpus-acquisition-design.md`.

**Architecture:** `evaluation/harness.py`'s `list_corpora()` gains an opt-in `include_toc_only` parameter so every existing script keeps ignoring the new corpus by default. `fetch_dnb_toc_corpus.py` is a standalone acquisition script (two modes: `--isbns-file` for targeted lookups, `--from-dump` for the full lobid-resources bulk dump) that writes only `manifest.json` to git (PDFs stay gitignored, same as every other corpus). `measure_dnb_scan_noise_stats.py` reuses `pdfalto_runner.py`/`layout_features.py` to compute real per-page contrast and per-line font-size dispersion, printed as a comparison table against `alto_scan_noise.py`'s hand-picked constants.

**Tech Stack:** Python 3.12, `httpx` (HTTP client, already a project dependency), stdlib `gzip`/`json`/`xml.etree.ElementTree`/`statistics`, `unittest` (matches this repo's existing test style), `uv run pytest`.

**Verified against the live API while writing this plan** (2026-08-14): the confirmed bulk-dump URL is `https://lobid.org/download/dumps/lobid-resources/latestLobidResources.jsonl.gz` (currently ~21.5GB gzip, one JSON object per line, `Content-Type: application/x-gzip`); `curl -s "https://lobid.org/resources/search?q=isbn:9783899718188&format=json"` returns a full record confirming the exact field shapes used below (`type`, `isbn`, `language`, `tableOfContents`, no reliable `doi` field — lobid-resources records rarely carry one, unlike Crossref).

**Do NOT run `fetch_dnb_toc_corpus.py --from-dump` against the real 21.5GB dump during implementation** — that's an hours-long, multi-GB download against a shared public service, and is explicitly a manual follow-up action for the user once the script is built and tested (see Task 4's smoke test instead, which uses the fast `--isbns-file` mode with a handful of real ISBNs).

---

## File Structure

- Modify: `evaluation/harness.py` — `list_corpora()` gains `include_toc_only: bool = False`.
- Modify: `tests/test_harness.py` — new test coverage for the above.
- Create: `evaluation/scripts/fetch_dnb_toc_corpus.py` — acquisition script (pure record-parsing helpers + CLI `main()`).
- Create: `tests/test_fetch_dnb_toc_corpus.py` — unit tests for the pure helpers (mocked HTTP, no live network).
- Create: `evaluation/corpus/dnb-toc-only/manifest.json` — written by the script itself (Task 4's smoke test), committed with a small number of real, verified entries.
- Create: `evaluation/scripts/measure_dnb_scan_noise_stats.py` — measurement script (pure ALTO-statistics helpers + CLI `main()`).
- Create: `tests/test_measure_dnb_scan_noise_stats.py` — unit tests against a hand-built ALTO fixture (matches `tests/test_alto_scan_noise.py`'s style).
- Modify: `evaluation/CLAUDE.md` — Step 1 workflow note.
- Modify: `evaluation/scripts/README.md` — `--help` dump entries for both new scripts, alphabetically placed.

---

### Task 1: `list_corpora(include_toc_only=False)` in the shared harness

**Files:**
- Modify: `evaluation/harness.py:41-50`
- Test: `tests/test_harness.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_harness.py`, inside `class TestListCorpora`:

```python
    def test_excludes_toc_only_corpus_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "corpus-a").mkdir()
            (root / "corpus-a" / "manifest.json").write_text('{"books": []}', encoding="utf-8")
            (root / "dnb-toc-only").mkdir()
            (root / "dnb-toc-only" / "manifest.json").write_text(
                '{"toc_only": true, "books": []}', encoding="utf-8",
            )
            with patch("evaluation.harness.CORPUS_ROOT", root):
                self.assertEqual(list_corpora(), ["corpus-a"])
                self.assertEqual(list_corpora(include_toc_only=True), ["corpus-a", "dnb-toc-only"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_harness.py::TestListCorpora::test_excludes_toc_only_corpus_by_default -v`
Expected: FAIL with `TypeError: list_corpora() got an unexpected keyword argument 'include_toc_only'`

- [ ] **Step 3: Implement**

Replace `evaluation/harness.py:41-50`:

```python
def list_corpora(include_toc_only: bool = False) -> list[str]:
    """Sorted names of every subfolder under evaluation/corpus/ that has a
    manifest.json -- the single source of truth for "what corpora exist"
    that every runner iterates over. A corpus whose manifest.json sets
    "toc_only": true (see dnb-toc-only/) is excluded unless
    include_toc_only=True: every existing caller assumes a corpus has
    fetchable PDFs and full .expected.json chapter fields, neither of
    which a toc-only corpus has."""
    if not CORPUS_ROOT.is_dir():
        return []
    names = []
    for p in CORPUS_ROOT.iterdir():
        manifest_path = p / "manifest.json"
        if not (p.is_dir() and manifest_path.exists()):
            continue
        if not include_toc_only:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("toc_only"):
                continue
        names.append(p.name)
    return sorted(names)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_harness.py -v`
Expected: PASS (all tests in the file, including the two pre-existing `TestListCorpora` tests)

- [ ] **Step 5: Commit**

```bash
git add evaluation/harness.py tests/test_harness.py
git commit -m "feat: add include_toc_only opt-in to list_corpora()"
```

---

### Task 2: `fetch_dnb_toc_corpus.py` — acquisition script

**Files:**
- Create: `evaluation/scripts/fetch_dnb_toc_corpus.py`
- Test: `tests/test_fetch_dnb_toc_corpus.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_fetch_dnb_toc_corpus.py`:

```python
"""Unit tests for evaluation/scripts/fetch_dnb_toc_corpus.py's pure logic
(record filtering, field extraction, streaming JSON-Lines decode) against
mocked httpx responses and an in-memory gzip stream -- no live network,
matching tests/test_discover_crossref_candidates.py's convention. The real
network-calling main()/_run_isbns_file()/_run_from_dump() orchestration is
exercised manually (see docs/superpowers/plans/2026-08-14-dnb-toc-corpus-acquisition.md
Task 4's smoke test), matching fetch_crossref_gt_corpus.py's existing
convention of no pytest coverage for its own network-calling entry point."""

import gzip
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from evaluation.scripts.fetch_dnb_toc_corpus import (
    _ChunkStreamReader,
    _append_book,
    _ensure_manifest_shell,
    _iter_dump_records_from_chunks,
    _load_existing_keys,
    _read_isbns_file,
    _record_key,
    _record_language,
    _record_matches,
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


class TestTocDownloadUrl(unittest.TestCase):
    def test_returns_first_entry_id(self):
        self.assertEqual(
            _toc_download_url(_SAMPLE_RECORD),
            "https://digitale-objekte.hbz-nrw.de/storage/2011/03/19/file_10/4104671.pdf",
        )

    def test_returns_none_when_absent(self):
        self.assertIsNone(_toc_download_url({}))


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
        self.assertEqual(entry["lobid_record"], _SAMPLE_RECORD)


class TestSearchByIsbn(unittest.TestCase):
    def test_returns_first_member(self):
        client = Mock()
        client.get.return_value = _json_response({"member": [_SAMPLE_RECORD]})
        self.assertEqual(_search_by_isbn("9783899718188", client), _SAMPLE_RECORD)

    def test_returns_none_when_no_member(self):
        client = Mock()
        client.get.return_value = _json_response({"member": []})
        self.assertIsNone(_search_by_isbn("0000000000000", client))


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_fetch_dnb_toc_corpus.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'evaluation.scripts.fetch_dnb_toc_corpus'`

- [ ] **Step 3: Write the implementation**

Create `evaluation/scripts/fetch_dnb_toc_corpus.py`:

```python
#!/usr/bin/env python3
"""Acquires real DNB-scanned table-of-contents PDFs via the lobid-resources
API (lobid.org/resources) into evaluation/corpus/dnb-toc-only/ -- see
docs/superpowers/specs/2026-08-14-dnb-toc-corpus-acquisition-design.md.

Every acquired record's PDF is a CC0-licensed, DNB-digitized TOC scan
(the "Kataloganreicherung" program), never the surrounding book, so this
corpus is deliberately shaped differently from open-access/copyrighted-scans
(see manifest_entry_from_record below and the design spec) -- no
extraction_type/embedded_toc/oa/download_url fields, and no
<id>.expected.json is produced by this script.

Two acquisition modes, because tableOfContents is not queryable
server-side in lobid-resources (confirmed empirically -- see design spec):

    uv run python evaluation/scripts/fetch_dnb_toc_corpus.py \\
        --isbns-file /tmp/isbns.txt --limit 20
    uv run python evaluation/scripts/fetch_dnb_toc_corpus.py \\
        --from-dump --limit 500

--from-dump streams the full weekly lobid-resources JSON-Lines dump
(~21.5GB gzip as of 2026-08-14, one bibliographic record per line) and
filters client-side -- there is no per-request cost per scanned record,
only per acquired match (one PDF download + a rate-limit sleep), but a
full scan is still an hours-long, many-GB operation against a shared
public service. Run it deliberately, not as part of routine development.

manifest.json is the only file this script writes to git -- PDFs stay
gitignored (evaluation/.gitignore's blanket *.pdf), same as every other
corpus. See "PDFs are never committed" in the design spec: this corpus's
full PDF set is meant for a separate Zenodo dataset upload, not this repo.
"""

import argparse
import gzip
import json
import sys
import time
from pathlib import Path
from typing import Iterator, Optional

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from evaluation.harness import corpus_dir

_DUMP_URL_DEFAULT = "https://lobid.org/download/dumps/lobid-resources/latestLobidResources.jsonl.gz"
_SEARCH_URL = "https://lobid.org/resources/search"
_CORPUS_NAME = "dnb-toc-only"

# lobid-resources carries language as a full ISO 639-2 URI
# (.../iso639-2/ger); every other corpus's manifest.json uses ISO 639-1
# two-letter codes ("de", "en") -- map the common cases so this field
# stays consistent with the shared manifest shape, falling back to the
# raw three-letter code for anything not in this short list.
_ISO_639_2_TO_1 = {
    "ger": "de", "eng": "en", "fre": "fr", "fra": "fr", "spa": "es",
    "ita": "it", "dut": "nl", "nld": "nl", "lat": "la", "rus": "ru",
}


def _record_matches(record: dict) -> bool:
    """A lobid-resources record is a usable acquisition target if it's
    typed as a Book or EditedVolume and carries a non-empty
    tableOfContents array."""
    types = record.get("type") or []
    if not any(t in ("Book", "EditedVolume") for t in types):
        return False
    return bool(record.get("tableOfContents"))


def _toc_download_url(record: dict) -> Optional[str]:
    for entry in record.get("tableOfContents") or []:
        url = entry.get("id")
        if url:
            return url
    return None


def _record_key(record: dict) -> str:
    """Manifest key (and PDF filename stem): the record's first ISBN when
    present, otherwise the numeric ID from its own lobid URI -- not every
    lobid-resources record carries an ISBN (confirmed empirically against
    the live dump)."""
    isbns = record.get("isbn") or []
    if isbns:
        return isbns[0]
    record_id = (record.get("id") or "").rstrip("#!")
    return record_id.rsplit("/", 1)[-1] or "unknown"


def _record_language(record: dict) -> Optional[str]:
    languages = record.get("language") or []
    if not languages:
        return None
    code = (languages[0].get("id") or "").rsplit("/", 1)[-1]
    if not code:
        return None
    return _ISO_639_2_TO_1.get(code, code)


def _record_doi(record: dict) -> Optional[str]:
    # lobid-resources records rarely carry a DOI (confirmed empirically --
    # unlike Crossref, DNB/hbz catalog records don't reliably have one).
    return record.get("doi")


def manifest_entry_from_record(record: dict, filename: str) -> dict:
    """Builds this corpus's manifest.json book entry. lobid_record holds
    the full record verbatim, nested under its own key rather than
    flattened -- it cost nothing extra to fetch (the whole record already
    has to be pulled to read tableOfContents) so it's kept in full for
    future analysis this script doesn't otherwise use; no other code in
    this repo reads that key."""
    return {
        "filename": filename,
        "title": record.get("title") or "",
        "language": _record_language(record),
        "doi": _record_doi(record),
        "toc_download_url": _toc_download_url(record),
        "license": "CC0-1.0",
        "license_source": "dnb",
        "lobid_record": record,
    }


def _search_by_isbn(isbn: str, client: httpx.Client) -> Optional[dict]:
    response = client.get(_SEARCH_URL, params={"q": f"isbn:{isbn}", "format": "json"})
    response.raise_for_status()
    members = response.json().get("member", [])
    return members[0] if members else None


class _ChunkStreamReader:
    """Minimal file-like adapter so gzip.GzipFile can decompress an
    iterator of byte chunks (httpx's streaming response body) without
    buffering the whole multi-GB response in memory."""

    def __init__(self, chunks: Iterator[bytes]):
        self._chunks = chunks
        self._buffer = b""

    def read(self, size: int = -1) -> bytes:
        while size < 0 or len(self._buffer) < size:
            try:
                self._buffer += next(self._chunks)
            except StopIteration:
                break
        if size < 0:
            result, self._buffer = self._buffer, b""
        else:
            result, self._buffer = self._buffer[:size], self._buffer[size:]
        return result


def _iter_dump_records_from_chunks(chunks: Iterator[bytes]) -> Iterator[dict]:
    with gzip.GzipFile(fileobj=_ChunkStreamReader(chunks)) as gz:
        for raw_line in gz:
            line = raw_line.strip()
            if line:
                yield json.loads(line)


def _iter_dump_records(url: str, client: httpx.Client) -> Iterator[dict]:
    with client.stream("GET", url) as response:
        response.raise_for_status()
        yield from _iter_dump_records_from_chunks(response.iter_bytes())


def _read_isbns_file(path: Path) -> list[str]:
    isbns = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            isbns.append(line)
    return isbns


def _load_existing_keys(manifest_path: Path) -> set[str]:
    if not manifest_path.exists():
        return set()
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {Path(book["filename"]).stem for book in data.get("books", [])}


def _ensure_manifest_shell(manifest_path: Path) -> None:
    if manifest_path.exists():
        return
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps({"toc_only": True, "books": []}, indent=2) + "\n", encoding="utf-8",
    )


def _append_book(manifest_path: Path, entry: dict) -> None:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["books"].append(entry)
    manifest_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )


def _acquire_record(
    record: dict,
    cdir: Path,
    manifest_path: Path,
    client: httpx.Client,
    rate_limit_seconds: float,
    seen_keys: set[str],
) -> bool:
    """Downloads one matched record's TOC PDF and appends its manifest
    entry. Returns True iff a new book was acquired (so callers can count
    toward --limit); False for a non-match, an already-acquired key, or a
    record with no resolvable TOC URL."""
    if not _record_matches(record):
        return False
    key = _record_key(record)
    if key in seen_keys:
        return False
    toc_url = _toc_download_url(record)
    if not toc_url:
        return False
    filename = f"{key}.pdf"
    response = client.get(toc_url)
    response.raise_for_status()
    (cdir / filename).write_bytes(response.content)
    _append_book(manifest_path, manifest_entry_from_record(record, filename))
    seen_keys.add(key)
    print(f"[fetch] {filename} <- {toc_url}")
    time.sleep(rate_limit_seconds)
    return True


def _run_isbns_file(args: argparse.Namespace, cdir: Path, manifest_path: Path, client: httpx.Client) -> None:
    seen_keys = _load_existing_keys(manifest_path)
    acquired = 0
    for isbn in _read_isbns_file(Path(args.isbns_file)):
        if args.limit is not None and acquired >= args.limit:
            break
        record = _search_by_isbn(isbn, client)
        if record is None:
            print(f"[skip] {isbn}: no lobid-resources record found")
            continue
        if _acquire_record(record, cdir, manifest_path, client, args.rate_limit_seconds, seen_keys):
            acquired += 1
        else:
            print(f"[skip] {isbn}: not a usable Book/EditedVolume-with-TOC record")
    print(f"Acquired {acquired} new book(s).")


def _run_from_dump(args: argparse.Namespace, cdir: Path, manifest_path: Path, client: httpx.Client) -> None:
    seen_keys = _load_existing_keys(manifest_path)
    acquired = 0
    scanned = 0
    for record in _iter_dump_records(args.dump_url, client):
        scanned += 1
        if scanned % 100_000 == 0:
            print(f"[scan] {scanned:,} records scanned, {acquired} acquired so far")
        if args.limit is not None and acquired >= args.limit:
            break
        if _acquire_record(record, cdir, manifest_path, client, args.rate_limit_seconds, seen_keys):
            acquired += 1
    print(f"Scanned {scanned:,} records, acquired {acquired} new book(s).")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--from-dump", action="store_true",
        help="Scan the full lobid-resources JSON-Lines dump for matching records (hours-long; see module docstring)",
    )
    mode.add_argument(
        "--isbns-file",
        help="Path to a text file of ISBNs (one per line, '#' comments allowed) to look up individually",
    )
    parser.add_argument(
        "--dump-url", default=_DUMP_URL_DEFAULT,
        help=f"lobid-resources dump URL for --from-dump (default: {_DUMP_URL_DEFAULT})",
    )
    parser.add_argument("--limit", type=int, default=None, help="Stop after acquiring this many new books")
    parser.add_argument(
        "--rate-limit-seconds", type=float, default=1.0,
        help="Delay after each TOC PDF download, to stay polite to DNB's servers (default: 1.0)",
    )
    args = parser.parse_args()

    cdir = corpus_dir(_CORPUS_NAME)
    cdir.mkdir(parents=True, exist_ok=True)
    manifest_path = cdir / "manifest.json"
    _ensure_manifest_shell(manifest_path)

    with httpx.Client(
        follow_redirects=True,
        timeout=httpx.Timeout(connect=30.0, read=120.0, write=30.0, pool=30.0),
    ) as client:
        if args.isbns_file:
            _run_isbns_file(args, cdir, manifest_path, client)
        else:
            _run_from_dump(args, cdir, manifest_path, client)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_fetch_dnb_toc_corpus.py -v`
Expected: PASS (every test class above)

- [ ] **Step 5: Commit**

```bash
git add evaluation/scripts/fetch_dnb_toc_corpus.py tests/test_fetch_dnb_toc_corpus.py
git commit -m "feat: add fetch_dnb_toc_corpus.py for DNB/lobid TOC-scan acquisition"
```

---

### Task 3: `evaluation/CLAUDE.md` workflow note

**Files:**
- Modify: `evaluation/CLAUDE.md` (Step 1 section, currently starting at line 302, "## Step 1: Transcribe the table of contents")

- [ ] **Step 1: Add the note**

In `evaluation/CLAUDE.md`, immediately after the `## Step 1: Transcribe the table of contents` heading (before its existing "Open the PDF and write out..." paragraph), insert:

```markdown
**Before transcribing by hand, check whether a DNB-digitized TOC scan
already exists** for this book: look in
`evaluation/corpus/dnb-toc-only/manifest.json` (see
`evaluation/scripts/fetch_dnb_toc_corpus.py` --
`docs/superpowers/specs/2026-08-14-dnb-toc-corpus-acquisition-design.md`)
for an entry with this book's ISBN, or query live:
`curl -s "https://lobid.org/resources/search?q=isbn:<isbn>&format=json"`
and check for a populated `tableOfContents` field. When present, it's a
ready-made, already-OCR'd TOC scan to transcribe from instead of locating
and reading the TOC pages inside the raw book PDF from scratch -- you
still verify every entry by hand against the actual book exactly as Step
3 below describes; this only saves the step of finding the TOC pages
visually.

```

- [ ] **Step 2: Verify the insertion renders correctly**

Run: `head -320 evaluation/CLAUDE.md | tail -30` and confirm the new paragraph sits between the `## Step 1` heading and the pre-existing "Open the PDF..." paragraph, with no broken markdown.

- [ ] **Step 3: Commit**

```bash
git add evaluation/CLAUDE.md
git commit -m "docs: point new-ground-truth workflow at the DNB TOC-scan corpus"
```

---

### Task 4: Smoke-test acquisition — populate and commit the real `dnb-toc-only/manifest.json`

This is the one task in this plan that touches the live lobid-resources API for real. Keep it small and fast (`--isbns-file`, not `--from-dump`) — the goal is a working, verified proof that the whole pipeline (search → filter → download → manifest write) functions end-to-end, not a full corpus. The user runs `--from-dump` separately, later, at their own pace.

**Files:**
- Create (gitignored, not committed): a scratch ISBN list
- Create (committed): `evaluation/corpus/dnb-toc-only/manifest.json`
- Create (gitignored): `evaluation/corpus/dnb-toc-only/*.pdf`

- [ ] **Step 1: Build a small real ISBN list**

Use ISBNs already known to have a DNB TOC scan (the one confirmed live while writing this plan, plus this project's own `copyrighted-scans`/`open-access` manifests are a good source of German-language academic titles likely to be in the DNB "Kataloganreicherung" program — spot-check 4-6 by hand first with `curl -s "https://lobid.org/resources/search?q=isbn:<isbn>&format=json" | python3 -c "import json,sys; d=json.load(sys.stdin); print(bool(d['member'][0].get('tableOfContents')) if d['member'] else 'NO MEMBER')"` and keep only the ones that print `True`):

```bash
uv run python -c "
import json
manifest = json.load(open('evaluation/corpus/copyrighted-scans/manifest.json'))
for b in manifest['books']:
    if b.get('doi') is None and b.get('language') == 'de':
        print(b['filename'].removesuffix('.pdf'))
" | head -20
```

Pick ISBN-shaped filenames from that output (or from `open-access/manifest.json`) and spot-check each with the `curl` one-liner above. Write the 5-10 that come back `True` to `/tmp/dnb_smoke_isbns.txt`, one per line. `9783899718188` (confirmed in this plan's header) must be one of them.

- [ ] **Step 2: Run the acquisition script for real**

```bash
uv run python evaluation/scripts/fetch_dnb_toc_corpus.py --isbns-file /tmp/dnb_smoke_isbns.txt --limit 10
```

Expected: `[fetch] <isbn>.pdf <- https://...` printed for each match, a final `Acquired N new book(s).` line, `evaluation/corpus/dnb-toc-only/manifest.json` created with `N` entries, and `N` PDFs written alongside it (gitignored — confirm with `git status` that only `manifest.json` shows as untracked, not the PDFs).

- [ ] **Step 3: Spot-check the acquired PDFs**

Open at least 2 of the downloaded PDFs (e.g. via the `Read` tool, or `open evaluation/corpus/dnb-toc-only/9783899718188.pdf`) and confirm each really is a table-of-contents scan with a DNB digitization footer, not a mismatched or corrupt link.

- [ ] **Step 4: Run the full non-integration test suite**

Run: `uv run pytest -q`
Expected: PASS. This corpus is excluded from every existing test by default (`list_corpora()` without `include_toc_only=True`), so nothing else should change.

- [ ] **Step 5: Commit only the manifest**

```bash
git status  # confirm only evaluation/corpus/dnb-toc-only/manifest.json is untracked, PDFs are gitignored
git add evaluation/corpus/dnb-toc-only/manifest.json
git commit -m "data: seed dnb-toc-only corpus with a small verified smoke-test batch"
```

---

### Task 5: `measure_dnb_scan_noise_stats.py` — real-scan calibration measurement

**Files:**
- Create: `evaluation/scripts/measure_dnb_scan_noise_stats.py`
- Test: `tests/test_measure_dnb_scan_noise_stats.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_measure_dnb_scan_noise_stats.py`:

```python
"""Unit tests for evaluation/scripts/measure_dnb_scan_noise_stats.py's
ALTO-statistics helpers, against the same style of hand-built fixture used
by tests/test_alto_scan_noise.py."""

import tempfile
import unittest
from pathlib import Path

from evaluation.scripts.measure_dnb_scan_noise_stats import (
    body_line_dispersion_ratios,
    contrast_ratios,
    summarize,
)

# Page 1: one title line (24.0) and three body lines -- two at exactly
# 10.0 (giving statistics.mode an unambiguous winner) and one at 10.3
# (a body-like line within the +/-10% dispersion band). Page 2 is
# intentionally blank (no TextLine at all) to confirm empty pages are
# skipped by both measurements, matching layout_features.py's convention.
_FIXTURE_ALTO_XML = """<?xml version="1.0" encoding="UTF-8"?>
<alto xmlns="http://www.loc.gov/standards/alto/ns-v3#">
  <Styles>
    <TextStyle ID="title" FONTFAMILY="sans" FONTSIZE="24.0"/>
    <TextStyle ID="body" FONTFAMILY="serif" FONTSIZE="10.0"/>
    <TextStyle ID="body_jit" FONTFAMILY="serif" FONTSIZE="10.3"/>
  </Styles>
  <Layout>
    <Page ID="Page1" PHYSICAL_IMG_NR="1" WIDTH="500" HEIGHT="600">
      <PrintSpace>
        <TextBlock ID="p1_b1">
          <TextLine ID="p1_t1" HPOS="200" VPOS="50" WIDTH="150" HEIGHT="24">
            <String ID="p1_w1" CONTENT="Chapter One" HPOS="200" WIDTH="150" HEIGHT="24" STYLEREFS="title"/>
          </TextLine>
          <TextLine ID="p1_t2" HPOS="48" VPOS="200" WIDTH="340" HEIGHT="12">
            <String ID="p1_w2" CONTENT="Text A" HPOS="48" WIDTH="340" HEIGHT="12" STYLEREFS="body"/>
          </TextLine>
          <TextLine ID="p1_t3" HPOS="48" VPOS="215" WIDTH="340" HEIGHT="12">
            <String ID="p1_w3" CONTENT="Text B" HPOS="48" WIDTH="340" HEIGHT="12" STYLEREFS="body"/>
          </TextLine>
          <TextLine ID="p1_t4" HPOS="48" VPOS="230" WIDTH="340" HEIGHT="12">
            <String ID="p1_w4" CONTENT="Text C" HPOS="48" WIDTH="340" HEIGHT="12" STYLEREFS="body_jit"/>
          </TextLine>
        </TextBlock>
      </PrintSpace>
    </Page>
    <Page ID="Page2" PHYSICAL_IMG_NR="2" WIDTH="500" HEIGHT="600">
      <PrintSpace/>
    </Page>
  </Layout>
</alto>
"""


class TestContrastRatios(unittest.TestCase):
    def test_one_ratio_per_non_empty_page(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.alto.xml"
            path.write_text(_FIXTURE_ALTO_XML, encoding="utf-8")
            ratios = contrast_ratios(path)
        self.assertEqual(len(ratios), 1)
        self.assertAlmostEqual(ratios[0], 2.4, places=3)  # 24.0 / 10.0 modal


class TestBodyLineDispersionRatios(unittest.TestCase):
    def test_excludes_title_includes_body_band(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.alto.xml"
            path.write_text(_FIXTURE_ALTO_XML, encoding="utf-8")
            ratios = body_line_dispersion_ratios(path)
        self.assertEqual(len(ratios), 3)  # two 10.0 lines + one 10.3 line; title excluded
        for ratio in ratios:
            self.assertLessEqual(abs(ratio - 1.0), 0.1)
        self.assertAlmostEqual(max(ratios), 1.03, places=3)


class TestSummarize(unittest.TestCase):
    def test_computes_expected_statistics(self):
        stats = summarize([1.0, 2.0, 3.0])
        self.assertEqual(stats["count"], 3)
        self.assertEqual(stats["min"], 1.0)
        self.assertEqual(stats["max"], 3.0)
        self.assertAlmostEqual(stats["mean"], 2.0)
        self.assertAlmostEqual(stats["median"], 2.0)
        self.assertAlmostEqual(stats["stdev"], 1.0)

    def test_empty_input(self):
        self.assertEqual(summarize([]), {"count": 0})


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_measure_dnb_scan_noise_stats.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'evaluation.scripts.measure_dnb_scan_noise_stats'`

- [ ] **Step 3: Write the implementation**

Create `evaluation/scripts/measure_dnb_scan_noise_stats.py`:

```python
#!/usr/bin/env python3
"""Measures real font-size contrast and per-line dispersion from the
dnb-toc-only corpus's ALTO output, to calibrate alto_scan_noise.py's
hand-picked synthetic constants (_CONTRAST_ALPHA, _FONT_JITTER) against
real scanned data -- see
docs/superpowers/specs/2026-08-14-dnb-toc-corpus-acquisition-design.md
section 3. Every page in this corpus is a confirmed TOC page by
construction (DNB only digitizes the TOC itself), so no per-page
labeling step is needed.

Usage:
    uv run python evaluation/scripts/measure_dnb_scan_noise_stats.py
    uv run python evaluation/scripts/measure_dnb_scan_noise_stats.py --pdfalto-bin ../pdfalto/pdfalto

The printed comparison table is the deliverable regardless of whether it
leads to changing alto_scan_noise.py's constants -- paste it into
evaluation/RESULTS.md by hand as a new follow-up subsection (this script
does not write RESULTS.md itself, matching every other evaluation script
in this directory)."""

import argparse
import statistics
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from evaluation.harness import corpus_dir, list_corpora, load_manifest_books
from evaluation.scripts.alto_scan_noise import _CONTRAST_ALPHA, _FONT_JITTER
from evaluation.scripts.layout_features import extract_page_features
from evaluation.scripts.pdfalto_runner import ensure_alto_xml, resolve_pdfalto_binary

_ALTO_NS = "{http://www.loc.gov/standards/alto/ns-v3#}"

# A line's font size counts as "body-like" (and so contributes a
# dispersion sample) when it sits within this fraction of the page's
# modal (most common) font size -- excludes titles/headers, which is
# exactly what _FONT_JITTER's per-clone noise is meant to model on body
# text, not title text (contrast compression is the separate,
# title-vs-body mechanism measured by contrast_ratios below).
_BODY_BAND = 0.1


def contrast_ratios(alto_path: Path) -> list[float]:
    """One font_size_max_ratio (max/modal font size) sample per non-empty
    page -- the real, uncompressed title/body contrast alto_scan_noise.py's
    _CONTRAST_ALPHA compresses synthetic (born-digital) ALTO toward."""
    features = extract_page_features(str(alto_path))
    return [f["font_size_max_ratio"] for f in features.values() if f["line_count"] > 0]


def body_line_dispersion_ratios(alto_path: Path) -> list[float]:
    """For each non-empty page, the ratio of every body-like line's font
    size (within _BODY_BAND of that page's modal size) to the modal size
    itself -- the real per-line size variation alto_scan_noise.py's
    _FONT_JITTER multiplicatively approximates on synthetic style clones."""
    tree = ET.parse(alto_path)
    root = tree.getroot()
    sizes_by_id = {
        style.get("ID"): float(style.get("FONTSIZE"))
        for style in root.iter(_ALTO_NS + "TextStyle")
        if style.get("ID") and style.get("FONTSIZE")
    }
    ratios: list[float] = []
    for page in root.iter(_ALTO_NS + "Page"):
        page_sizes = []
        for line in page.iter(_ALTO_NS + "TextLine"):
            string = line.find(_ALTO_NS + "String")
            if string is None:
                continue
            refs = (string.get("STYLEREFS") or "").split()
            if refs and refs[0] in sizes_by_id:
                page_sizes.append(sizes_by_id[refs[0]])
        if not page_sizes:
            continue
        modal = statistics.mode(page_sizes)
        if modal <= 0:
            continue
        ratios.extend(
            size / modal for size in page_sizes if abs(size / modal - 1.0) <= _BODY_BAND
        )
    return ratios


def summarize(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--corpus", default="dnb-toc-only", help="Corpus to measure (default: dnb-toc-only)")
    parser.add_argument("--pdfalto-bin", help="Path to the pdfalto binary (see pdfalto_runner.py)")
    args = parser.parse_args()

    if args.corpus not in list_corpora(include_toc_only=True):
        print(f"Corpus '{args.corpus}' not found (or has no manifest.json).")
        return 1

    pdfalto_bin = resolve_pdfalto_binary(args.pdfalto_bin)
    cdir = corpus_dir(args.corpus)
    cache_dir = cdir / ".layout-cache"

    all_contrast: list[float] = []
    all_dispersion: list[float] = []
    for book in load_manifest_books(args.corpus):
        pdf_path = cdir / book["filename"]
        if not pdf_path.exists():
            print(f"[skip] {book['filename']}: PDF not present locally")
            continue
        alto_path = ensure_alto_xml(pdf_path, cache_dir, pdfalto_bin)
        all_contrast.extend(contrast_ratios(alto_path))
        all_dispersion.extend(body_line_dispersion_ratios(alto_path))

    contrast_stats = summarize(all_contrast)
    dispersion_stats = summarize(all_dispersion)

    print(f"\n=== {args.corpus}: real-scan measurements vs. alto_scan_noise.py constants ===\n")
    print(f"Title/body contrast ratio (font_size_max_ratio, n={contrast_stats.get('count', 0)}):")
    print(f"  measured: {contrast_stats}")
    print(f"  current _CONTRAST_ALPHA range: {_CONTRAST_ALPHA}")
    print(f"\nBody-line font-size dispersion (ratio to page modal size, within +/-{_BODY_BAND:.0%}, n={dispersion_stats.get('count', 0)}):")
    print(f"  measured: {dispersion_stats}")
    print(f"  current _FONT_JITTER range: {_FONT_JITTER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_measure_dnb_scan_noise_stats.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add evaluation/scripts/measure_dnb_scan_noise_stats.py tests/test_measure_dnb_scan_noise_stats.py
git commit -m "feat: add measure_dnb_scan_noise_stats.py for real-scan noise calibration"
```

---

### Task 6: Smoke-test the measurement script against the real acquired corpus

**Files:** none (read-only run against Task 4's committed data; writes only to the gitignored `.layout-cache/`)

Requires the `pdfalto` binary. It is NOT on `PATH` — pass its absolute path explicitly. **If running inside a git worktree** (e.g. `.claude/worktrees/<name>/`), `evaluation/CLAUDE.md`'s documented relative path `../pdfalto/pdfalto` resolves relative to the *worktree* directory and will be wrong — use the absolute path instead.

- [ ] **Step 1: Locate the pdfalto binary**

```bash
ls -la /Users/cboulanger/Code/pdfalto/pdfalto
```

If this file doesn't exist on the machine running this task, skip Steps 2-3 below, note in the task's completion summary that this step was skipped (no pdfalto binary available in this environment), and move on — Task 5's unit tests already cover this script's actual logic.

- [ ] **Step 2: Run it for real**

```bash
uv run python evaluation/scripts/measure_dnb_scan_noise_stats.py --pdfalto-bin /Users/cboulanger/Code/pdfalto/pdfalto
```

Expected: no errors, a printed `=== dnb-toc-only: real-scan measurements vs. alto_scan_noise.py constants ===` table with `n` equal to the total non-empty-page count across Task 4's small smoke-test batch (a handful of pages per book — this is a tiny preview, not the calibration-grade sample size the design spec's "Decision criteria" calls for; that requires the user's own later `--from-dump` run).

- [ ] **Step 3: Record the output**

Copy the full printed table into this task's completion notes (do not edit `evaluation/RESULTS.md` yet — this sample is too small to be a meaningful calibration update; that's explicitly a follow-up for after the user runs `--from-dump` for real, see the plan header).

- [ ] **Step 4: Confirm nothing else broke**

```bash
uv run pytest -q
```

Expected: PASS.

- [ ] **Step 5: Commit (only if something changed)**

`.layout-cache/` is gitignored, so this step likely has nothing to commit. Run `git status` to confirm; if clean, skip the commit.

---

### Task 7: `evaluation/scripts/README.md` — `--help` dump entries

**Files:**
- Modify: `evaluation/scripts/README.md`

- [ ] **Step 1: Generate both `--help` dumps**

```bash
uv run python evaluation/scripts/fetch_dnb_toc_corpus.py --help
uv run python evaluation/scripts/measure_dnb_scan_noise_stats.py --help
```

- [ ] **Step 2: Insert alphabetically**

`evaluation/scripts/README.md` is alphabetical by script name (confirmed: `add_toc_ground_truth.py`, `build_crossref_gt_ground_truth.py`, `clean_scanned_pdf.py`, ...). Insert a `## fetch_dnb_toc_corpus.py` section (immediately after `## fetch_evaluation_pdfs.py` if that entry sorts earlier, otherwise in correct alpha order relative to neighboring entries) and a `## measure_dnb_scan_noise_stats.py` section, each following the existing format:

```markdown
## fetch_dnb_toc_corpus.py

Acquires real DNB-scanned table-of-contents PDFs via the lobid-resources
API into evaluation/corpus/dnb-toc-only/.

```
<exact --help output from Step 1>
```
```

```markdown
## measure_dnb_scan_noise_stats.py

Measures real font-size contrast/dispersion from the dnb-toc-only corpus's
ALTO to calibrate alto_scan_noise.py's synthetic constants.

```
<exact --help output from Step 1>
```
```

Also update this file's opening paragraph if it enumerates script names anywhere that would need the two new ones added (check for an itemized list near the top; if the intro is generic prose with no exhaustive list, no change needed there).

- [ ] **Step 3: Verify no other script's entry shifted unexpectedly**

Run: `grep -c "^## " evaluation/scripts/README.md` before and after — should increase by exactly 2.

- [ ] **Step 4: Commit**

```bash
git add evaluation/scripts/README.md
git commit -m "docs: add --help reference entries for the two new DNB corpus scripts"
```

---

## Follow-ups explicitly deferred to the user (not part of this plan, per the design spec's "Out of scope")

- Running `fetch_dnb_toc_corpus.py --from-dump` for real (hours-long, ~21.5GB) to grow `dnb-toc-only/` to the "few hundred books" scale the design spec's "Decision criteria" calls for.
- Re-running `measure_dnb_scan_noise_stats.py` against that larger corpus and, if the numbers warrant it, updating `alto_scan_noise.py`'s `_CONTRAST_ALPHA`/`_FONT_JITTER` constants and writing the comparison up in `evaluation/RESULTS.md`.
- Assembling and publishing the full acquired PDF set as a Zenodo dataset.
- Building an `<id>.expected.json` generator for this corpus (feasible for `citation_pages` per the design spec, not built here).
- **The design spec's Section 3 "second, purely diagnostic use"** — running the already-trained page-local layout classifier against this corpus's page features as an out-of-population recall check. Deliberately not built in this plan: `evaluate_layout_toc_classifier.py` has no persisted model artifact (it retrains fresh every run inside its own LOBO evaluation loop), so this would require factoring a reusable train/score path out of that script's internals — real added scope for what the design spec itself calls "a sanity signal only, not a new training input" and does not gate in its "Decision criteria." Worth a follow-up spec/task of its own once `dnb-toc-only/` has real scale (a handful of smoke-test pages isn't a meaningful out-of-population sample anyway).
