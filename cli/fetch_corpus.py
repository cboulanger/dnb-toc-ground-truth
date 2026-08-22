#!/usr/bin/env python3
"""Acquires real DNB-scanned table-of-contents PDFs via the lobid-resources
API (lobid.org/resources) into data/corpus/pilot/ -- see
docs/superpowers/specs/2026-08-14-dnb-toc-corpus-acquisition-design.md.

Every acquired record's PDF is a CC0-licensed, DNB-digitized TOC scan
(the "Kataloganreicherung" program), never the surrounding book, so this
corpus is deliberately shaped differently from open-access/copyrighted-scans
(see manifest_entry_from_record below and the design spec) -- no
extraction_type/embedded_toc/oa/download_url fields, and no
<id>.expected.json is produced by this script.

Two acquisition modes, because tableOfContents is not queryable
server-side in lobid-resources (confirmed empirically -- see design spec):

    uv run python cli/fetch_corpus.py \\
        --isbns-file /tmp/isbns.txt --limit 20
    uv run python cli/fetch_corpus.py \\
        --from-dump --limit 500

--from-dump streams the full weekly lobid-resources JSON-Lines dump
(~21.5GB gzip as of 2026-08-14, one bibliographic record per line) and
filters client-side -- there is no per-request cost per scanned record,
only per acquired match (one PDF download + a rate-limit sleep), but a
full scan is still an hours-long, many-GB operation against a shared
public service. Run it deliberately, not as part of routine development.

manifest.json is the only file this script writes to git -- PDFs stay
gitignored, same as every other corpus file. See "PDFs are never
committed" in the design spec: this corpus's full PDF set is meant for a
separate Zenodo dataset upload, not this repo.
"""

import argparse
import gzip
import json
import time
from pathlib import Path
from typing import Iterator, Optional
from urllib.parse import urlsplit

import httpx

from dnb_toc_ground_truth import corpus, crossref, inference

_DUMP_URL_DEFAULT = "https://lobid.org/download/dumps/lobid-resources/latestLobidResources.jsonl.gz"
_SEARCH_URL = "https://lobid.org/resources/search"

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
    typed as an EditedVolume (an edited collection/Festschrift/reference
    volume -- the book template this project's evaluation corpus already
    targets, per the design spec) and carries a non-empty tableOfContents
    array. Deliberately requires "EditedVolume" specifically, not just
    "Book": confirmed live (2026-08-15) that lobid-resources types
    single-author monographs and theses as bare ["...", "Book"] too --
    e.g. isbn:9783844019384 (["BibliographicResource", "Thesis", "Book"])
    and isbn:9783868674095 (["BibliographicResource", "Book"], a
    Lehrbuch/textbook) -- so the original any(Book-or-EditedVolume) filter
    let single-author and textbook TOCs into a corpus meant to target
    edited-volume TOC layouts specifically."""
    types = record.get("type") or []
    if "EditedVolume" not in types:
        return False
    return bool(record.get("tableOfContents"))


def _toc_download_url(record: dict) -> Optional[str]:
    """A record's tableOfContents[].id, or None. Rejects a schemeless/
    relative URL (e.g. "/www.example.edu/toc.pdf" -- found live in the
    lobid-resources dump for at least one record, 2026-08-15) rather than
    returning it: httpx.Client.get() raises ValueError for such a URL
    (not an httpx.HTTPError subclass), which _acquire_record's retry
    handling doesn't catch -- an uncaught ValueError previously killed an
    entire --from-dump run outright instead of skipping the one bad
    record. Treating it as "no toc url" here reuses the existing skip
    path instead."""
    for entry in record.get("tableOfContents") or []:
        url = entry.get("id")
        if url and urlsplit(url).scheme in ("http", "https"):
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


def _record_api_url(record: dict) -> Optional[str]:
    """The stable, directly-fetchable URL for re-downloading this
    record's full lobid-resources data on demand (see
    manifest_entry_from_record) -- the record's own lobid URI with the
    "#!" JSON-LD fragment stripped and format=json appended so it
    resolves to plain JSON with no Accept header needed."""
    record_id = (record.get("id") or "").rstrip("#!")
    if not record_id:
        return None
    return f"{record_id}?format=json"


def manifest_entry_from_record(record: dict, filename: str) -> dict:
    """Builds this corpus's manifest.json book entry. The full lobid
    record is NOT embedded here -- it used to be, under a "lobid_record"
    key, but at real corpus scale that bloated manifest.json into an
    unreviewable multi-hundred-thousand-line file (~1,000 lines per book,
    mostly library holdings data ("hasItem") no code reads). lobid_url
    points back to the same data instead, re-fetchable on demand;
    _acquire_record separately writes the full record to
    .lobid-cache/<key>.lobid.json (gitignored, like the PDF) for anything
    that wants it locally without a network round-trip -- kept in its own
    subdirectory, out of the corpus dir's top level, the same way
    .ocr-cache/ and .layout-cache/ already are for other corpora."""
    return {
        "filename": filename,
        "title": record.get("title") or "",
        "language": _record_language(record),
        "doi": _record_doi(record),
        "toc_download_url": _toc_download_url(record),
        "license": "CC0-1.0",
        "license_source": "dnb",
        "lobid_url": _record_api_url(record),
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
    manifest_path: Path,
    client: httpx.Client,
    rate_limit_seconds: float,
    seen_keys: set[str],
    contact_email: Optional[str] = None,
    crossref_cache_dir: Optional[Path] = None,
    min_chapters: int = crossref.DEFAULT_MIN_CHAPTERS_FOR_EVAL,
) -> Optional[str]:
    """Downloads one matched record's TOC PDF and appends its manifest
    entry. Returns None iff a new book was acquired (so callers can count
    toward --limit); otherwise a short skip-reason string -- a non-match
    ("no matching type/toc"), an already-acquired key ("already acquired"),
    a record with no resolvable TOC URL ("no toc url"), or a network error
    downloading the PDF ("download failed: <exc>"). A download failure is
    caught here rather than propagated so one bad record (a dead link, a
    dropped connection, a transient 5xx) skips that record instead of
    aborting the whole run -- see the module docstring on why that matters
    for --from-dump's hours-long scans."""
    if not _record_matches(record):
        return "no matching type/toc"
    key = _record_key(record)
    if key in seen_keys:
        return "already acquired"
    toc_url = _toc_download_url(record)
    if not toc_url:
        return "no toc url"
    filename = f"{key}.pdf"
    try:
        response = client.get(toc_url)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        return f"download failed: {exc}"
    corpus.pdf_dir().mkdir(parents=True, exist_ok=True)
    (corpus.pdf_dir() / filename).write_bytes(response.content)
    corpus.lobid_cache_dir().mkdir(parents=True, exist_ok=True)
    (corpus.lobid_cache_dir() / f"{key}.lobid.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    entry = manifest_entry_from_record(record, filename)
    isbn = crossref.normalize_isbn(key)
    if isbn:
        crossref_data = crossref.fetch_crossref_book(
            isbn, client, contact_email, crossref_cache_dir or corpus.crossref_cache_dir(),
        )
        if crossref_data.doi:
            entry["doi"] = crossref_data.doi
        crossref.write_evaluation_entry(isbn, crossref_data, corpus.evaluation_dir(), min_chapters)
    _append_book(manifest_path, entry)
    seen_keys.add(key)
    print(f"[fetch] {filename} <- {toc_url}")
    time.sleep(rate_limit_seconds)
    return None


def _run_isbns_file(
    args: argparse.Namespace, manifest_path: Path, client: httpx.Client,
    contact_email: Optional[str], crossref_cache_dir: Path, min_chapters: int,
) -> None:
    seen_keys = _load_existing_keys(manifest_path)
    acquired = 0
    for isbn in _read_isbns_file(Path(args.isbns_file)):
        if args.limit is not None and acquired >= args.limit:
            break
        try:
            record = _search_by_isbn(isbn, client)
        except httpx.HTTPError as exc:
            print(f"[skip] {isbn}: lookup failed: {exc}")
            continue
        if record is None:
            print(f"[skip] {isbn}: no lobid-resources record found")
            continue
        reason = _acquire_record(
            record, manifest_path, client, args.rate_limit_seconds, seen_keys,
            contact_email, crossref_cache_dir, min_chapters,
        )
        if reason is None:
            acquired += 1
        else:
            print(f"[skip] {isbn}: {reason}")
    print(f"Acquired {acquired} new book(s).")


def _scan_and_acquire(
    records: Iterator[dict],
    manifest_path: Path,
    client: httpx.Client,
    rate_limit_seconds: float,
    limit: Optional[int],
    seen_keys: set[str],
    acquired_so_far: int,
    contact_email: Optional[str],
    crossref_cache_dir: Path,
    min_chapters: int,
) -> tuple[int, int]:
    """Consumes records from the given iterator, acquiring matches until
    either the iterator is exhausted or acquired_so_far plus newly
    acquired reaches limit -- stops consuming immediately once the limit
    is hit, rather than draining the rest of the iterator. Returns
    (records_scanned_this_call, newly_acquired_this_call). Pulled out of
    _run_from_dump as a pure, retry-agnostic unit so a dropped dump
    connection can be retried by simply calling this again with a fresh
    iterator and an updated acquired_so_far -- see _run_from_dump."""
    scanned = 0
    acquired = 0
    for record in records:
        scanned += 1
        if scanned % 100_000 == 0:
            print(f"[scan] {scanned:,} records scanned this attempt, {acquired_so_far + acquired} acquired so far")
        if limit is not None and acquired_so_far + acquired >= limit:
            break
        reason = _acquire_record(
            record, manifest_path, client, rate_limit_seconds, seen_keys,
            contact_email, crossref_cache_dir, min_chapters,
        )
        if reason is None:
            acquired += 1
        elif reason.startswith("download failed"):
            # A network error is worth surfacing even during a --from-dump
            # scan (unlike the ordinary "record doesn't match"/"already
            # acquired"/"no toc url" skips, which are far too common across
            # millions of scanned records to print without spamming the
            # run's output).
            print(f"[skip] {_record_key(record)}: {reason}")
    return scanned, acquired


def _run_from_dump(
    args: argparse.Namespace, manifest_path: Path, client: httpx.Client,
    contact_email: Optional[str], crossref_cache_dir: Path, min_chapters: int,
) -> None:
    seen_keys = _load_existing_keys(manifest_path)
    acquired = 0
    attempt = 0
    while True:
        try:
            scanned, newly = _scan_and_acquire(
                _iter_dump_records(args.dump_url, client), manifest_path, client,
                args.rate_limit_seconds, args.limit, seen_keys, acquired,
                contact_email, crossref_cache_dir, min_chapters,
            )
            acquired += newly
            if args.limit is not None and acquired >= args.limit:
                print(f"Acquired {acquired} new book(s) (limit reached).")
                return
            print(f"Scanned {scanned:,} records this attempt (dump exhausted), acquired {acquired} new book(s) total.")
            return
        except httpx.HTTPError as exc:
            attempt += 1
            if attempt > args.max_retries:
                print(f"[error] dump stream failed {attempt} time(s), giving up: {exc}")
                raise
            backoff = min(2 ** attempt, 60)
            print(
                f"[retry {attempt}/{args.max_retries}] dump stream dropped ({exc}); "
                f"reconnecting in {backoff}s and rescanning from the start "
                f"(already-acquired books are skipped via seen_keys, so this only costs time)"
            )
            time.sleep(backoff)


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
    parser.add_argument(
        "--max-retries", type=int, default=5,
        help="For --from-dump: how many times to reconnect and rescan after a dropped connection before giving up (default: 5)",
    )
    parser.add_argument(
        "--manifest-path", type=Path, default=None,
        help=(
            "Override where manifest.json entries are read from and appended to "
            "(default: data/corpus/pilot/manifest.json). PDFs and "
            ".lobid-cache/ always go to the real corpus directory regardless -- "
            "only the tracked manifest write is redirectable, e.g. to a scratch "
            "copy that gets merged into the real manifest.json once a long run "
            "finishes, without touching the committed file mid-run."
        ),
    )
    parser.add_argument(
        "--contact-email", default=None,
        help="Crossref polite-pool contact email (default: config file's \"contact_email\")",
    )
    parser.add_argument(
        "--crossref-cache-dir", type=Path, default=None,
        help="Override where Crossref DOI/chapter data is cached (default: data/corpus/pilot/.crossref-cache/)",
    )
    parser.add_argument(
        "--config-file", type=Path, default=Path(inference.DEFAULT_CONFIG_FILENAME),
        help=f"Path to the config file (default: {inference.DEFAULT_CONFIG_FILENAME})",
    )
    parser.add_argument(
        "--corpus", default=None,
        help=f"Corpus to operate on (default: config file's \"corpus\", or {corpus.DEFAULT_CORPUS_NAME!r})",
    )
    parser.add_argument(
        "--min-chapters", type=int, default=crossref.DEFAULT_MIN_CHAPTERS_FOR_EVAL,
        help=(
            "Minimum page-numbered Crossref chapters a book needs before its evaluation-corpus "
            f"entry is written (default: {crossref.DEFAULT_MIN_CHAPTERS_FOR_EVAL})"
        ),
    )
    args = parser.parse_args()

    config = inference.load_config(args.config_file)
    corpus.set_corpus(args.corpus or config.get("corpus") or corpus.DEFAULT_CORPUS_NAME, create=True)
    corpus.corpus_dir().mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest_path or corpus.manifest_path()
    _ensure_manifest_shell(manifest_path)
    contact_email = args.contact_email or config.get("contact_email")
    crossref_cache_dir = args.crossref_cache_dir or corpus.crossref_cache_dir()

    with httpx.Client(
        follow_redirects=True,
        timeout=httpx.Timeout(connect=30.0, read=120.0, write=30.0, pool=30.0),
    ) as client:
        if args.isbns_file:
            _run_isbns_file(args, manifest_path, client, contact_email, crossref_cache_dir, args.min_chapters)
        else:
            _run_from_dump(args, manifest_path, client, contact_email, crossref_cache_dir, args.min_chapters)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
