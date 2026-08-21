# DNB TOC corpus acquisition: corrections Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix three real problems found while running `evaluation/scripts/fetch_dnb_toc_corpus.py --from-dump` for real against the live 21.5GB lobid-resources dump (2026-08-15): (1) it crashed on a dropped connection with no retry, (2) the `lobid_record` field embedded verbatim in `manifest.json` makes that file unreviewable at real scale (293,904 lines for 305 books), (3) `_record_matches`'s `type` filter is too broad and let single-author/thesis/textbook records into a corpus meant to target edited-volume TOC layouts specifically. Then migrate the already-acquired 305-book batch to the corrected schema, purge the wrongly-matched entries, and re-run the bulk acquisition.

**Architecture:** `_record_matches` narrows to `"EditedVolume" in type` (confirmed live against two of the wrongly-matched ISBNs -- both typed plain `["...", "Book"]`, no `EditedVolume`). `manifest_entry_from_record` drops the inline `lobid_record` field in favor of a `lobid_url` (a directly re-fetchable API URL derived from the record's own `id`); `_acquire_record` separately writes the full record to a new gitignored `<key>.lobid.json` file alongside the PDF. `_run_from_dump`'s scan-and-acquire loop is extracted into a pure, testable `_scan_and_acquire` helper; the thin outer loop in `_run_from_dump` catches `httpx.HTTPError`, backs off, and reconnects (full dump rescan from the start -- `seen_keys` prevents re-downloading anything already acquired, so a rescan just costs time, not correctness). A new one-off script, `evaluation/scripts/migrate_dnb_toc_lobid_storage.py`, converts the already-committed 305-book manifest to the new schema and purges every entry that isn't actually `EditedVolume`-typed.

**Tech Stack:** Same as the original plan (`docs/superpowers/plans/2026-08-14-dnb-toc-corpus-acquisition.md`) -- Python 3.12, httpx, stdlib gzip/json, unittest.

**Root cause, confirmed live (2026-08-15):**

```
$ curl -s "https://lobid.org/resources/search?q=isbn:9783844019384&format=json" | ...
type: ['BibliographicResource', 'Thesis', 'Book']       # single-author thesis
$ curl -s "https://lobid.org/resources/search?q=isbn:9783868674095&format=json" | ...
type: ['BibliographicResource', 'Book']                  # Lehrbuch/textbook, no EditedVolume
```

`_record_matches`'s original filter (`any(t in ("Book", "EditedVolume") for t in types)`) accepted both, because nearly every lobid-resources book record carries bare `"Book"` in its `type` list regardless of whether it's also an edited collection -- the `"Book"` branch of that `any(...)` check was doing nothing to narrow the match, since `"EditedVolume"` records already also carry `"Book"` (confirmed in the original design spec's own sample: `["BibliographicResource","EditedVolume","Book"]`). The fix is to drop the `"Book"`-alone branch entirely.

**Where this plan's own worktree work ends and real, main-checkout-only execution begins:** Tasks 1-5 happen in an isolated worktree with unit tests against synthetic fixtures -- no real network, no real file deletion. Tasks 6-7 run for real against the actual 305-book batch and the actual live dump, and **must run in the main checkout, not a worktree** -- the original bulk-acquisition run's 9 smoke-test PDFs were silently lost when their worktree was removed after merging (gitignored files only ever existed in that worktree's own directory, not the main checkout, and `git worktree remove` deletes everything in the worktree directory including untracked/ignored files). Don't repeat that mistake: merge Tasks 1-5 to main first, then do Tasks 6-7 directly in the main checkout.

---

## File Structure

- Modify: `evaluation/scripts/fetch_dnb_toc_corpus.py` -- `_record_matches`, `manifest_entry_from_record`, `_acquire_record`, new `_record_api_url`, new `_scan_and_acquire`, rewritten `_run_from_dump`, new `--max-retries` flag.
- Modify: `tests/test_fetch_dnb_toc_corpus.py` -- new/updated tests for all of the above.
- Modify: `evaluation/.gitignore` -- add `*.lobid.json`.
- Create: `evaluation/scripts/migrate_dnb_toc_lobid_storage.py` -- one-off migration+purge script (has a `--dry-run` flag; no dedicated pytest suite, matching this repo's convention for one-time reconciliation scripts like `build_crossref_gt_ground_truth.py` -- verified instead by dry-run inspection and a full manual run against the real data in Task 6).
- Modify: `docs/superpowers/specs/2026-08-14-dnb-toc-corpus-acquisition-design.md` -- correct the manifest schema and matching-criterion sections.
- Modify (Task 6, real data): `evaluation/corpus/dnb-toc-only/manifest.json`, plus new `<key>.lobid.json` files (gitignored) and deletions of purged PDFs.
- Modify (Task 7, real data): `evaluation/corpus/dnb-toc-only/manifest.json` grows further via a real `--from-dump` run.

---

### Task 1: Narrow `_record_matches` to `EditedVolume` only

**Files:**
- Modify: `evaluation/scripts/fetch_dnb_toc_corpus.py` (the `_record_matches` function)
- Test: `tests/test_fetch_dnb_toc_corpus.py` (the `TestRecordMatches` class)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_fetch_dnb_toc_corpus.py`, inside `class TestRecordMatches`:

```python
    def test_rejects_plain_book_without_edited_volume(self):
        # Confirmed live 2026-08-15 (isbn:9783868674095): a Lehrbuch
        # typed just ["BibliographicResource", "Book"] -- no
        # "EditedVolume" -- must NOT match even though it has a TOC and
        # "Book" is technically in its type list.
        record = {**_SAMPLE_RECORD, "type": ["BibliographicResource", "Book"]}
        self.assertFalse(_record_matches(record))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_fetch_dnb_toc_corpus.py::TestRecordMatches -v`
Expected: FAIL (`test_rejects_plain_book_without_edited_volume` fails -- the current filter accepts bare `"Book"`)

- [ ] **Step 3: Implement**

Replace `_record_matches` in `evaluation/scripts/fetch_dnb_toc_corpus.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_fetch_dnb_toc_corpus.py -v`
Expected: PASS (all tests, including the pre-existing `TestRecordMatches` tests -- `_SAMPLE_RECORD`'s type already includes `"EditedVolume"`, so `test_matches_book_with_toc` still passes unchanged)

- [ ] **Step 5: Commit**

```bash
git add evaluation/scripts/fetch_dnb_toc_corpus.py tests/test_fetch_dnb_toc_corpus.py
git commit -m "fix: require EditedVolume type specifically in fetch_dnb_toc_corpus.py"
```

---

### Task 2: Replace inline `lobid_record` with `lobid_url` + a separate `.lobid.json` file

**Files:**
- Modify: `evaluation/scripts/fetch_dnb_toc_corpus.py` (`manifest_entry_from_record`, `_acquire_record`; new `_record_api_url`)
- Modify: `evaluation/.gitignore`
- Test: `tests/test_fetch_dnb_toc_corpus.py`

- [ ] **Step 1: Write the failing tests**

Replace the existing `TestManifestEntryFromRecord` class in `tests/test_fetch_dnb_toc_corpus.py` with:

```python
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
```

Add `_record_api_url` to the import list at the top of the file (alongside the existing `manifest_entry_from_record` import).

Then update `TestAcquireRecord`'s existing "new-record success" test case (find it -- the one asserting the PDF was written and the manifest entry appended) to also assert the `.lobid.json` side file was written correctly:

```python
        lobid_path = cdir / f"{key}.lobid.json"  # use whatever key variable that test already has
        self.assertTrue(lobid_path.exists())
        self.assertEqual(json.loads(lobid_path.read_text(encoding="utf-8")), _SAMPLE_RECORD)
```

(Adapt variable names to whatever that existing test method already uses for `cdir`/`key`/the sample record -- read the current `TestAcquireRecord` class first to match its exact style rather than guessing.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_fetch_dnb_toc_corpus.py -v`
Expected: FAIL (`_record_api_url` doesn't exist yet; `manifest_entry_from_record` still returns `lobid_record`; no `.lobid.json` gets written)

- [ ] **Step 3: Implement**

Add `_record_api_url`, right after `_record_doi` in `evaluation/scripts/fetch_dnb_toc_corpus.py`:

```python
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
```

Replace `manifest_entry_from_record`:

```python
def manifest_entry_from_record(record: dict, filename: str) -> dict:
    """Builds this corpus's manifest.json book entry. The full lobid
    record is NOT embedded here -- it used to be, under a "lobid_record"
    key, but at real corpus scale that bloated manifest.json into an
    unreviewable multi-hundred-thousand-line file (~1,000 lines per book,
    mostly library holdings data ("hasItem") no code reads). lobid_url
    points back to the same data instead, re-fetchable on demand;
    _acquire_record separately writes the full record to
    <key>.lobid.json (gitignored, like the PDF) for anything that wants
    it locally without a network round-trip."""
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
```

In `_acquire_record`, right after the PDF is written (`(cdir / filename).write_bytes(response.content)`), add:

```python
    (cdir / f"{key}.lobid.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
```

(before the `_append_book(...)` call, so both side effects happen before the manifest entry is recorded).

In `evaluation/.gitignore`, add a new line:

```
*.lobid.json
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_fetch_dnb_toc_corpus.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add evaluation/scripts/fetch_dnb_toc_corpus.py tests/test_fetch_dnb_toc_corpus.py evaluation/.gitignore
git commit -m "fix: replace inline lobid_record with a gitignored .lobid.json side file"
```

---

### Task 3: Retry/reconnect resilience for `--from-dump`'s streaming connection

**Files:**
- Modify: `evaluation/scripts/fetch_dnb_toc_corpus.py` (`_run_from_dump`; new `_scan_and_acquire`; `main`'s argparse)
- Test: `tests/test_fetch_dnb_toc_corpus.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_fetch_dnb_toc_corpus.py` (add `_scan_and_acquire` to the import list):

```python
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

        with tempfile.TemporaryDirectory() as tmp:
            cdir = Path(tmp)
            manifest_path = cdir / "manifest.json"
            _ensure_manifest_shell(manifest_path)
            client = Mock()
            client.get.return_value = _json_response({})  # PDF download response; content below
            client.get.return_value.content = b"%PDF-fake"
            scanned, newly_acquired = _scan_and_acquire(
                _record_stream(), cdir, manifest_path, client,
                rate_limit_seconds=0, limit=2, seen_keys=set(), acquired_so_far=1,
            )
        self.assertEqual(newly_acquired, 1)
        self.assertEqual(scanned, 2)

    def test_consumes_whole_iterator_when_no_limit(self):
        records = [
            {**_SAMPLE_RECORD, "isbn": ["1111111111111"]},
            {**_SAMPLE_RECORD, "isbn": ["2222222222222"]},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            cdir = Path(tmp)
            manifest_path = cdir / "manifest.json"
            _ensure_manifest_shell(manifest_path)
            client = Mock()
            client.get.return_value = _json_response({})
            client.get.return_value.content = b"%PDF-fake"
            scanned, newly_acquired = _scan_and_acquire(
                iter(records), cdir, manifest_path, client,
                rate_limit_seconds=0, limit=None, seen_keys=set(), acquired_so_far=0,
            )
        self.assertEqual(newly_acquired, 2)
        self.assertEqual(scanned, 2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_fetch_dnb_toc_corpus.py::TestScanAndAcquire -v`
Expected: FAIL with `ImportError`/`NameError` (`_scan_and_acquire` doesn't exist yet)

- [ ] **Step 3: Implement**

Replace `_run_from_dump` in `evaluation/scripts/fetch_dnb_toc_corpus.py` with a pure helper plus a thin retrying wrapper:

```python
def _scan_and_acquire(
    records: Iterator[dict],
    cdir: Path,
    manifest_path: Path,
    client: httpx.Client,
    rate_limit_seconds: float,
    limit: Optional[int],
    seen_keys: set[str],
    acquired_so_far: int,
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
        reason = _acquire_record(record, cdir, manifest_path, client, rate_limit_seconds, seen_keys)
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


def _run_from_dump(args: argparse.Namespace, cdir: Path, manifest_path: Path, client: httpx.Client) -> None:
    seen_keys = _load_existing_keys(manifest_path)
    acquired = 0
    attempt = 0
    while True:
        try:
            scanned, newly = _scan_and_acquire(
                _iter_dump_records(args.dump_url, client), cdir, manifest_path, client,
                args.rate_limit_seconds, args.limit, seen_keys, acquired,
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
```

In `main()`, add a new argparse option (near `--rate-limit-seconds`):

```python
    parser.add_argument(
        "--max-retries", type=int, default=5,
        help="For --from-dump: how many times to reconnect and rescan after a dropped connection before giving up (default: 5)",
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_fetch_dnb_toc_corpus.py -v`
Expected: PASS (every test, including the pre-existing ones -- `_run_from_dump` itself stays intentionally untested per this file's documented convention for network-calling orchestration, but the new `_scan_and_acquire` core is covered)

- [ ] **Step 5: Commit**

```bash
git add evaluation/scripts/fetch_dnb_toc_corpus.py tests/test_fetch_dnb_toc_corpus.py
git commit -m "fix: retry/reconnect --from-dump on a dropped connection instead of crashing"
```

---

### Task 4: One-off migration + purge script

**Files:**
- Create: `evaluation/scripts/migrate_dnb_toc_lobid_storage.py`

No dedicated pytest suite for this one (matches this repo's convention for one-time reconciliation scripts, e.g. `build_crossref_gt_ground_truth.py` -- see `evaluation/scripts/README.md`). Verified instead by `--dry-run` inspection against a tiny synthetic fixture here, then a real dry-run and real run against the actual data in Task 6.

- [ ] **Step 1: Write the script**

Create `evaluation/scripts/migrate_dnb_toc_lobid_storage.py`:

```python
#!/usr/bin/env python3
"""One-time reconciliation: migrates evaluation/corpus/dnb-toc-only/manifest.json
from its original schema (a "lobid_record" field embedding the full
lobid-resources record verbatim) to the corrected one (a "lobid_url"
field plus a separate <key>.lobid.json side file) -- see
docs/superpowers/plans/2026-08-15-dnb-toc-corpus-corrections.md Task 2.

While rewriting each entry, also drops any book whose lobid_record isn't
actually EditedVolume-typed (see Task 1 of that plan -- the original
_record_matches filter was too broad and let single-author/thesis/
textbook records in). A dropped book's PDF and any already-written
.lobid.json are deleted; its manifest entry is removed entirely.

Usage:
    uv run python evaluation/scripts/migrate_dnb_toc_lobid_storage.py --dry-run
    uv run python evaluation/scripts/migrate_dnb_toc_lobid_storage.py
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from evaluation.harness import corpus_dir
from evaluation.scripts.fetch_dnb_toc_corpus import _record_api_url, _record_matches

_CORPUS_NAME = "dnb-toc-only"


def migrate(cdir: Path, dry_run: bool) -> None:
    manifest_path = cdir / "manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))

    kept = []
    purged = []
    for book in data["books"]:
        key = Path(book["filename"]).stem
        record = book.get("lobid_record")
        if record is None:
            # Already migrated (no lobid_record field) -- leave as-is.
            kept.append(book)
            continue

        if not _record_matches(record):
            purged.append((key, book.get("title", "")))
            if not dry_run:
                (cdir / book["filename"]).unlink(missing_ok=True)
                (cdir / f"{key}.lobid.json").unlink(missing_ok=True)
            continue

        new_book = {k: v for k, v in book.items() if k != "lobid_record"}
        new_book["lobid_url"] = _record_api_url(record)
        kept.append(new_book)
        if not dry_run:
            (cdir / f"{key}.lobid.json").write_text(
                json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
            )

    print(f"{'[DRY RUN] ' if dry_run else ''}Kept {len(kept)} book(s), purged {len(purged)}:")
    for key, title in purged:
        print(f"  - {key}: {title}")

    if not dry_run:
        data["books"] = kept
        manifest_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Report what would change without writing or deleting anything")
    args = parser.parse_args()
    migrate(corpus_dir(_CORPUS_NAME), args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Build a tiny synthetic fixture and dry-run it**

```bash
uv run python -c "
import json
from pathlib import Path
tmp = Path('/tmp/dnb_migrate_test')
tmp.mkdir(exist_ok=True)
(tmp / 'manifest.json').write_text(json.dumps({
    'toc_only': True,
    'books': [
        {
            'filename': 'keep123.pdf', 'title': 'Edited Vol', 'language': 'de', 'doi': None,
            'toc_download_url': 'https://example.org/keep123.pdf', 'license': 'CC0-1.0', 'license_source': 'dnb',
            'lobid_record': {'id': 'http://lobid.org/resources/keep123#!', 'type': ['BibliographicResource', 'EditedVolume', 'Book'], 'tableOfContents': [{'id': 'https://example.org/keep123.pdf'}]},
        },
        {
            'filename': 'purge456.pdf', 'title': 'Single Author Book', 'language': 'de', 'doi': None,
            'toc_download_url': 'https://example.org/purge456.pdf', 'license': 'CC0-1.0', 'license_source': 'dnb',
            'lobid_record': {'id': 'http://lobid.org/resources/purge456#!', 'type': ['BibliographicResource', 'Book'], 'tableOfContents': [{'id': 'https://example.org/purge456.pdf'}]},
        },
    ],
}))
(tmp / 'keep123.pdf').write_bytes(b'%PDF-fake')
(tmp / 'purge456.pdf').write_bytes(b'%PDF-fake')
print('fixture ready at', tmp)
"
uv run python -c "
from pathlib import Path
from evaluation.scripts.migrate_dnb_toc_lobid_storage import migrate
migrate(Path('/tmp/dnb_migrate_test'), dry_run=True)
"
```

Expected output: `Kept 1 book(s), purged 1:` followed by `  - purge456: Single Author Book`. Confirm with `ls /tmp/dnb_migrate_test/` that nothing was actually written or deleted (both PDFs and the original manifest.json untouched -- dry-run made no filesystem changes).

- [ ] **Step 3: Run it for real against the fixture and verify the result**

```bash
uv run python -c "
from pathlib import Path
from evaluation.scripts.migrate_dnb_toc_lobid_storage import migrate
migrate(Path('/tmp/dnb_migrate_test'), dry_run=False)
"
uv run python -c "
import json
data = json.load(open('/tmp/dnb_migrate_test/manifest.json'))
books = data['books']
assert len(books) == 1, books
assert books[0]['filename'] == 'keep123.pdf'
assert 'lobid_record' not in books[0]
assert books[0]['lobid_url'] == 'http://lobid.org/resources/keep123?format=json'
print('manifest OK')
"
ls /tmp/dnb_migrate_test/
```

Expected: `manifest.json` now has exactly one book (`keep123.pdf`) with `lobid_url` set and no `lobid_record`; `keep123.lobid.json` exists; `purge456.pdf` no longer exists on disk.

```bash
rm -rf /tmp/dnb_migrate_test
```

- [ ] **Step 4: Confirm the full test suite still passes**

Run: `uv run pytest -q`
Expected: PASS (this task adds no pytest coverage of its own, so this just confirms nothing else broke)

- [ ] **Step 5: Commit**

```bash
git add evaluation/scripts/migrate_dnb_toc_lobid_storage.py
git commit -m "feat: add one-off dnb-toc-only lobid-storage migration + EditedVolume purge script"
```

---

### Task 5: Update the design spec to match the corrected design

**Files:**
- Modify: `docs/superpowers/specs/2026-08-14-dnb-toc-corpus-acquisition-design.md`

- [ ] **Step 1: Correct the matching-criterion description**

Find the paragraph in Scope section 1 describing the acquisition filter (search for `"filter client-side for records with a non-empty"`). Replace the phrase `type including EditedVolume or Book` with `type including EditedVolume specifically (not "Book" alone -- see the 2026-08-15 corrections plan for why that broader filter let single-author/thesis/textbook records through)`.

- [ ] **Step 2: Correct the manifest-schema description**

Find the paragraph describing the `"lobid_record"` manifest field (search for `"the full lobid-resources record as fetched"`). Replace the whole bullet describing `"lobid_record"` with:

```markdown
- `"lobid_url"` -- a directly re-fetchable URL for this record's full
  lobid-resources data (the record's own lobid URI, `format=json`
  appended). The full record itself is NOT embedded in `manifest.json`
  -- an earlier version of this design did that under a `"lobid_record"`
  key, but at real corpus scale (~1,000 lines per book, mostly library
  holdings data no code reads) that made `manifest.json` an
  unreviewable multi-hundred-thousand-line file. Instead,
  `fetch_dnb_toc_corpus.py` writes the full record to a separate,
  gitignored `<id>.lobid.json` file alongside the PDF (see
  `evaluation/.gitignore`'s `*.lobid.json` entry) -- available locally
  without a network round-trip, but never committed, same rationale as
  the PDFs themselves.
```

- [ ] **Step 3: Verify the edits render correctly**

Run: `grep -n "lobid_url\|lobid_record\|EditedVolume specifically" docs/superpowers/specs/2026-08-14-dnb-toc-corpus-acquisition-design.md` and read the surrounding context to confirm both edits are coherent and no stale `lobid_record` references remain describing it as embedded in the manifest.

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/specs/2026-08-14-dnb-toc-corpus-acquisition-design.md
git commit -m "docs: correct design spec for EditedVolume-only matching and lobid_url storage"
```

---

## Tasks 6-7 run in the main checkout, NOT a worktree (see the plan header's warning)

### Task 6: Migrate and purge the real 305-book batch

**Files (real data, main checkout):**
- Modify: `evaluation/corpus/dnb-toc-only/manifest.json`
- Create (gitignored): `evaluation/corpus/dnb-toc-only/<key>.lobid.json` per kept book
- Delete: PDFs and manifest entries for every non-`EditedVolume` book

- [ ] **Step 1: Dry-run against the real data first**

```bash
uv run python evaluation/scripts/migrate_dnb_toc_lobid_storage.py --dry-run
```

Read the full purge list. Confirm it includes the known examples from this plan's header (`9783840375644`, `9783842331976`, `9783844019384`, `9783848756353`, `9783863910280`, `9783866493810`, `9783868674095`, `9783868677577`, `9783868943535`) among possibly many more.

- [ ] **Step 2: Run it for real**

```bash
uv run python evaluation/scripts/migrate_dnb_toc_lobid_storage.py
```

- [ ] **Step 3: Verify the result**

```bash
uv run python -c "
import json
data = json.load(open('evaluation/corpus/dnb-toc-only/manifest.json'))
books = data['books']
print('remaining books:', len(books))
assert all('lobid_record' not in b for b in books)
assert all('lobid_url' in b for b in books)
print('schema OK')
"
ls evaluation/corpus/dnb-toc-only/*.lobid.json | wc -l
git status --short evaluation/corpus/dnb-toc-only/ | head -5
```

Confirm: no `lobid_record` fields remain, every kept book has `lobid_url`, the `.lobid.json` count matches the kept-book count, and `git status` shows `manifest.json` as the only tracked change (the `.lobid.json` files and PDFs are gitignored, matching `*.pdf`'s existing pattern now that `*.lobid.json` was added in Task 2).

- [ ] **Step 4: Run the full test suite**

Run: `uv run pytest -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add evaluation/corpus/dnb-toc-only/manifest.json
git commit -m "data: migrate dnb-toc-only manifest to lobid_url schema, purge non-EditedVolume entries"
```

---

### Task 7: Re-run the bulk acquisition with the corrected script

**Files (real data, main checkout):** `evaluation/corpus/dnb-toc-only/manifest.json` grows further; new PDFs and `.lobid.json` files (gitignored) appear.

- [ ] **Step 1: Launch the corrected run**

```bash
uv run python evaluation/scripts/fetch_dnb_toc_corpus.py --from-dump --limit 500
```

Run this in the background (it can take a long time) and monitor its output. With Task 3's retry/reconnect fix, a dropped connection should now log a `[retry N/5]` message and keep going instead of crashing the whole process.

- [ ] **Step 2: When it finishes (or is stopped deliberately), verify**

```bash
uv run python -c "
import json
data = json.load(open('evaluation/corpus/dnb-toc-only/manifest.json'))
books = data['books']
print('total books:', len(books))
assert all('lobid_record' not in b for b in books)
"
uv run pytest -q
```

Spot-check a small sample of the newly-acquired PDFs (e.g. 2-3, same approach as the original plan's Task 4) to confirm they're genuinely TOC-shaped edited-volume scans, not mismatched links.

- [ ] **Step 3: Commit**

```bash
git add evaluation/corpus/dnb-toc-only/manifest.json
git commit -m "data: grow dnb-toc-only corpus via corrected --from-dump acquisition"
```

- [ ] **Step 4: Update RESULTS.md if the corpus is now large enough to be calibration-grade**

If the final book count comfortably exceeds the "few hundred books" bar from `docs/superpowers/specs/2026-08-14-dnb-toc-corpus-acquisition-design.md`'s "Decision criteria", re-run `evaluation/scripts/measure_dnb_scan_noise_stats.py --pdfalto-bin <path>` and update (don't just append another subsection to) the preliminary "Follow-up: first real-scan measurement..." subsection this plan's predecessor added to `evaluation/RESULTS.md`, replacing its "preliminary, smoke-test scale" framing with real numbers now that the sample size supports it.
