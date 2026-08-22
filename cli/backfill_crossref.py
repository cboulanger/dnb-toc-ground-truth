#!/usr/bin/env python3
"""Backfills Crossref book DOI and chapter data for existing manifest.json
entries that already have a .expected.json ground-truth file but no doi
yet -- see design spec
docs/superpowers/specs/2026-08-21-crossref-cross-validation-design.md.

For every such book, looks up its ISBN on Crossref (dnb_toc_ground_truth.
crossref.fetch_crossref_book): if Crossref has a DOI for the book, writes
it into manifest.json, regardless of whether Crossref also has usable
chapter data for it. Chapter data (if any) is cached to
.crossref-cache/<isbn>.crossref.json either way, as a side effect of
fetch_crossref_book itself. Already-cached ISBNs are skipped on repeat
runs unless --force is passed.

Usage:
    uv run python cli/backfill_crossref.py
    uv run python cli/backfill_crossref.py --force
    uv run python cli/backfill_crossref.py --contact-email you@example.org
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import httpx

from dnb_toc_ground_truth import corpus, crossref, inference


def _needs_backfill(book: dict) -> bool:
    if book.get("doi"):
        return False
    key = corpus.manifest_key(book)
    return corpus.expected_json_path(key).exists()


def backfill(
    manifest_path: Path,
    client: httpx.Client,
    contact_email: Optional[str],
    cache_dir: Path,
    force: bool,
    eval_dir: Path,
    min_chapters: int,
) -> tuple[int, int, int, int]:
    """Returns (checked, dois_found, chapter_lists_cached,
    evaluation_entries_written) -- checked counts only books that pass
    _needs_backfill; dois_found counts those where Crossref returned a
    doi; chapter_lists_cached counts those where Crossref returned at
    least one chapter; evaluation_entries_written counts those where
    crossref.write_evaluation_entry actually wrote a file (at least
    min_chapters page-numbered chapters)."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    books = manifest["books"]
    checked = 0
    dois_found = 0
    chapter_lists_cached = 0
    evaluation_entries_written = 0
    manifest_changed = False

    for book in books:
        if not _needs_backfill(book):
            continue
        key = corpus.manifest_key(book)
        isbn = crossref.normalize_isbn(key)
        if isbn is None:
            print(f"[skip] {key}: not a valid ISBN")
            continue
        checked += 1
        data = crossref.fetch_crossref_book(isbn, client, contact_email, cache_dir, force=force)
        if data.doi:
            book["doi"] = data.doi
            dois_found += 1
            manifest_changed = True
        if data.chapters:
            chapter_lists_cached += 1
        if crossref.write_evaluation_entry(isbn, data, eval_dir, min_chapters):
            evaluation_entries_written += 1
        print(f"[{key}] doi={data.doi or 'none'} chapters={len(data.chapters)}")

    if manifest_changed:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return checked, dois_found, chapter_lists_cached, evaluation_entries_written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--force", action="store_true",
        help="Re-query Crossref even for an already-cached ISBN",
    )
    parser.add_argument(
        "--contact-email", default=None,
        help="Crossref polite-pool contact email (default: config file's \"contact_email\")",
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
    corpus.set_corpus(args.corpus or config.get("corpus") or corpus.DEFAULT_CORPUS_NAME)
    contact_email = args.contact_email or config.get("contact_email")

    with httpx.Client(follow_redirects=True) as client:
        checked, found, cached, written = backfill(
            corpus.manifest_path(), client, contact_email, corpus.crossref_cache_dir(), args.force,
            corpus.evaluation_dir(), args.min_chapters,
        )
    print(
        f"\n{checked} book(s) checked, {found} DOI(s) found, {cached} chapter list(s) cached, "
        f"{written} evaluation-corpus entry(ies) written."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
