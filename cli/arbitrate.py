"""Surfaces pilot-corpus books whose two vision-model TOC extractions
didn't clear cli/generate_ground_truth.py's agreement gate, so a
Claude Code session can arbitrate the conflict directly -- see design
spec docs/superpowers/specs/2026-08-16-dnb-toc-arbitration-design.md.
This script only REPORTS and records rejections; it never decides. The
arbitrator reads a book's report, opens the PDF's actual TOC pages via
the Read tool when the text alone doesn't settle it, then either writes
data/corpus/pilot/ground-truth/<key>.expected.json directly (same schema
as a passing book, "verified": true) or runs this script's `reject`
subcommand to permanently record the book as unrecoverable.

    uv run python cli/arbitrate.py
    uv run python cli/arbitrate.py reject 9783515114868 "both models hallucinate on this scan"
"""

import argparse
import json
from datetime import date
from pathlib import Path

from dnb_toc_ground_truth import corpus
from dnb_toc_ground_truth.matching import diff_toc_entries
from dnb_toc_ground_truth.toc_entry import TocEntry
from dnb_toc_ground_truth.vision import load_cached_kind, load_cached_llm_entries, versioned_cache_dir


def _load_rejected_keys() -> set[str]:
    path = corpus.arbitration_rejected_path()
    if not path.exists():
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    return {entry["key"] for entry in data["rejected"]}


def _cached_book_keys(cache_directory: Path) -> list[str]:
    """Every distinct book key with at least one <key>.<model>.json file
    in the current schema version's cache subdirectory (versioned_cache_dir
    -- see its docstring in vision.py; an older version's leftover
    files elsewhere in cache_directory are never surfaced here), sorted for
    stable output. Splitting on the FIRST "." is safe here (unlike
    _cached_models_for_book's model-id slicing below) because book keys are
    manifest filenames' stems -- ISBNs or DNB ids -- which never themselves
    contain a dot."""
    return sorted({p.name.split(".", 1)[0] for p in versioned_cache_dir(cache_directory).glob("*.json")})


def _cached_models_for_book(cache_directory: Path, key: str) -> dict[str, list[TocEntry]]:
    """Every model's cached entries for one book key, keyed by model id
    (the cache filename's middle segment, <key>.<model>.json -- sliced
    rather than split on ".", since a model id can itself contain a dot,
    e.g. "qwen3.6-27b"). Globs the current schema version's cache
    subdirectory only, same reasoning as _cached_book_keys."""
    result: dict[str, list[TocEntry]] = {}
    for path in sorted(versioned_cache_dir(cache_directory).glob(f"{key}.*.json")):
        model = path.name[len(key) + 1: -len(".json")]
        entries = load_cached_llm_entries(cache_directory, key, model)
        if entries is not None:
            result[model] = entries
    return result


def _cached_kinds_for_book(cache_directory: Path, key: str) -> dict[str, str]:
    """Every cached model's extraction "kind" ("vision"/"text") for one
    book key -- same globbing convention as _cached_models_for_book, read
    via load_cached_kind (vision.py). Used only to
    label format_book_report's output (see its own docstring)."""
    result: dict[str, str] = {}
    for path in sorted(versioned_cache_dir(cache_directory).glob(f"{key}.*.json")):
        model = path.name[len(key) + 1: -len(".json")]
        result[model] = load_cached_kind(cache_directory, key, model)
    return result


def books_needing_arbitration(cache_directory: Path) -> list[str]:
    """Book keys with cached model output, no .expected.json yet, and not
    already permanently rejected."""
    rejected = _load_rejected_keys()
    needing = []
    for key in _cached_book_keys(cache_directory):
        if key in rejected:
            continue
        if corpus.expected_json_path(key).exists():
            continue
        needing.append(key)
    return needing


def _format_entry(entry: TocEntry) -> str:
    page = entry.printed_page_number if entry.printed_page_number is not None else "?"
    return f"    p.{page!s:>4}  {entry.title}"


def _kind_label(model: str, kind: str) -> str:
    # "vision"/"text" are the only kinds write_cached_llm_entries writes
    # today, but an unrecognized value (a future third extraction path, a
    # typo, hand-edited cache JSON) must surface as itself, not get
    # silently folded into "text (OCR'd)" -- this report's whole point is
    # to make a human trust its labels at a glance, so a wrong label here
    # would actively mislead rather than merely look unpolished.
    if kind == "vision":
        prefix = "vision"
    elif kind == "text":
        prefix = "text (OCR'd)"
    else:
        prefix = kind
    return f"{prefix}: {model}"


def format_book_report(
    key: str, title: str, pdf_path: Path, models_to_entries: dict[str, list[TocEntry]],
    kinds: dict[str, str] | None = None,
) -> str:
    """Human-readable diff for one book -- the actual disagreement, ready
    for Claude (or a human) to arbitrate. Handles the normal two-model
    case, the single-surviving-model case (the other model's response
    was empty/malformed), and defensively falls back to a plain per-model
    listing for any other count. Each model name in the rendered report is
    prefixed with its extraction "kind" via _kind_label
    (kinds.get(model, "vision")) -- e.g. "vision: Qwen/..." vs.
    "text (OCR'd): meta-llama/..." -- see design spec
    docs/superpowers/specs/2026-08-20-dnb-toc-vision-text-pairing-design.md
    section 6: a mixed-source disagreement is legible to a human arbitrator
    at a glance instead of looking like plain model disagreement. `kinds`
    defaults every model to "vision" when omitted or when a model is
    missing from it -- the pre-existing, all-vision behavior."""
    kinds = kinds or {}

    def label(model: str) -> str:
        return _kind_label(model, kinds.get(model, "vision"))

    lines = [f"=== {key} -- {title} ===", f"PDF: {pdf_path}"]
    model_names = sorted(models_to_entries)
    if len(model_names) == 1:
        model = model_names[0]
        entries = models_to_entries[model]
        lines.append(f"Only {label(model)} returned usable output ({len(entries)} entries) -- verify directly against the page images:")
        for entry in entries:
            lines.append(_format_entry(entry))
        return "\n".join(lines)
    if len(model_names) != 2:
        lines.append(f"Expected 1 or 2 cached models, found {len(model_names)}: {model_names} -- review each list directly:")
        for model in model_names:
            lines.append(f"  -- {label(model)} ({len(models_to_entries[model])} entries) --")
            for entry in models_to_entries[model]:
                lines.append(_format_entry(entry))
        return "\n".join(lines)
    model_a, model_b = model_names
    entries_a, entries_b = models_to_entries[model_a], models_to_entries[model_b]
    matched, only_a, only_b = diff_toc_entries(entries_a, entries_b)
    rate = len(matched) / max(len(entries_a), len(entries_b))
    lines.append(f"{label(model_a)}: {len(entries_a)} entries, {label(model_b)}: {len(entries_b)} entries -- {len(matched)} matched, rate={rate:.2f}")
    if only_a:
        lines.append(f"  Only in {label(model_a)}:")
        for entry in only_a:
            lines.append(_format_entry(entry))
    if only_b:
        lines.append(f"  Only in {label(model_b)}:")
        for entry in only_b:
            lines.append(_format_entry(entry))
    return "\n".join(lines)


def _list(cache_directory: Path) -> int:
    needing = books_needing_arbitration(cache_directory)
    if not needing:
        print("No books currently need arbitration.")
        return 0
    titles = {corpus.manifest_key(book): book.get("title", "") for book in corpus.load_manifest_books()}
    for key in needing:
        models_to_entries = _cached_models_for_book(cache_directory, key)
        kinds = _cached_kinds_for_book(cache_directory, key)
        print(format_book_report(key, titles.get(key, ""), corpus.pdf_path(key), models_to_entries, kinds))
        print()
    return 0


def reject_book(key: str, reason: str, today=date.today) -> int:
    """Permanently records key as unrecoverable so future arbitration
    passes never resurface it. Errors (returns 1) rather than silently
    overwriting if key is already rejected."""
    path = corpus.arbitration_rejected_path()
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"rejected": []}
    if any(entry["key"] == key for entry in data["rejected"]):
        print(f"{key} is already marked rejected -- not overwriting.")
        return 1
    data["rejected"].append({"key": key, "reason": reason, "rejected_at": today().isoformat()})
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("list", help="List books needing arbitration (default)")
    reject_parser = subparsers.add_parser("reject", help="Permanently mark a book as unrecoverable")
    reject_parser.add_argument("key")
    reject_parser.add_argument("reason")
    args = parser.parse_args()

    if args.command == "reject":
        return reject_book(args.key, args.reason)
    return _list(corpus.llm_cache_dir())


if __name__ == "__main__":
    raise SystemExit(main())
