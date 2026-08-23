#!/usr/bin/env python3
"""Prints corpus coverage counts -- manifest size, ground-truth
coverage (and how it was produced), per-model LLM-cache reading counts
(models with fewer than _MIN_MODEL_READINGS are omitted -- a one-off
smoke-test endpoint isn't worth a permanent table row), outstanding
arbitration/rejection backlog, the held-out eval tier, and Crossref
evaluation-corpus size -- as a markdown table, and rewrites it into
README.md's "Current status" section in place.

Run this after a `generate_ground_truth.py`/`arbitrate.py`/
`backfill_crossref.py`/`skip_list.py` batch changes the numbers, so the
README doesn't go stale (see `AGENTS.md`).

    uv run python cli/corpus_status.py           # print + update README.md
    uv run python cli/corpus_status.py --check    # print only, exit 1 if README.md is stale
"""

import argparse
import json
from pathlib import Path

from dnb_toc_ground_truth import corpus, inference, vision

_MARKER_START = "<!-- corpus-status:start -->"
_MARKER_END = "<!-- corpus-status:end -->"

_MIN_MODEL_READINGS = 50


def _cached_model_counts() -> dict[str, int]:
    """Every distinct model id with at least one llm-cache entry, mapped
    to how many distinct books have one -- discovered from disk (same
    "<key>.<safe_model>.json" filenames `vision.cache_path` writes), not
    a hardcoded roster, so a newly-added or retired endpoint shows up
    (or disappears) here without editing this script. Splitting on the
    FIRST "." is safe only because no manifest key itself contains a
    "." (verified against the current pilot corpus) -- the model
    portion can and does (e.g. "Mistral-Small-3.2-24B-Instruct-2506")."""
    cache_dir = vision.versioned_cache_dir(corpus.llm_cache_dir())
    counts: dict[str, int] = {}
    if not cache_dir.exists():
        return counts
    for path in cache_dir.glob("*.json"):
        _key, _, rest = path.name.partition(".")
        safe_model = rest[: -len(".json")]
        model = safe_model.replace("__", "/")
        counts[model] = counts.get(model, 0) + 1
    return counts


def _ground_truth_source_counts() -> dict[str, int]:
    """How many ground-truth files came from each `"source"` value
    ("bulk_gate" vs "agent_arbitration", or a future third route) --
    the arbitration rate the README's Methodology section already calls
    out as this project's core efficiency metric, so it's worth keeping
    fresh alongside the raw counts rather than only in prose."""
    counts: dict[str, int] = {}
    for path in corpus.ground_truth_dir().glob("*.expected.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        source = data.get("source", "unknown")
        counts[source] = counts.get(source, 0) + 1
    return counts


def _cached_book_keys() -> set[str]:
    return {p.name.split(".", 1)[0] for p in vision.versioned_cache_dir(corpus.llm_cache_dir()).glob("*.json")}


def _rejected_keys() -> set[str]:
    path = corpus.arbitration_rejected_path()
    if not path.exists():
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    return {entry["key"] for entry in data["rejected"]}


def _eval_tier_count() -> int:
    path = corpus.eval_tier_ids_path()
    if not path.exists():
        return 0
    return len(json.loads(path.read_text(encoding="utf-8")))


def _skip_list_count() -> int:
    path = corpus.model_skip_list_path()
    if not path.exists():
        return 0
    return len(json.loads(path.read_text(encoding="utf-8"))["skipped"])


def build_status_table() -> str:
    manifest_count = len(corpus.load_manifest_books())
    ground_truth_count = len(list(corpus.ground_truth_dir().glob("*.expected.json")))
    evaluation_count = len(list(corpus.evaluation_dir().glob("*.expected.json")))
    model_counts = _cached_model_counts()
    source_counts = _ground_truth_source_counts()
    rejected = _rejected_keys()
    needing_arbitration = len(_cached_book_keys() - {p.stem.removesuffix(".expected") for p in corpus.ground_truth_dir().glob("*.expected.json")} - rejected)

    lines = ["| Metric | Count |", "|---|---|", f"| Manifest books | {manifest_count} |",
              f"| Books with ground truth | {ground_truth_count} |",
              f"| — via two-model gate (`bulk_gate`) | {source_counts.get('bulk_gate', 0)} |",
              f"| — via arbitration (`agent_arbitration`) | {source_counts.get('agent_arbitration', 0)} |"]
    for model in sorted(model_counts):
        if model_counts[model] < _MIN_MODEL_READINGS:
            continue
        lines.append(f"| Books with a `{model}` reading | {model_counts[model]} |")
    lines.append(f"| Books awaiting arbitration | {needing_arbitration} |")
    lines.append(f"| Books permanently rejected (unrecoverable) | {len(rejected)} |")
    lines.append(f"| Known model/book skips (retry once fixed) | {_skip_list_count()} |")
    lines.append(f"| Held-out eval-tier sample | {_eval_tier_count()} |")
    lines.append(f"| Crossref evaluation-corpus entries | {evaluation_count} |")
    return "\n".join(lines)


def update_readme(table: str, readme_path: Path) -> bool:
    """Replaces the text between the marker comments with `table`.
    Returns whether the file's content changed (so --check can tell
    "already current" from "just rewrote it"). Raises SystemExit if
    either marker is missing -- fail loud rather than silently no-op on
    a README that's been restructured."""
    text = readme_path.read_text(encoding="utf-8")
    start = text.find(_MARKER_START)
    end = text.find(_MARKER_END)
    if start == -1 or end == -1:
        raise SystemExit(f"{readme_path}: missing {_MARKER_START!r}/{_MARKER_END!r} markers")
    new_text = text[: start + len(_MARKER_START)] + "\n\n" + table + "\n\n" + text[end:]
    if new_text == text:
        return False
    readme_path.write_text(new_text, encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--check", action="store_true",
        help="Print the table but don't write README.md; exit 1 if README.md's current section is stale",
    )
    parser.add_argument(
        "--readme", type=Path, default=None,
        help="Path to the README to update (default: this repo's top-level README.md)",
    )
    parser.add_argument(
        "--corpus", default=None,
        help=f"Corpus to operate on (default: config file's \"corpus\", or {corpus.DEFAULT_CORPUS_NAME!r})",
    )
    parser.add_argument(
        "--config-file", type=Path, default=Path(inference.DEFAULT_CONFIG_FILENAME),
        help=f"Path to the config file (default: {inference.DEFAULT_CONFIG_FILENAME})",
    )
    args = parser.parse_args()

    config = inference.load_config(args.config_file)
    corpus.set_corpus(args.corpus or config.get("corpus") or corpus.DEFAULT_CORPUS_NAME)

    table = build_status_table()
    print(table)

    readme_path = args.readme or (corpus.repo_root() / "README.md")
    if args.check:
        current = readme_path.read_text(encoding="utf-8")
        start = current.find(_MARKER_START)
        end = current.find(_MARKER_END)
        if start == -1 or end == -1:
            raise SystemExit(f"{readme_path}: missing {_MARKER_START!r}/{_MARKER_END!r} markers")
        is_current = current[start + len(_MARKER_START) : end].strip() == table
        print("README.md is up to date." if is_current else "README.md is STALE -- run without --check to update.")
        return 0 if is_current else 1

    changed = update_readme(table, readme_path)
    print(f"\n{readme_path} {'updated' if changed else 'already up to date'}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
