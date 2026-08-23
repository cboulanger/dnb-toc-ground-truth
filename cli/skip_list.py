"""Manages model-skip-list.json -- (book key, model id) pairs that
cli/generate_ground_truth.py should never attempt, because a prior run
demonstrated that specific model hangs or produces malformed output on
that specific book (e.g. numind/NuExtract3's non-terminating/repetitive
generation on a handful of unusually large, deeply-nested TOCs -- see
docs/superpowers/specs/2026-08-21-dnb-toc-ground-truth-extraction-design.md's
neighboring investigation notes). Skipping the pair entirely avoids
burning the hard wall-clock timeout's full retry budget on a call that's
already known not to terminate cleanly, while leaving the book's OTHER
model readings (and the book itself) untouched -- this is not the same
as cli/arbitrate.py's `reject`, which discards a whole book.

A skip is meant to be temporary: once a fix is tried (e.g. guided/
structured decoding), `remove` clears the entry so the next
generate_ground_truth.py run attempts that pair again.

    uv run python cli/skip_list.py add 3789082120 "numind/NuExtract3" "non-terminating generation (repetition loop); frequency_penalty 0.1-0.3 tried, unreliable and regresses other books"
    uv run python cli/skip_list.py remove 3789082120 "numind/NuExtract3"
    uv run python cli/skip_list.py list
"""

import argparse
import json
from datetime import date
from pathlib import Path

from dnb_toc_ground_truth import corpus, inference


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"skipped": []}


def load_skip_set() -> set[tuple[str, str]]:
    """Every currently-skipped (key, model) pair -- what
    generate_ground_truth.py checks before attempting an endpoint call."""
    data = _load(corpus.model_skip_list_path())
    return {(entry["key"], entry["model"]) for entry in data["skipped"]}


def add_skip(key: str, model: str, reason: str, today=date.today) -> int:
    """Errors (returns 1) rather than silently overwriting if the pair
    is already skipped -- same non-clobbering contract as
    cli/arbitrate.py's reject_book."""
    path = corpus.model_skip_list_path()
    data = _load(path)
    if any(entry["key"] == key and entry["model"] == model for entry in data["skipped"]):
        print(f"{key}/{model} is already on the skip list -- not overwriting.")
        return 1
    data["skipped"].append({"key": key, "model": model, "reason": reason, "skipped_at": today().isoformat()})
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return 0


def remove_skip(key: str, model: str) -> int:
    """Errors (returns 1) if the pair isn't currently on the list."""
    path = corpus.model_skip_list_path()
    data = _load(path)
    remaining = [e for e in data["skipped"] if not (e["key"] == key and e["model"] == model)]
    if len(remaining) == len(data["skipped"]):
        print(f"{key}/{model} is not on the skip list.")
        return 1
    data["skipped"] = remaining
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return 0


def _list() -> int:
    data = _load(corpus.model_skip_list_path())
    if not data["skipped"]:
        print("No books are currently skipped for any model.")
        return 0
    for entry in data["skipped"]:
        print(f"{entry['key']} / {entry['model']} (since {entry['skipped_at']}): {entry['reason']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--corpus", default=None,
        help=f"Corpus to operate on (default: config file's \"corpus\", or {corpus.DEFAULT_CORPUS_NAME!r})",
    )
    parser.add_argument(
        "--config-file", type=Path, default=Path(inference.DEFAULT_CONFIG_FILENAME),
        help=f"Path to the config file (default: {inference.DEFAULT_CONFIG_FILENAME})",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_parser = subparsers.add_parser("add", help="Skip a (key, model) pair")
    add_parser.add_argument("key")
    add_parser.add_argument("model")
    add_parser.add_argument("reason")

    remove_parser = subparsers.add_parser("remove", help="Un-skip a (key, model) pair")
    remove_parser.add_argument("key")
    remove_parser.add_argument("model")

    subparsers.add_parser("list", help="List every currently-skipped pair")

    args = parser.parse_args()

    config = inference.load_config(args.config_file)
    corpus.set_corpus(args.corpus or config.get("corpus") or corpus.DEFAULT_CORPUS_NAME)

    if args.command == "add":
        return add_skip(args.key, args.model, args.reason)
    if args.command == "remove":
        return remove_skip(args.key, args.model)
    return _list()


if __name__ == "__main__":
    raise SystemExit(main())
