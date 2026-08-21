"""Selects a stratified held-out eval-tier sample for the pilot corpus
(design spec
docs/superpowers/specs/2026-08-15-dnb-toc-ground-truth-generation-design.md
section 5). Reads each candidate book's .lobid-cache/<id>.lobid.json for
its publication decade and the manifest's language field, and draws a
sample whose decade/language spread mirrors the corpus's own -- so the
held-out eval tier used to score NuExtract fine-tuning, the heuristic
line-parsing harness, and the classifier pilot isn't accidentally
dominated by one era or language.

Not a pytest test, run once (or re-run after the corpus grows further):

    uv run python cli/select_eval_sample.py --sample-size 75
"""

import argparse
import json
import random

from dnb_toc_ground_truth import corpus

_DEFAULT_SEED = 20260815


def _decade(lobid_record: dict) -> str:
    """Best-effort publication decade from a .lobid-cache record's
    "publication" list, e.g. {"startDate": "2002", ...} -> "2000s". Falls
    back to "unknown" when absent or unparseable, so a book with no usable
    date still gets sampled rather than silently excluded from the pool."""
    publication = lobid_record.get("publication") or []
    for event in publication:
        start = event.get("startDate") or event.get("dateStatement")
        if start and start[:4].isdigit():
            return f"{(int(start[:4]) // 10) * 10}s"
    return "unknown"


def stratify_sample(
    books: list[dict], lobid_records: dict[str, dict], sample_size: int, seed: int = _DEFAULT_SEED,
) -> list[str]:
    """books: manifest entries. lobid_records: manifest_key -> parsed
    .lobid-cache JSON (missing entries treated as {}, i.e. "unknown"
    decade). Returns a stratified sample of manifest keys, roughly
    proportional to each (decade, language) stratum's share of `books`,
    capped at sample_size total. Deterministic for a fixed seed/input, so
    re-running with unchanged corpus contents reproduces the same
    sample."""
    strata: dict[tuple[str, str], list[str]] = {}
    for entry in books:
        key = corpus.manifest_key(entry)
        decade = _decade(lobid_records.get(key, {}))
        language = entry.get("language") or "unknown"
        strata.setdefault((decade, language), []).append(key)

    total = len(books)
    rng = random.Random(seed)
    selected: list[str] = []
    for stratum_keys in strata.values():
        share = round(sample_size * len(stratum_keys) / total)
        share = min(share, len(stratum_keys))
        selected.extend(rng.sample(stratum_keys, share))

    # Rounding can land a couple of keys under/over sample_size; trim or
    # top up from the remaining pool deterministically rather than drift
    # silently away from the requested size.
    if len(selected) > sample_size:
        selected = rng.sample(selected, sample_size)
    elif len(selected) < sample_size:
        remaining = [k for keys in strata.values() for k in keys if k not in selected]
        selected.extend(rng.sample(remaining, min(sample_size - len(selected), len(remaining))))
    return sorted(selected)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sample-size", type=int, default=75)
    parser.add_argument("--seed", type=int, default=_DEFAULT_SEED)
    args = parser.parse_args()

    books = corpus.load_manifest_books()
    lobid_records = {}
    for entry in books:
        key = corpus.manifest_key(entry)
        lobid_path = corpus.lobid_cache_dir() / f"{key}.lobid.json"
        if lobid_path.exists():
            lobid_records[key] = json.loads(lobid_path.read_text(encoding="utf-8"))

    selected = stratify_sample(books, lobid_records, args.sample_size, args.seed)
    output_path = corpus.eval_tier_ids_path()
    output_path.write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(selected)} eval-tier IDs to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
