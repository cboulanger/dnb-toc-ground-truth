"""Corpus-loading helpers for dnb-toc-ground-truth -- slim, single-corpus
equivalent of chapter-segmentation's evaluation/harness.py. This repo has
exactly one corpus (data/corpus/pilot/), so there's no multi-corpus
list_corpora() indirection to carry over. PDFs live under pdf/ and
ground-truth JSON lives under ground-truth/ -- see design spec
docs/superpowers/specs/2026-08-21-dnb-toc-ground-truth-extraction-design.md
"Repo layout"."""

import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CORPUS_DIR = _REPO_ROOT / "data" / "corpus" / "pilot"


def corpus_dir() -> Path:
    return CORPUS_DIR


def pdf_dir() -> Path:
    return CORPUS_DIR / "pdf"


def ground_truth_dir() -> Path:
    return CORPUS_DIR / "ground-truth"


def llm_cache_dir() -> Path:
    return CORPUS_DIR / "llm-cache"


def lobid_cache_dir() -> Path:
    return CORPUS_DIR / ".lobid-cache"


def locks_dir() -> Path:
    return CORPUS_DIR / ".locks"


def manifest_path() -> Path:
    return CORPUS_DIR / "manifest.json"


def eval_tier_ids_path() -> Path:
    return CORPUS_DIR / "eval_tier_ids.json"


def arbitration_rejected_path() -> Path:
    return CORPUS_DIR / "arbitration-rejected.json"


def pdf_path(key: str) -> Path:
    return pdf_dir() / f"{key}.pdf"


def expected_json_path(key: str) -> Path:
    return ground_truth_dir() / f"{key}.expected.json"


def manifest_key(entry: dict) -> str:
    return Path(entry["filename"]).stem


def load_manifest_books() -> list[dict]:
    return json.loads(manifest_path().read_text(encoding="utf-8"))["books"]
