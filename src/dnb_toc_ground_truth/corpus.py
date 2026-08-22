"""Corpus-loading helpers for dnb-toc-ground-truth -- slim equivalent of
chapter-segmentation's evaluation/harness.py. This repo can hold more
than one corpus under data/corpus/<name>/ (today, in practice, just
"pilot"); CORPUS_DIR is the CURRENTLY SELECTED one, switched via
set_corpus() -- CLI scripts call that once, from a --corpus flag
(defaulting to .config.json's "corpus" key, then DEFAULT_CORPUS_NAME),
before making any other corpus.* call (see cli/README.md's shared
--corpus flag docs). Every helper below reads the CORPUS_DIR module
attribute at call time rather than closing over it, so reassigning it
retargets all of them immediately -- this is a single-process CLI tool,
never concurrently serving two corpora at once, so a plain module global
is sufficient; it is not a stand-in for a real multi-tenant design.

PDFs live under pdf/ and ground-truth JSON lives under ground-truth/ --
see design spec
docs/superpowers/specs/2026-08-21-dnb-toc-ground-truth-extraction-design.md
"Repo layout"."""

import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_CORPUS_ROOT = _REPO_ROOT / "data" / "corpus"

DEFAULT_CORPUS_NAME = "pilot"

CORPUS_DIR = _CORPUS_ROOT / DEFAULT_CORPUS_NAME


def repo_root() -> Path:
    """The repo checkout root -- e.g. so a caller can turn an absolute
    corpus.py path into a path relative to the repo, such as
    evaluation_site.py linking a score to the GitHub blob URL of the
    file that produced it."""
    return _REPO_ROOT


def corpus_root() -> Path:
    """Parent of every corpus directory (data/corpus/) -- what
    list_corpora() scans and set_corpus() resolves names against."""
    return _CORPUS_ROOT


def list_corpora() -> list[str]:
    """Every data/corpus/<name>/ directory that has a manifest.json,
    sorted -- real disk discovery, not a hardcoded literal, so a new
    corpus directory shows up here (and in set_corpus()'s "Available"
    list) on its own."""
    if not _CORPUS_ROOT.exists():
        return []
    return sorted(p.name for p in _CORPUS_ROOT.iterdir() if p.is_dir() and (p / "manifest.json").exists())


def set_corpus(name: str, *, create: bool = False) -> None:
    """Switches CORPUS_DIR (and so every helper below) to
    data/corpus/<name>/. Raises ValueError up front for a name with no
    manifest.json there, listing what IS available, rather than
    deferring to a confusing FileNotFoundError somewhere downstream --
    unless create=True (cli/fetch_corpus.py's own job is bootstrapping a
    brand-new corpus's manifest.json, so it can't require one to already
    exist)."""
    global CORPUS_DIR
    candidate = _CORPUS_ROOT / name
    if not create and not (candidate / "manifest.json").exists():
        available = list_corpora()
        raise ValueError(
            f"No corpus named {name!r} (expected {candidate / 'manifest.json'} to exist). "
            f"Available: {available or 'none'}."
        )
    CORPUS_DIR = candidate


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


def crossref_cache_dir() -> Path:
    return CORPUS_DIR / ".crossref-cache"


def evaluation_dir() -> Path:
    return CORPUS_DIR / "evaluation"


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


def evaluation_json_path(key: str) -> Path:
    return evaluation_dir() / f"{key}.expected.json"


def manifest_key(entry: dict) -> str:
    return Path(entry["filename"]).stem


def load_manifest_books() -> list[dict]:
    return json.loads(manifest_path().read_text(encoding="utf-8"))["books"]
