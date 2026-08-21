"""Generates structured ground truth for the dnb-toc-only pilot corpus.
For every manifest book not held out in eval_tier_ids.json (see
select_eval_sample.py), not already carrying a ground-truth JSON file
(bulk-gated or arbitrated), and not permanently rejected
(arbitration-rejected.json), sends the book's TOC pages to every model
named via --use-vision/--use-text (resolved against --endpoints-file),
and writes a ground-truth file only when at least two of the resulting
reads agree well enough (dnb_toc_ground_truth.matching.gate_books,
>=0.90 agreement between the best-agreeing pair) -- see design spec
docs/superpowers/specs/2026-08-21-dnb-toc-ground-truth-extraction-design.md.
Books that don't clear the gate are skipped and reported, not partially
written -- run cli/arbitrate.py on them next.

Safe to run two invocations concurrently against the same checkout --
each book is claimed via a per-key lock file under .locks/ before either
process touches its cache or spends an API call on it.

    uv run python cli/generate_ground_truth.py --use-vision modelA,modelB --limit 50
    uv run python cli/generate_ground_truth.py --use-vision modelA --use-text modelC
    uv run python cli/generate_ground_truth.py --spot-check 30

--spot-check N does not generate anything -- instead it samples N books
that already passed the bulk-tier gate (i.e. "verified": false;
already-human-verified eval-tier entries are excluded) and walks through
a manual, terminal-driven visual Accept/Reject check against the real
PDF for each, then prints the measured accept rate as an estimate of the
gate's real precision.
"""

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Optional

from openai import RateLimitError

from dnb_toc_ground_truth import corpus
from dnb_toc_ground_truth.inference import (
    DEFAULT_CONFIG_FILENAME, DEFAULT_ENDPOINTS_FILENAME, ModelEndpoint,
    load_config, load_endpoint_entries, resolve_model_endpoints,
)
from dnb_toc_ground_truth.matching import gate_books, toc_entry_to_gt_dict
from dnb_toc_ground_truth.ocr import text_extract_toc_entries
from dnb_toc_ground_truth.toc_entry import TocEntry
from dnb_toc_ground_truth.vision import (
    load_cached_kind, load_cached_llm_entries, vision_extract_toc_entries, write_cached_llm_entries,
)

_RATE_LIMIT_WINDOW_ORDER = ("day", "hour", "minute")
_INLINE_RETRY_WINDOWS = frozenset({"hour", "minute"})


def _binding_rate_limit_window(headers) -> Optional[str]:
    """Which of a provider's optional `x-ratelimit-remaining-<window>`
    response headers is actually at 0 -- i.e. which window is the real
    reason this request was rejected. Returns the LONGEST zeroed window
    (day > hour > minute) when more than one is reported at 0, since
    that's the one whose reset actually gates recovery. None if no
    `remaining-*` header reports exactly "0" (a 429 for some other
    reason, or a provider that doesn't send these headers at all --
    every caller must treat that the same as an unclassifiable rate
    limit and fall back to blind backoff)."""
    if not headers:
        return None
    zeroed = {
        key.lower().rsplit("-", 1)[-1]
        for key, value in headers.items()
        if key.lower().startswith("x-ratelimit-remaining-") and value.strip() == "0"
    }
    for window in _RATE_LIMIT_WINDOW_ORDER:
        if window in zeroed:
            return window
    return None


def _retry_after_seconds(headers) -> Optional[float]:
    if not headers:
        return None
    value = headers.get("retry-after")
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


async def _call_with_retry(
    coro_fn, attempts: int = 6, base_delay: float = 2.0, rate_limit_delay: float = 20.0, sleep=asyncio.sleep,
):
    """Exponential backoff for a non-429 failure. A 429 instead schedules
    its retry from the response's own rate-limit headers when present
    (`retry-after` for the exact delay, `x-ratelimit-remaining-<window>`
    to identify which window is actually binding), falling back to a
    blind `rate_limit_delay * attempt_number` linear backoff when those
    headers are absent (not every OpenAI-compatible provider sends
    them). A 429 whose binding window is "day" gives up immediately
    instead of sleeping -- a daily quota, once exhausted, typically does
    not reset within one script invocation's realistic lifetime, so
    blind or even header-precise inline retrying for it just burns wall
    time. Re-invoking the script once the daily quota actually resets
    already skips every book with a cached/decided result, so nothing
    extra is lost by not waiting inline. `sleep` is injectable so tests
    don't actually wait."""
    last_exc: Optional[Exception] = None
    for attempt in range(attempts):
        try:
            return await coro_fn()
        except Exception as exc:  # noqa: BLE001 -- any failure here (network, parse) is retryable
            last_exc = exc
            if attempt >= attempts - 1:
                break
            if isinstance(exc, RateLimitError):
                response = getattr(exc, "response", None)
                headers = response.headers if response is not None else None
                window = _binding_rate_limit_window(headers)
                if window is not None and window not in _INLINE_RETRY_WINDOWS:
                    break
                retry_after = _retry_after_seconds(headers)
                delay = retry_after if retry_after is not None else rate_limit_delay * (attempt + 1)
            else:
                delay = base_delay * 2 ** attempt
            await sleep(delay)
    raise last_exc


_GATE_THRESHOLD_DEFAULT = 0.90


def _run_book_entries(
    key: str, entries_by_endpoint: list[list[TocEntry]], threshold: float,
) -> tuple[str, bool, str]:
    """Core per-book gating logic, given every endpoint's already-
    extracted TocEntry list -- kept separate from PDF/network I/O so
    it's directly unit-testable with synthetic entries. Returns (key,
    passed, reason); reason is "ok" on success, else why the book was
    skipped ("no_entries", "below_threshold")."""
    if all(not entries for entries in entries_by_endpoint):
        return key, False, "no_entries"
    passed, entries, _winning_pair = gate_books(entries_by_endpoint, threshold=threshold)
    if not passed:
        return key, False, "below_threshold"
    gt_path = corpus.expected_json_path(key)
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    gt_path.write_text(
        json.dumps(
            {"entries": [toc_entry_to_gt_dict(e) for e in entries], "verified": False, "source": "bulk_gate"},
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    return key, True, "ok"


_LOCK_STALE_AFTER_SECONDS = 1800.0


def _lock_path(key: str) -> Path:
    return corpus.locks_dir() / f"{key}.lock"


def _acquire_lock(key: str, *, stale_after: float = _LOCK_STALE_AFTER_SECONDS) -> bool:
    """Claims `key` for this process via an atomic exclusive file create
    -- safe against two separate generate_ground_truth.py invocations
    sharing the same checkout racing on the same book. A lock older than
    `stale_after` is assumed to belong to a crashed/killed process and
    is reclaimed by deleting it and retrying the exclusive create once."""
    lock_path = _lock_path(key)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        lock_path.touch(exist_ok=False)
        return True
    except FileExistsError:
        pass
    try:
        age = time.time() - lock_path.stat().st_mtime
    except FileNotFoundError:
        age = None
    if age is not None and age < stale_after:
        return False
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass
    try:
        lock_path.touch(exist_ok=False)
        return True
    except FileExistsError:
        return False


def _release_lock(key: str) -> None:
    try:
        _lock_path(key).unlink()
    except FileNotFoundError:
        pass


async def _run_book(
    key: str, pdf_path: Path, endpoints: list[ModelEndpoint], semaphore: asyncio.Semaphore,
    cache_directory: Path, threshold: float, *, sleep=asyncio.sleep,
) -> tuple[str, bool, str]:
    """Thin I/O wrapper around _run_book_entries -- calls
    vision_extract_toc_entries or text_extract_toc_entries (per each
    endpoint's own `.kind`) through the cache, then _call_with_retry on a
    miss, and delegates every endpoint's resulting entry list to
    _run_book_entries. Catches any exception and reports it as a
    failed-but-tuple-shaped result instead of letting it propagate -- one
    book's failure must never abort the rest of a long, unattended,
    budget-spending batch run."""
    if not _acquire_lock(key):
        return key, False, "locked_by_another_process"
    try:
        entries_by_endpoint: list[list[TocEntry]] = []
        for endpoint in endpoints:
            cached = load_cached_llm_entries(cache_directory, key, endpoint.model_id)
            if cached is not None and load_cached_kind(cache_directory, key, endpoint.model_id) == endpoint.kind:
                entries = cached
            else:
                async def _call(ep=endpoint):
                    async with semaphore:
                        if ep.kind == "text":
                            return await text_extract_toc_entries(pdf_path, ep.model_id, ep.client)
                        return await vision_extract_toc_entries(pdf_path, ep.model_id, ep.client)
                entries = await _call_with_retry(_call, sleep=sleep)
                if entries:
                    write_cached_llm_entries(cache_directory, key, endpoint.model_id, entries, kind=endpoint.kind)
            entries_by_endpoint.append(entries)
        return _run_book_entries(key, entries_by_endpoint, threshold)
    except Exception as exc:  # noqa: BLE001 -- must never let one book crash the whole batch
        print(f"[error] {key}: {exc}{_rate_limit_headers_suffix(exc)}")
        return key, False, f"error: {type(exc).__name__}"
    finally:
        _release_lock(key)


def _rate_limit_headers_suffix(exc: Exception) -> str:
    response = getattr(exc, "response", None)
    if not isinstance(exc, RateLimitError) or response is None:
        return ""
    relevant = {k: v for k, v in response.headers.items() if "ratelimit" in k.lower() or k.lower() == "retry-after"}
    if not relevant:
        return ""
    return " [" + ", ".join(f"{k}={v}" for k, v in sorted(relevant.items())) + "]"


async def _run_all(
    keys_and_paths: list[tuple[str, Path]], endpoints: list[ModelEndpoint], concurrency: int,
    cache_directory: Path, threshold: float,
) -> list[tuple[str, bool, str]]:
    semaphore = asyncio.Semaphore(concurrency)
    return list(await asyncio.gather(*[
        _run_book(key, path, endpoints, semaphore, cache_directory, threshold)
        for key, path in keys_and_paths
    ]))


def _is_stale_bulk_gate_entry(gt_path: Path) -> bool:
    """True if gt_path is a BULK-tier ("source": "bulk_gate") file
    missing the "skip" key on at least one entry -- the current
    extraction standard (verbatim per-line extraction plus a "skip"
    flag) always sets it, so its absence marks a file written to an
    older standard that should be regenerated. An "agent_arbitration"
    file is never touched here regardless of this check."""
    try:
        data = json.loads(gt_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    if data.get("source") != "bulk_gate":
        return False
    entries = data.get("entries", [])
    return bool(entries) and not all("skip" in e for e in entries)


def _still_needs_a_decision(book: dict, eval_tier_ids: set[str], rejected_ids: set[str]) -> bool:
    key = corpus.manifest_key(book)
    if key in eval_tier_ids or key in rejected_ids:
        return False
    gt_path = corpus.expected_json_path(key)
    if not gt_path.exists():
        return True
    return _is_stale_bulk_gate_entry(gt_path)


def _resolve_endpoints(args: argparse.Namespace, config: dict) -> list[ModelEndpoint]:
    """Resolves every requested vision/text model against
    --endpoints-file, config-file defaults filled in for any CLI flag
    left unset. Requires at least one --use-vision model; --use-text is
    optional. Raises SystemExit naming what's wrong on a user error (no
    vision model given at all)."""
    endpoints_file = Path(args.endpoints_file or config.get("endpoints_file", DEFAULT_ENDPOINTS_FILENAME))
    use_vision = args.use_vision or config.get("use_vision") or []
    use_text = args.use_text or config.get("use_text") or []
    if not use_vision:
        raise SystemExit("--use-vision is required (directly or via the config file's \"use_vision\" key)")
    entries = load_endpoint_entries(endpoints_file)
    vision_endpoints = resolve_model_endpoints(use_vision, "vision", entries)
    text_endpoints = resolve_model_endpoints(use_text, "text", entries) if use_text else []
    return vision_endpoints + text_endpoints


def _generate(args: argparse.Namespace, config: dict) -> int:
    eval_tier_path = corpus.eval_tier_ids_path()
    eval_tier_ids = set(json.loads(eval_tier_path.read_text(encoding="utf-8"))) if eval_tier_path.exists() else set()
    rejected_path = corpus.arbitration_rejected_path()
    rejected_ids = (
        {entry["key"] for entry in json.loads(rejected_path.read_text(encoding="utf-8"))["rejected"]}
        if rejected_path.exists() else set()
    )

    books = corpus.load_manifest_books()
    eligible = [b for b in books if _still_needs_a_decision(b, eval_tier_ids, rejected_ids)]
    limit = args.limit if args.limit is not None else config.get("limit")
    if limit is not None:
        eligible = eligible[:limit]
    candidates = [
        (corpus.manifest_key(b), corpus.pdf_path(corpus.manifest_key(b)))
        for b in eligible if corpus.pdf_path(corpus.manifest_key(b)).exists()
    ]
    missing_pdf_count = len(eligible) - len(candidates)

    endpoints = _resolve_endpoints(args, config)
    concurrency = args.concurrency if args.concurrency is not None else config.get("concurrency", 4)
    threshold = args.gate_threshold if args.gate_threshold is not None else config.get("gate_threshold", _GATE_THRESHOLD_DEFAULT)

    results = asyncio.run(_run_all(candidates, endpoints, concurrency, corpus.llm_cache_dir(), threshold))
    passed = [r for r in results if r[1]]
    by_reason: dict[str, int] = {}
    for _, ok, reason in results:
        if not ok:
            by_reason[reason] = by_reason.get(reason, 0) + 1
    print("Endpoints used: " + ", ".join(f"{e.kind}={e.label}" for e in endpoints))
    print(f"{len(passed)}/{len(results)} books passed the gate and got ground-truth JSON written.")
    for reason, count in sorted(by_reason.items()):
        print(f"  {count} skipped: {reason}")
    if missing_pdf_count:
        print(f"  {missing_pdf_count} skipped: missing_pdf (not downloaded locally)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--limit", type=int, default=None, help="Process at most this many books (smoke-test convenience)")
    parser.add_argument("--concurrency", type=int, default=None, help="How many books to process concurrently (default: 4, or config file's \"concurrency\")")
    parser.add_argument(
        "--spot-check", type=int, default=None, metavar="N",
        help="Instead of generating, sample N passing bulk-tier books and walk through a visual Accept/Reject check",
    )
    parser.add_argument(
        "--use-vision", type=lambda s: [m.strip() for m in s.split(",") if m.strip()], default=None, metavar="MODEL[,MODEL...]",
        help="Model id(s) to resolve against --endpoints-file for the VISION side -- required (directly or via config), at least one",
    )
    parser.add_argument(
        "--use-text", type=lambda s: [m.strip() for m in s.split(",") if m.strip()], default=None, metavar="MODEL[,MODEL...]",
        help="Model id(s) to resolve against --endpoints-file for the TEXT (OCR'd) side -- optional",
    )
    parser.add_argument("--endpoints-file", type=Path, default=None, help=f"Path to the endpoints file (default: {DEFAULT_ENDPOINTS_FILENAME}, or config file's \"endpoints_file\")")
    parser.add_argument("--config-file", type=Path, default=Path(DEFAULT_CONFIG_FILENAME), help=f"Path to the config file (default: {DEFAULT_CONFIG_FILENAME})")
    parser.add_argument("--gate-threshold", type=float, default=None, help="Whole-book agreement threshold, 0-1 (default: 0.90, or config file's \"gate_threshold\")")
    args = parser.parse_args()

    config = load_config(args.config_file)
    if args.spot_check is not None:
        return _spot_check(args.spot_check)
    return _generate(args, config)


def _spot_check(n: int) -> int:
    """Terminal-driven precision check: sample n books that passed the
    bulk-tier gate, print each one's PDF path and generated entries, and
    prompt for a manual Accept/Reject after visually opening the PDF --
    then report measured precision for the gate threshold. Only samples
    books whose "verified" field is False (bulk-tier, machine-gated)."""
    import random

    passing = []
    for p in sorted(corpus.ground_truth_dir().glob("*.expected.json")):
        gt = json.loads(p.read_text(encoding="utf-8"))
        if gt.get("verified") is False:
            passing.append(p.name.removesuffix(".expected.json"))
    sample = random.sample(passing, min(max(n, 0), len(passing)))
    accepted = 0
    for key in sample:
        gt = json.loads(corpus.expected_json_path(key).read_text(encoding="utf-8"))
        print(f"\n=== {key} ===\nPDF: {corpus.pdf_path(key)}")
        print(json.dumps(gt["entries"], indent=2, ensure_ascii=False))
        answer = input("Matches the scan? [y/N] ").strip().lower()
        if answer == "y":
            accepted += 1
    if sample:
        print(f"\nSpot-check precision: {accepted}/{len(sample)} = {accepted / len(sample):.0%}")
    else:
        print("No passing books found to spot-check yet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
