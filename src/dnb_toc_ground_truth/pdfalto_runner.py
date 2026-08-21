"""Runs the pdfalto binary (https://github.com/kermitt2/pdfalto) against a
PDF and caches its ALTO XML output -- not vendored, matching the Kreuzberg
OCR sidecar precedent of treating an external tool as developer-provided,
not bundled. See
docs/superpowers/specs/2026-08-10-layout-based-toc-classifier-pilot-design.md."""

import os
import subprocess
from pathlib import Path


def resolve_pdfalto_binary(cli_arg: str | None) -> str:
    """Resolves the pdfalto binary path: explicit --pdfalto-bin flag, then
    the PDFALTO_BIN environment variable, then bare "pdfalto" on PATH."""
    if cli_arg:
        return cli_arg
    env_value = os.environ.get("PDFALTO_BIN")
    if env_value:
        return env_value
    return "pdfalto"


def alto_xml_path(pdf_path: Path, cache_dir: Path) -> Path:
    return cache_dir / f"{pdf_path.stem}.alto.xml"


def ensure_alto_xml(pdf_path: Path, cache_dir: Path, pdfalto_bin: str) -> Path:
    """Returns the cached ALTO XML path for pdf_path, running pdfalto only
    if the cache entry doesn't already exist. Raises RuntimeError if
    pdfalto exits non-zero or doesn't produce the expected output file."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path = alto_xml_path(pdf_path, cache_dir)
    if output_path.exists():
        return output_path

    result = subprocess.run(
        [pdfalto_bin, "-skipGraphs", str(pdf_path), str(output_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not output_path.exists():
        output_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"pdfalto failed on {pdf_path} (exit {result.returncode}): {result.stderr}"
        )
    return output_path
