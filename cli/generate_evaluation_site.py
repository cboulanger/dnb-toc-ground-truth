#!/usr/bin/env python3
"""Builds the static GitHub Pages site presenting crossref-evaluation
scores -- see dnb_toc_ground_truth.evaluation_site for the page-building
logic and .github/workflows/pages.yml for how this is run and deployed
on every push to main. Reads only committed corpus data (ground truth,
the committed Crossref evaluation corpus, and llm-cache) -- no network
access, no secrets required.

Usage:
    uv run python cli/generate_evaluation_site.py
    uv run python cli/generate_evaluation_site.py --output-dir _site
"""

import argparse
from pathlib import Path

from dnb_toc_ground_truth.evaluation_site import write_site


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--output-dir", type=Path, default=Path("_site"),
        help="Directory to write the site into (default: _site/)",
    )
    args = parser.parse_args()
    write_site(args.output_dir)
    print(f"Wrote evaluation site to {args.output_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
