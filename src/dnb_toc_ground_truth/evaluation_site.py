"""Builds the static GitHub Pages site presenting crossref-evaluation
scores (dnb_toc_ground_truth.crossref_evaluation) for every corpus this
repo has (corpus.list_corpora()) -- see cli/generate_evaluation_site.py
for the CLI wrapper this feeds, and .github/workflows/pages.yml for how
the site is built and deployed on every push to main."""

import html as _html
from dataclasses import dataclass
from pathlib import Path

from dnb_toc_ground_truth import corpus
from dnb_toc_ground_truth.crossref_evaluation import (
    BookMetrics,
    discover_cached_models,
    evaluate_corpus,
    evaluate_model_corpus,
)


@dataclass(frozen=True)
class SourceScores:
    label: str
    is_ground_truth: bool
    results: list[BookMetrics]
    uncovered_count: int


@dataclass(frozen=True)
class CorpusData:
    name: str
    titles: dict[str, str]
    sources: list[SourceScores]


def collect_corpus_data() -> CorpusData:
    """Runs the ground-truth crossref evaluation plus every discovered
    llm-cache model's, for corpus.py's CURRENTLY SELECTED corpus
    (corpus.set_corpus()) -- callers that want a specific corpus must
    call corpus.set_corpus(name) first, same as any other corpus.*
    consumer."""
    titles = {corpus.manifest_key(book): book.get("title", "") for book in corpus.load_manifest_books()}
    gt_results, gt_uncovered = evaluate_corpus()
    sources = [
        SourceScores(label="Ground truth", is_ground_truth=True, results=gt_results, uncovered_count=len(gt_uncovered)),
    ]
    for model in discover_cached_models():
        model_results, no_cache = evaluate_model_corpus(model)
        sources.append(SourceScores(label=model, is_ground_truth=False, results=model_results, uncovered_count=len(no_cache)))
    return CorpusData(name=corpus.corpus_dir().name, titles=titles, sources=sources)


_STYLE = """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
       max-width: 60rem; margin: 2rem auto; padding: 0 1.5rem; color: #1a1a1a; line-height: 1.5; }
h1, h2, h3 { line-height: 1.25; }
a { color: #0550ae; }
table { border-collapse: collapse; width: 100%; margin: 0.75rem 0 1.75rem; }
th, td { text-align: left; padding: 0.3rem 0.6rem; border-bottom: 1px solid #e2e2e2; }
th { border-bottom: 2px solid #999; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
tr.mean { font-weight: 700; }
tr.mean td { border-top: 2px solid #999; }
section.ground-truth h2 { color: #0a5c2b; }
section.ground-truth table { font-weight: 700; }
.caveats { background: #fff8e6; border: 1px solid #e8d488; border-radius: 6px; padding: 0.9rem 1.2rem; }
.caveats ul { margin: 0.4rem 0 0; padding-left: 1.2rem; }
.coverage { color: #555; font-size: 0.9rem; margin: -1rem 0 1rem; }
footer { color: #777; font-size: 0.85rem; margin-top: 3rem; }
"""


def _page(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_html.escape(title)}</title>
<style>{_STYLE}</style>
</head>
<body>
{body}
<footer>Generated from committed corpus data by cli/generate_evaluation_site.py.</footer>
</body>
</html>
"""


def render_index_html(corpus_names: list[str]) -> str:
    links = "\n".join(f'<li><a href="{_html.escape(name)}.html">{_html.escape(name)}</a></li>' for name in corpus_names)
    body = f"""<h1>Crossref evaluation</h1>
<p>This project cross-checks its ground truth -- and, per book, each
vision-LLM's raw table-of-contents read -- against an independent,
non-LLM data source: <a href="https://www.crossref.org/">Crossref</a>'s
own per-chapter registration metadata (title, authors, page range), for
books that have one. For each book, the committed ground truth's real
chapters are compared against Crossref's registered chapters (title and
first-page-number match) to get a precision/recall/F1 score; the same
comparison is run again for every LLM's raw, pre-agreement-gate cache
entry, so a model's extraction quality can be read directly off an
independent signal.</p>
<div class="caveats">
<strong>Caveats -- read before trusting a number below:</strong>
<ul>
<li><strong>The sample is small and not random.</strong> Only a minority
of this project's ground-truthed books have enough Crossref-registered,
page-numbered chapters to produce a usable comparison at all -- Crossref
registration skews toward larger, more prominent publishers.</li>
<li><strong>Crossref's own data is publisher-submitted, not vetted by
this project, and often deficient</strong> -- missing page numbers,
chapters registered as one part instead of individually, or simply never
registered at all. A low score against Crossref does not mean the
ground truth (or a model) is wrong.</li>
<li><strong>Scores are macro-averaged</strong> -- the mean of each
compared book's own precision/recall/F1, every book weighted equally
regardless of length.</li>
</ul>
</div>
<h2>Corpora</h2>
<ul>
{links}
</ul>
"""
    return _page("Crossref evaluation", body)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _render_source_section(source: SourceScores, titles: dict[str, str]) -> str:
    section_class = "ground-truth" if source.is_ground_truth else "model"
    rows = sorted(source.results, key=lambda r: (titles.get(r.key, r.key).lower(), r.key))
    row_html = "\n".join(
        f"<tr><td>{_html.escape(titles.get(r.key, r.key))}</td>"
        f'<td class="num">{r.precision:.0%}</td><td class="num">{r.recall:.0%}</td><td class="num">{r.f1:.0%}</td>'
        f'<td class="num">{r.tp}</td><td class="num">{r.fp}</td><td class="num">{r.fn}</td></tr>'
        for r in rows
    )
    if source.results:
        mean_precision = _mean([r.precision for r in source.results])
        mean_recall = _mean([r.recall for r in source.results])
        mean_f1 = _mean([r.f1 for r in source.results])
        mean_row = (
            f'<tr class="mean"><td>Mean ({len(source.results)} book(s))</td>'
            f'<td class="num">{mean_precision:.0%}</td><td class="num">{mean_recall:.0%}</td>'
            f'<td class="num">{mean_f1:.0%}</td><td class="num">-</td><td class="num">-</td><td class="num">-</td></tr>'
        )
        table = f"""<table>
<thead><tr><th>Book</th><th class="num">Precision</th><th class="num">Recall</th><th class="num">F1</th>
<th class="num">TP</th><th class="num">FP</th><th class="num">FN</th></tr></thead>
<tbody>
{row_html}
{mean_row}
</tbody>
</table>"""
    else:
        table = "<p>No crossref-sample books had data for this source.</p>"
    coverage = f'<p class="coverage">{source.uncovered_count} crossref-sample book(s) with no data for this source.</p>'
    heading = "Ground truth" if source.is_ground_truth else f"Model: {_html.escape(source.label)}"
    return f'<section class="{section_class}">\n<h2>{heading}</h2>\n{table}\n{coverage}\n</section>'


def render_corpus_html(data: CorpusData) -> str:
    sections = "\n".join(_render_source_section(source, data.titles) for source in data.sources)
    body = f"""<p><a href="index.html">&larr; Corpora</a></p>
<h1>{_html.escape(data.name)}</h1>
<p>Per-book precision/recall/F1 against the committed Crossref evaluation
corpus. Ground truth is the project's own committed data (highlighted
below); each model section scores that model's raw, pre-agreement-gate
llm-cache extraction over the same books.</p>
{sections}
"""
    return _page(f"{data.name} -- Crossref evaluation", body)


def write_site(output_dir: Path) -> None:
    """Renders index.html plus one <name>.html per corpus.list_corpora()
    entry. Temporarily switches corpus.py's selected corpus (via
    corpus.set_corpus()) to each in turn, restoring whatever was
    selected beforehand once done -- callers other than
    cli/generate_evaluation_site.py's main() shouldn't observe a
    lasting change to which corpus is selected."""
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_names = corpus.list_corpora()
    previous = corpus.corpus_dir().name
    try:
        for name in corpus_names:
            corpus.set_corpus(name)
            (output_dir / f"{name}.html").write_text(render_corpus_html(collect_corpus_data()), encoding="utf-8")
    finally:
        corpus.set_corpus(previous)
    (output_dir / "index.html").write_text(render_index_html(corpus_names), encoding="utf-8")
    (output_dir / ".nojekyll").write_text("", encoding="utf-8")
