"""Builds the static GitHub Pages site presenting crossref-evaluation
scores (dnb_toc_ground_truth.crossref_evaluation) for every corpus this
repo has (corpus.list_corpora()) -- see cli/generate_evaluation_site.py
for the CLI wrapper this feeds, and .github/workflows/pages.yml for how
the site is built and deployed on every push to main."""

import html as _html
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dnb_toc_ground_truth import corpus, vision
from dnb_toc_ground_truth.crossref_evaluation import (
    BookMetrics,
    discover_cached_models,
    evaluate_corpus,
    evaluate_model_corpus,
)
from dnb_toc_ground_truth.model_agreement import (
    ModelGroundTruthMetrics,
    PairAgreement,
    PairCandidateScore,
    arbitration_ground_truth_agreement,
    discover_all_cached_models,
    pairwise_model_agreement,
    rank_candidate_pairs,
)

# Mirrors `git remote get-url origin` -- not derived at runtime since the
# generator has no other reason to shell out to git, and this repo isn't
# expected to be forked under a different owner/name while this site is
# still relevant.
_GITHUB_REPO_BASE = "https://github.com/cboulanger/dnb-toc-ground-truth"


@dataclass(frozen=True)
class SourceScores:
    label: str
    is_ground_truth: bool
    results: list[BookMetrics]
    uncovered_count: int
    # None for ground truth, else the model id passed to
    # evaluate_model_corpus() -- lets _render_source_section() find the
    # right llm-cache file to link each row's score to.
    model: Optional[str] = None


@dataclass(frozen=True)
class CorpusData:
    name: str
    titles: dict[str, str]
    toc_urls: dict[str, str]
    sources: list[SourceScores]


@dataclass(frozen=True)
class ModelComparisonData:
    name: str
    models: list[str]
    agreements: list[PairAgreement]
    gt_metrics: list[ModelGroundTruthMetrics]
    crossref_results: dict[str, tuple[list[BookMetrics], list[str]]]
    scored_pairs: list[PairCandidateScore]
    unscored_pairs: list[tuple[str, str]]


def collect_corpus_data() -> CorpusData:
    """Runs the ground-truth crossref evaluation plus every discovered
    llm-cache model's, for corpus.py's CURRENTLY SELECTED corpus
    (corpus.set_corpus()) -- callers that want a specific corpus must
    call corpus.set_corpus(name) first, same as any other corpus.*
    consumer."""
    books = corpus.load_manifest_books()
    titles = {corpus.manifest_key(book): book.get("title", "") for book in books}
    toc_urls = {corpus.manifest_key(book): book["toc_download_url"] for book in books if book.get("toc_download_url")}
    gt_results, gt_uncovered = evaluate_corpus()
    sources = [
        SourceScores(label="Ground truth", is_ground_truth=True, results=gt_results, uncovered_count=len(gt_uncovered)),
    ]
    for model in discover_cached_models():
        model_results, no_cache = evaluate_model_corpus(model)
        sources.append(SourceScores(
            label=model, is_ground_truth=False, results=model_results, uncovered_count=len(no_cache), model=model,
        ))
    return CorpusData(name=corpus.corpus_dir().name, titles=titles, toc_urls=toc_urls, sources=sources)


def collect_model_comparison_data() -> ModelComparisonData:
    """Same "operates on corpus.py's CURRENTLY SELECTED corpus" contract
    as collect_corpus_data()."""
    models = discover_all_cached_models()
    agreements = pairwise_model_agreement(models)
    gt_metrics = arbitration_ground_truth_agreement(models)
    crossref_results = {model: evaluate_model_corpus(model) for model in models}
    scored_pairs, unscored_pairs = rank_candidate_pairs(agreements, gt_metrics)
    return ModelComparisonData(
        name=corpus.corpus_dir().name, models=models, agreements=agreements, gt_metrics=gt_metrics,
        crossref_results=crossref_results, scored_pairs=scored_pairs, unscored_pairs=unscored_pairs,
    )


def _source_data_path(key: str, model: Optional[str]) -> Path:
    """The committed JSON file that produced one book's score for a
    source -- ground-truth's own .expected.json, or the exact llm-cache
    file evaluate_model_corpus() read for a model."""
    if model is None:
        return corpus.expected_json_path(key)
    return vision.cache_path(corpus.llm_cache_dir(), key, model)


def _blob_url(path: Path) -> str:
    """A GitHub "view this file" URL for a path inside the repo
    checkout -- lets a score in the rendered site link back to the exact
    committed JSON that produced it."""
    return f"{_GITHUB_REPO_BASE}/blob/main/{path.relative_to(corpus.repo_root())}"


def _model_comparison_filename(name: str) -> str:
    """The output filename for a corpus's model-comparison page --
    single source of truth so render_index_html, render_corpus_html,
    and write_site can never drift out of sync on this convention."""
    return f"{name}-model-comparison.html"


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
tr.ground-truth td { font-weight: 700; }
section.ground-truth h2 { color: #0a5c2b; }
section.ground-truth table { font-weight: 700; }
.caveats { background: #fff8e6; border: 1px solid #e8d488; border-radius: 6px; padding: 0.9rem 1.2rem; }
.caveats ul { margin: 0.4rem 0 0; padding-left: 1.2rem; }
.coverage { color: #555; font-size: 0.9rem; margin: -1rem 0 1rem; }
footer { color: #777; font-size: 0.85rem; margin-top: 3rem; }
"""


def _link(href: str, text: str, *, escape_href: bool = True, new_tab: bool = True) -> str:
    """new_tab=True (the default) opens in a new tab, with rel="noopener
    noreferrer" alongside target="_blank" so the new tab can't reach
    back into this page via window.opener -- for every link that takes
    the reader off this site's own pages (Crossref, a book's original
    TOC scan host, a GitHub blob). new_tab=False for links between this
    site's own pages (index <-> a corpus page) -- plain in-tab
    navigation, since a reader moving around the site itself shouldn't
    pile up tabs for it. escape_href=False for a href already built from
    html.escape'd or known-safe pieces (avoids double-escaping)."""
    safe_href = _html.escape(href) if escape_href else href
    target = ' target="_blank" rel="noopener noreferrer"' if new_tab else ""
    return f'<a href="{safe_href}"{target}>{text}</a>'


# Tooltip text for each metric column -- shared by _metric_header_cells()
# so both the per-source tables and the top-of-page overview table explain
# their columns identically instead of drifting apart.
_METRIC_TOOLTIPS = {
    "Precision": "Share of this source's chapters that match a Crossref-registered chapter: TP / (TP + FP).",
    "Recall": "Share of Crossref-registered chapters this source correctly found: TP / (TP + FN).",
    "F1": "Harmonic mean of precision and recall.",
    "TP": "True positives -- chapters that matched between this source and Crossref.",
    "FP": "False positives -- chapters this source has that Crossref does not.",
    "FN": "False negatives -- Crossref-registered chapters this source is missing.",
}


def _th(label: str, *, tooltip: Optional[str] = None, num: bool = False) -> str:
    css_class = ' class="num"' if num else ""
    title_attr = f' title="{_html.escape(tooltip)}"' if tooltip else ""
    return f"<th{css_class}{title_attr}>{label}</th>"


def _metric_header_cells() -> str:
    """The <th> cells shared by every table that reports per-book or
    mean precision/recall/F1/TP/FP/FN -- keeps the column tooltips
    defined once in _METRIC_TOOLTIPS instead of duplicated per table."""
    return "".join(_th(label, tooltip=tooltip, num=True) for label, tooltip in _METRIC_TOOLTIPS.items())


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
    links = "\n".join(
        f"<li>{_link(f'{name}.html', _html.escape(name), new_tab=False)} "
        f"({_link(_model_comparison_filename(name), 'model comparison', new_tab=False)})</li>"
        for name in corpus_names
    )
    body = f"""<h1>Crossref evaluation</h1>
<p>This project cross-checks its ground truth -- and, per book, each
vision-LLM's raw table-of-contents read -- against an independent,
non-LLM data source: {_link("https://www.crossref.org/", "Crossref")}'s
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


def _render_book_cell(key: str, title: str, toc_urls: dict[str, str]) -> str:
    text = _html.escape(title or key)
    toc_url = toc_urls.get(key)
    if toc_url is None:
        return f"<td>{text}</td>"
    return f"<td>{_link(toc_url, text)}</td>"


def _render_source_section(source: SourceScores, titles: dict[str, str], toc_urls: dict[str, str]) -> str:
    section_class = "ground-truth" if source.is_ground_truth else "model"
    rows = sorted(source.results, key=lambda r: (-r.f1, titles.get(r.key, r.key).lower(), r.key))
    row_html = "\n".join(
        f"<tr>{_render_book_cell(r.key, titles.get(r.key, r.key), toc_urls)}"
        f'<td class="num">{r.precision:.0%}</td><td class="num">{r.recall:.0%}</td><td class="num">{r.f1:.0%}</td>'
        f'<td class="num">{r.tp}</td><td class="num">{r.fp}</td><td class="num">{r.fn}</td>'
        f'<td>{_link(_blob_url(_source_data_path(r.key, source.model)), "JSON")}</td></tr>'
        for r in rows
    )
    if source.results:
        mean_precision = _mean([r.precision for r in source.results])
        mean_recall = _mean([r.recall for r in source.results])
        mean_f1 = _mean([r.f1 for r in source.results])
        mean_row = (
            f'<tr class="mean"><td>Mean ({len(source.results)} book(s))</td>'
            f'<td class="num">{mean_precision:.0%}</td><td class="num">{mean_recall:.0%}</td>'
            f'<td class="num">{mean_f1:.0%}</td><td class="num">-</td><td class="num">-</td><td class="num">-</td>'
            f'<td>-</td></tr>'
        )
        table = f"""<table>
<thead><tr><th>Book</th>{_metric_header_cells()}<th>Data</th></tr></thead>
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


def _render_overview_table(sources: list[SourceScores]) -> str:
    """The bottom ("mean") row of every per-source table in one place, so
    a reader can compare ground truth against every model at a glance
    without opening each section below. Sources with no results (nothing
    to average) are left out rather than shown as an empty/zero row."""
    rows = []
    for source in sources:
        if not source.results:
            continue
        row_class = ' class="ground-truth"' if source.is_ground_truth else ""
        mean_precision = _mean([r.precision for r in source.results])
        mean_recall = _mean([r.recall for r in source.results])
        mean_f1 = _mean([r.f1 for r in source.results])
        rows.append(
            f'<tr{row_class}><td>{_html.escape(source.label)} ({len(source.results)} book(s))</td>'
            f'<td class="num">{mean_precision:.0%}</td><td class="num">{mean_recall:.0%}</td>'
            f'<td class="num">{mean_f1:.0%}</td>'
            f'<td class="num">-</td><td class="num">-</td><td class="num">-</td></tr>'
        )
    row_html = "\n".join(rows)
    if not row_html:
        return ""
    return f"""<h2>Overview</h2>
<table>
<thead><tr><th>Source</th>{_metric_header_cells()}</tr></thead>
<tbody>
{row_html}
</tbody>
</table>"""


def render_corpus_html(data: CorpusData) -> str:
    overview = _render_overview_table(data.sources)
    sections = "\n".join(_render_source_section(source, data.titles, data.toc_urls) for source in data.sources)
    body = f"""<p>{_link("index.html", "&larr; Corpora", new_tab=False)}</p>
<h1>{_html.escape(data.name)}</h1>
<p>Per-book precision/recall/F1 against the committed Crossref evaluation
corpus. Ground truth is the project's own committed data (highlighted
below); each model section scores that model's raw, pre-agreement-gate
llm-cache extraction over the same books.</p>
<p>{_link(_model_comparison_filename(data.name), "Model comparison &rarr;", new_tab=False)}</p>
{overview}
{sections}
"""
    return _page(f"{data.name} -- Crossref evaluation", body)


def _heatmap_color(rate: float) -> str:
    """White (0%) -> a mid green (100%), linearly interpolated -- the
    printed "NN% (n=NNN)" text is always the primary read, this is a
    secondary visual cue only."""
    r = round(255 + (46 - 255) * rate)
    g = round(255 + (125 - 255) * rate)
    b = round(255 + (50 - 255) * rate)
    return f"rgb({r},{g},{b})"


def _render_agreement_heatmap(models: list[str], agreements: list[PairAgreement]) -> str:
    by_pair = {frozenset((a.model_a, a.model_b)): a for a in agreements}
    header = "".join(f"<th>{_html.escape(m)}</th>" for m in models)
    rows = []
    for row_model in models:
        cells = []
        for col_model in models:
            if row_model == col_model:
                cells.append("<td></td>")
                continue
            pair = by_pair.get(frozenset((row_model, col_model)))
            if pair is None:
                cells.append('<td class="num">-</td>')
            else:
                color = _heatmap_color(pair.mean_agreement)
                cells.append(f'<td class="num" style="background:{color}">{pair.mean_agreement:.0%} (n={pair.n_books})</td>')
        rows.append(f"<tr><th>{_html.escape(row_model)}</th>{''.join(cells)}</tr>")
    return f"""<div style="overflow-x:auto">
<table>
<thead><tr><th></th>{header}</tr></thead>
<tbody>
{''.join(rows)}
</tbody>
</table>
</div>"""


def _render_model_accuracy_table(
    models: list[str], gt_metrics: list[ModelGroundTruthMetrics],
    crossref_results: dict[str, tuple[list[BookMetrics], list[str]]],
) -> str:
    gt_by_model = {m.model: m for m in gt_metrics}
    ordered = sorted(models, key=lambda m: -(gt_by_model[m].f1 if m in gt_by_model else -1.0))
    rows = []
    for model in ordered:
        gt = gt_by_model.get(model)
        gt_cells = (
            f'<td class="num">{gt.precision:.0%}</td><td class="num">{gt.recall:.0%}</td>'
            f'<td class="num">{gt.f1:.0%}</td><td class="num">{gt.n_books}</td>'
            if gt else '<td class="num">-</td><td class="num">-</td><td class="num">-</td><td class="num">-</td>'
        )
        cr_results, _ = crossref_results.get(model, ([], []))
        if cr_results:
            mean_f1 = _mean([r.f1 for r in cr_results])
            cr_cells = (
                f'<td class="num">{_mean([r.precision for r in cr_results]):.0%}</td>'
                f'<td class="num">{_mean([r.recall for r in cr_results]):.0%}</td>'
                f'<td class="num">{mean_f1:.0%}</td><td class="num">{len(cr_results)}</td>'
            )
        else:
            cr_cells = '<td class="num">-</td><td class="num">-</td><td class="num">-</td><td class="num">-</td>'
        bar_cell = (
            f'<td><div style="width:{gt.f1:.0%}; background:#2e7d32; height:0.5rem;"></div></td>'
            if gt else "<td></td>"
        )
        rows.append(
            f"<tr><td>{_html.escape(model)}</td>{gt_cells}{cr_cells}{bar_cell}</tr>"
        )
    return f"""<table>
<thead><tr><th>Model</th><th class="num">Arb. P</th><th class="num">Arb. R</th><th class="num">Arb. F1</th>
<th class="num">Arb. N</th><th class="num">Crossref P</th><th class="num">Crossref R</th>
<th class="num">Crossref F1</th><th class="num">Crossref N</th><th title="Visualizes the Arb. F1 column">Arb. F1 (bar)</th></tr></thead>
<tbody>
{''.join(rows)}
</tbody>
</table>"""


def _render_candidate_ranking_table(scored_pairs: list[PairCandidateScore], unscored_pairs: list[tuple[str, str]]) -> str:
    if not scored_pairs:
        table = "<p>No pair has arbitration-GT coverage for both models yet.</p>"
    else:
        rows = "\n".join(
            f"<tr><td>{_html.escape(p.model_a)}</td><td>{_html.escape(p.model_b)}</td>"
            f'<td class="num">{p.f1_a:.0%}</td><td class="num">{p.f1_b:.0%}</td>'
            f'<td class="num">{p.observed_agreement:.0%}</td><td class="num">{p.expected_agreement:.0%}</td>'
            f'<td class="num">{p.kappa:+.2f}</td><td class="num">{p.score:+.2f}</td><td class="num">{p.n_books}</td></tr>'
            for p in scored_pairs
        )
        table = f"""<table>
<thead><tr><th>Model A</th><th>Model B</th><th class="num">F1 A</th><th class="num">F1 B</th>
<th class="num">Observed</th><th class="num">Expected</th><th class="num">Kappa</th><th class="num">Score</th>
<th class="num">N</th></tr></thead>
<tbody>
{rows}
</tbody>
</table>"""
    if unscored_pairs:
        items = "".join(f"<li>{_html.escape(a)} + {_html.escape(b)}</li>" for a, b in unscored_pairs)
        table += f"<p>Unscored pairs (missing arbitration-GT coverage for one or both models):</p><ul>{items}</ul>"
    return table


def render_model_comparison_html(data: ModelComparisonData) -> str:
    back_links = (
        f'<p>{_link("index.html", "&larr; Corpora", new_tab=False)} '
        f'{_link(f"{data.name}.html", f"&larr; {_html.escape(data.name)}", new_tab=False)}</p>'
    )
    if not data.models:
        body = f"""{back_links}
<h1>{_html.escape(data.name)} -- Model comparison</h1>
<p>No cached model readings found for this corpus.</p>
"""
        return _page(f"{data.name} -- Model comparison", body)
    body = f"""{back_links}
<h1>{_html.escape(data.name)} -- Model comparison</h1>
<p>Corpus-level metrics for choosing which two models to pair for the
bulk-tier two-model agreement gate: how much two models' raw TOC
readings agree with each other, how close each model is to
arbitration-sourced ground truth (Claude-transcribed directly from the
TOC page images, independent of any model's own raw reading) and to an
independent Crossref cross-check, and a derived ranking of candidate
pairs.</p>
<div class="caveats">
<strong>Caveats -- read before trusting a number below:</strong>
<ul>
<li><strong>Agreement measures how much two models say the same
thing, not whether either is right.</strong> Two models sharing the
same systematic blind spot (e.g. the same architecture family) would
still show high agreement.</li>
<li><strong>A HIGH kappa is the specific red flag for that failure
mode</strong> -- agreement in excess of what each model's own
arbitration-GT accuracy already predicts under independent errors is
exactly why the candidate-pair ranking penalizes it rather than
simply sorting by raw agreement. The kappa calculation is a
simplification (it treats each model's arbitration-GT F1 as a flat
per-entry "probability of being correct"), so read it directionally,
not as a precise probability.</li>
<li><strong>The ranking's score is NOT weighted by how many books it's
based on.</strong> A pair backed by only a handful of shared/arbitrated
books can outrank a pair backed by hundreds -- always check each row's
N alongside its score, and treat a low-N score as provisional.</li>
<li><strong>Arbitration-GT coverage differs per model and grows over
time</strong> as more of the corpus gets arbitrated -- a low N means a
score is based on fewer books and can shift as more books are
arbitrated.</li>
<li><strong>Crossref coverage is small and skewed</strong> toward
larger, more prominent publishers -- see this corpus's main page for
the full caveat.</li>
</ul>
</div>
<h2>Pairwise agreement</h2>
{_render_agreement_heatmap(data.models, data.agreements)}
<h2>Per-model accuracy</h2>
{_render_model_accuracy_table(data.models, data.gt_metrics, data.crossref_results)}
<h2>Candidate pair ranking</h2>
{_render_candidate_ranking_table(data.scored_pairs, data.unscored_pairs)}
"""
    return _page(f"{data.name} -- Model comparison", body)


def write_site(output_dir: Path) -> None:
    """Renders index.html plus one <name>.html and one
    <name>-model-comparison.html per corpus.list_corpora() entry.
    Temporarily switches corpus.py's selected corpus (via
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
            (output_dir / _model_comparison_filename(name)).write_text(
                render_model_comparison_html(collect_model_comparison_data()), encoding="utf-8",
            )
    finally:
        corpus.set_corpus(previous)
    (output_dir / "index.html").write_text(render_index_html(corpus_names), encoding="utf-8")
    (output_dir / ".nojekyll").write_text("", encoding="utf-8")
