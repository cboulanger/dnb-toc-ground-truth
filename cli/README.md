# CLI scripts reference

One-line description plus a full `--help` dump for every script in this
directory, alphabetically. Regenerate an entry by running
`uv run python cli/<name>.py --help` whenever that script's arguments
change.

Every script below except `generate_evaluation_site.py` (which builds
pages for every corpus at once) takes a shared `--corpus` flag selecting
which `data/corpus/<name>/` directory it operates on -- defaulting to
`.config.json`'s `"corpus"` key, then `"pilot"`. See
`dnb_toc_ground_truth.corpus.set_corpus`/`list_corpora`.

## `arbitrate.py`

Surfaces books whose model reads didn't clear `generate_ground_truth.py`'s
agreement gate, so an AI agent (or human) session can arbitrate the conflict
directly -- see `AGENTS.md`'s "Arbitrating below-gate books". This
script only REPORTS and records rejections; it never decides.

```
usage: arbitrate.py [-h] [--corpus CORPUS] [--config-file CONFIG_FILE]
                    {list,reject} ...

Surfaces pilot-corpus books whose two vision-model TOC extractions didn't
clear cli/generate_ground_truth.py's agreement gate, so a strong, multimodal
AI agent (such as Claude) can arbitrate the conflict directly -- see design
spec docs/superpowers/specs/2026-08-16-dnb-toc-arbitration-design.md. This
script only REPORTS and records rejections; it never decides. The arbitrator
reads a book's report, opens the PDF's actual TOC pages via the Read tool when
the text alone doesn't settle it, then either writes data/corpus/pilot/ground-
truth/<key>.expected.json directly (same schema as a passing book, "verified":
true) or runs this script's `reject` subcommand to permanently record the book
as unrecoverable.

positional arguments:
  {list,reject}
    list                List books needing arbitration (default)
    reject              Permanently mark a book as unrecoverable

options:
  -h, --help            show this help message and exit
  --corpus CORPUS       Corpus to operate on (default: config file's "corpus",
                        or 'pilot')
  --config-file CONFIG_FILE
                        Path to the config file (default: .config.json)
```

## `backfill_crossref.py`

Backfills Crossref book DOI and cached chapter data for existing
manifest entries that already have ground truth but no DOI yet, and
writes each book's filtered evaluation-corpus entry alongside it.

```
usage: backfill_crossref.py [-h] [--force] [--contact-email CONTACT_EMAIL]
                            [--config-file CONFIG_FILE] [--corpus CORPUS]
                            [--min-chapters MIN_CHAPTERS]

Backfills Crossref book DOI and chapter data for existing manifest.json
entries that already have a .expected.json ground-truth file but no doi yet --
see design spec docs/superpowers/specs/2026-08-21-crossref-cross-validation-
design.md.

options:
  -h, --help            show this help message and exit
  --force               Re-query Crossref even for an already-cached ISBN
  --contact-email CONTACT_EMAIL
                        Crossref polite-pool contact email (default: config
                        file's "contact_email")
  --config-file CONFIG_FILE
                        Path to the config file (default: .config.json)
  --corpus CORPUS       Corpus to operate on (default: config file's "corpus",
                        or 'pilot')
  --min-chapters MIN_CHAPTERS
                        Minimum page-numbered Crossref chapters a book needs
                        before its evaluation-corpus entry is written
                        (default: 3)
```

## `corpus_status.py`

Prints manifest/ground-truth/per-model/arbitration-backlog/evaluation
coverage counts as a markdown table and rewrites it into the top-level
`README.md`'s "Current status" section in place -- see `AGENTS.md`.

```
usage: corpus_status.py [-h] [--check] [--readme README] [--corpus CORPUS]
                        [--config-file CONFIG_FILE]

Prints corpus coverage counts -- manifest size, ground-truth coverage (and how
it was produced), per-model LLM-cache reading counts, outstanding
arbitration/rejection backlog, the held-out eval tier, and Crossref
evaluation-corpus size -- as a markdown table, and rewrites it into
README.md's "Current status" section in place.

options:
  -h, --help            show this help message and exit
  --check               Print the table but don't write README.md; exit 1 if
                        README.md's current section is stale
  --readme README       Path to the README to update (default: this repo's
                        top-level README.md)
  --corpus CORPUS       Corpus to operate on (default: config file's "corpus",
                        or 'pilot')
  --config-file CONFIG_FILE
                        Path to the config file (default: .config.json)
```

## `evaluate_crossref.py`

Measures precision/recall/F1 between this corpus's ground truth (and,
optionally, one or more llm-cache models' raw extractions) and its
committed Crossref evaluation corpus, reusing `matching.diff_toc_entries`.
See the "Crossref evaluation" section in the top-level `README.md` for
the methodology and its constraints.

```
usage: evaluate_crossref.py [-h] [--full] [--min-f1 MIN_F1] [--model MODEL]
                            [--all-models] [--corpus CORPUS]
                            [--config-file CONFIG_FILE]

Measures precision/recall/F1 between this corpus's own ground truth and its
committed Crossref evaluation corpus, plus optionally one or more llm-cache
models' raw extractions -- see dnb_toc_ground_truth.crossref_evaluation for
the full methodology.

options:
  -h, --help            show this help message and exit
  --full                Print a per-book precision/recall/F1 line for every
                        compared book, not just the aggregate mean
  --min-f1 MIN_F1       Exit 1 if the aggregate mean ground-truth F1 falls
                        below this (0-1). Unset: no gate enforced.
  --model MODEL         Also score this model's cached llm-cache extraction
                        against the crossref sample, alongside ground truth
                        (repeatable). Model id as it appears in
                        data/corpus/pilot/llm-cache/v2/ filenames (e.g.
                        'Qwen/Qwen3-Omni-30B-A3B-Instruct' or its sanitized
                        'Qwen__Qwen3-Omni-30B-A3B-Instruct' form).
  --all-models          Score every model with at least one llm-cache entry
                        for a crossref-sample book, without naming them
                        individually. Combines with --model.
  --corpus CORPUS       Corpus to operate on (default: config file's "corpus",
                        or 'pilot')
  --config-file CONFIG_FILE
                        Path to the config file (default: .config.json)
```

## `fetch_corpus.py`

Acquires real DNB-scanned table-of-contents PDFs via the lobid-resources
API into `data/corpus/pilot/pdf/`.

```
usage: fetch_corpus.py [-h] (--from-dump | --isbns-file ISBNS_FILE)
                       [--dump-url DUMP_URL] [--limit LIMIT]
                       [--rate-limit-seconds RATE_LIMIT_SECONDS]
                       [--max-retries MAX_RETRIES]
                       [--manifest-path MANIFEST_PATH]
                       [--contact-email CONTACT_EMAIL]
                       [--crossref-cache-dir CROSSREF_CACHE_DIR]
                       [--config-file CONFIG_FILE] [--corpus CORPUS]
                       [--min-chapters MIN_CHAPTERS]

Acquires real DNB-scanned table-of-contents PDFs via the lobid-resources
API (lobid.org/resources) into data/corpus/pilot/ -- see
docs/superpowers/specs/2026-08-14-dnb-toc-corpus-acquisition-design.md.

options:
  -h, --help            show this help message and exit
  --from-dump           Scan the full lobid-resources JSON-Lines dump for
                        matching records (hours-long; see module docstring)
  --isbns-file ISBNS_FILE
                        Path to a text file of ISBNs (one per line, '#'
                        comments allowed) to look up individually
  --dump-url DUMP_URL   lobid-resources dump URL for --from-dump (default:
                        https://lobid.org/download/dumps/lobid-
                        resources/latestLobidResources.jsonl.gz)
  --limit LIMIT         Stop after acquiring this many new books
  --rate-limit-seconds RATE_LIMIT_SECONDS
                        Delay after each TOC PDF download, to stay polite to
                        DNB's servers (default: 1.0)
  --max-retries MAX_RETRIES
                        For --from-dump: how many times to reconnect and
                        rescan after a dropped connection before giving up
                        (default: 5)
  --manifest-path MANIFEST_PATH
                        Override where manifest.json entries are read from and
                        appended to (default:
                        data/corpus/pilot/manifest.json). PDFs and .lobid-
                        cache/ always go to the real corpus directory
                        regardless -- only the tracked manifest write is
                        redirectable, e.g. to a scratch copy that gets merged
                        into the real manifest.json once a long run finishes,
                        without touching the committed file mid-run.
  --contact-email CONTACT_EMAIL
                        Crossref polite-pool contact email (default: config
                        file's "contact_email")
  --crossref-cache-dir CROSSREF_CACHE_DIR
                        Override where Crossref DOI/chapter data is cached
                        (default: data/corpus/pilot/.crossref-cache/)
  --config-file CONFIG_FILE
                        Path to the config file (default: .config.json)
  --corpus CORPUS       Corpus to operate on (default: config file's "corpus",
                        or 'pilot'). If the named corpus has no
                        data/corpus/<name>/manifest.json yet, one is created
                        -- this is how a brand-new corpus gets bootstrapped.
  --min-chapters MIN_CHAPTERS
                        Minimum page-numbered Crossref chapters a book needs
                        before its evaluation-corpus entry is written
                        (default: 3)
```

## `generate_evaluation_site.py`

Builds the static GitHub Pages site presenting crossref-evaluation scores
(ground truth plus every cached model) for every corpus
(`dnb_toc_ground_truth.corpus.list_corpora()`) -- not just one, so it has
no `--corpus` flag of its own. Reads only committed corpus data -- no
network access, no secrets required. Run automatically by
`.github/workflows/pages.yml` on every push to `main`.

```
usage: generate_evaluation_site.py [-h] [--output-dir OUTPUT_DIR]

Builds the static GitHub Pages site presenting crossref-evaluation scores --
see dnb_toc_ground_truth.evaluation_site for the page-building logic and
.github/workflows/pages.yml for how this is run and deployed on every push to
main. Reads only committed corpus data (ground truth, the committed Crossref
evaluation corpus, and llm-cache) -- no network access, no secrets required.

options:
  -h, --help            show this help message and exit
  --output-dir OUTPUT_DIR
                        Directory to write the site into (default: _site/)
```

## `generate_ground_truth.py`

Generates bulk-tier ground truth by sending each book's TOC pages to
every model named via `--use-vision`/`--use-text`, writing a
ground-truth file only when at least two of the resulting reads agree
well enough.

```
usage: generate_ground_truth.py [-h] [--limit LIMIT]
                                [--concurrency CONCURRENCY] [--spot-check N]
                                [--use-vision MODEL[,MODEL...]]
                                [--use-text MODEL[,MODEL...]]
                                [--endpoints-file ENDPOINTS_FILE]
                                [--config-file CONFIG_FILE]
                                [--gate-threshold GATE_THRESHOLD]
                                [--corpus CORPUS]

Generates structured ground truth for the dnb-toc-only pilot corpus. For every
manifest book not held out in eval_tier_ids.json (see select_eval_sample.py),
not already carrying a ground-truth JSON file (bulk-gated or arbitrated), and
not permanently rejected (arbitration-rejected.json), sends the book's TOC
pages to every model named via --use-vision/--use-text (resolved against
--endpoints-file), and writes a ground-truth file only when at least two of
the resulting reads agree well enough
(dnb_toc_ground_truth.matching.gate_books, >=0.90 agreement between the best-
agreeing pair) -- see design spec docs/superpowers/specs/2026-08-21-dnb-toc-
ground-truth-extraction-design.md. Books that don't clear the gate are skipped
and reported, not partially written -- run cli/arbitrate.py on them next.

options:
  -h, --help            show this help message and exit
  --limit LIMIT         Process at most this many books (smoke-test
                        convenience)
  --concurrency CONCURRENCY
                        How many books to process concurrently (default: 4, or
                        config file's "concurrency")
  --spot-check N        Instead of generating, sample N passing bulk-tier
                        books and walk through a visual Accept/Reject check
  --use-vision MODEL[,MODEL...]
                        Model id(s) to resolve against --endpoints-file for
                        the VISION side -- required (directly or via config),
                        at least one
  --use-text MODEL[,MODEL...]
                        Model id(s) to resolve against --endpoints-file for
                        the TEXT (OCR'd) side -- optional
  --endpoints-file ENDPOINTS_FILE
                        Path to the endpoints file (default: .endpoints, or
                        config file's "endpoints_file")
  --config-file CONFIG_FILE
                        Path to the config file (default: .config.json)
  --gate-threshold GATE_THRESHOLD
                        Whole-book agreement threshold, 0-1 (default: 0.90, or
                        config file's "gate_threshold")
  --corpus CORPUS       Corpus to operate on (default: config file's "corpus",
                        or 'pilot')
```

## `select_eval_sample.py`

Selects a stratified held-out eval-tier sample, so it isn't accidentally
dominated by one publication era or language.

```
usage: select_eval_sample.py [-h] [--sample-size SAMPLE_SIZE] [--seed SEED]
                             [--corpus CORPUS] [--config-file CONFIG_FILE]

Selects a stratified held-out eval-tier sample for the pilot corpus (design
spec docs/superpowers/specs/2026-08-15-dnb-toc-ground-truth-generation-
design.md section 5). Reads each candidate book's .lobid-cache/<id>.lobid.json
for its publication decade and the manifest's language field, and draws a
sample whose decade/language spread mirrors the corpus's own -- so the held-
out eval tier used to score NuExtract fine-tuning, the heuristic line-parsing
harness, and the classifier pilot isn't accidentally dominated by one era or
language.

options:
  -h, --help            show this help message and exit
  --sample-size SAMPLE_SIZE
  --seed SEED
  --corpus CORPUS       Corpus to operate on (default: config file's "corpus",
                        or 'pilot')
  --config-file CONFIG_FILE
                        Path to the config file (default: .config.json)
```
