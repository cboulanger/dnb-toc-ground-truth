# MPCDF LLM Inference Service -- empirical notes

Web UI at `https://llm.mpcdf.mpg.de`. This is a self-service web
dashboard that spins up a per-user, per-session vLLM (or Ollama) server
on Viper-GPU (or DAIS) as a SLURM job under the hood, for a user-chosen
duration (up to 8h), exposing an OpenAI-compatible chat-completions API.
Written up here because none of this is documented anywhere obvious and
every item below was learned the hard way, live, against real sessions
(2026-08-18/19) integrating it as an inference endpoint for
`cli/generate_ground_truth.py` (see
`docs/superpowers/specs/2026-08-18-inference-endpoint-abstraction-design.md`).

## Endpoint URL convention

Each spawned session gets its own base URL of the form
`https://llm.mpcdf.mpg.de/<session-hash>/v1` and its own API key --
**the `/v1` suffix is required and not obvious from the dashboard UI**;
confirmed by curling the bare session-hash URL first (404) then
`/v1/models` (real JSON). `dnb_toc_ground_truth.inference`'s endpoints-
file loader normalizes a `url` missing the `/v1` suffix by appending it
automatically -- see `.endpoints`/`.endpoints.dist` below.

## Faster setup: paste the dashboard session table directly

Every endpoint this project can call -- however many models you're
running -- lives in one `--endpoints-file` (default `.endpoints`, see
`.endpoints.dist` for the shape). Paste each session row exactly as
copied from the dashboard (tab-separated `field<TAB>value` lines, no
reformatting) as one array entry, or use the plain-text pasted-table
format (one or more blocks separated by a blank line) -- both are
auto-detected by `dnb_toc_ground_truth.inference.load_endpoint_entries`.
`url`, `key`, and the `--model=` value inside `framework_args` (or an
explicit `model` field) are extracted automatically. Which models a
given run actually calls is then named explicitly via
`--use-vision`/`--use-text` (matched against each entry's `model` id) --
not by file position, so an `.endpoints` file can carry more sessions
than any one invocation uses.

Example `.endpoints` (plain-text pasted-table format) with two sessions:

```text
framework	vLLM
framework_args	--model=mistralai/Pixtral-12B-2409 --tensor-parallel-size=2 --trust-remote-code
host	10.179.7.234:24100
key	<key-a>
url	https://llm.mpcdf.mpg.de/y382b105ryopxy89/v1

framework	vLLM
framework_args	--model=Qwen/Qwen3-Omni-30B-A3B-Instruct --tensor-parallel-size=2 --trust-remote-code
host	10.179.7.235:24100
key	<key-b>
url	https://llm.mpcdf.mpg.de/yv19vgf4wyyab90l/v1
```

Then run against it, naming the two model ids explicitly:

```bash
uv run python cli/generate_ground_truth.py \
  --use-vision mistralai/Pixtral-12B-2409,Qwen/Qwen3-Omni-30B-A3B-Instruct --limit 50
```

(`--endpoints-file` defaults to `.endpoints` in the repo root; pass a
path explicitly, e.g. `--endpoints-file /tmp/other-endpoints.json`, to
use a different file.)

## Dashboard "Running" ≠ API ready

The dashboard's session table can show a job as `Running` (SLURM job
started, GPUs allocated, remaining-time counting down) well before the
vLLM server inside has actually finished downloading/loading the model
and started answering requests. **An inactive or not-yet-ready session's
`/v1/models` returns the generic dashboard SPA's static `index.html`
fallback page -- with HTTP 200, not an error.** Both a genuinely dead
session and a live-but-still-loading one look identical at the HTTP
status-code level; the only way to tell is to inspect the actual
response: `content-type: text/html` + `server: Caddy` + a suspiciously
old `last-modified` (a static asset date, not live inference output)
means "not really up yet," vs. a real `{"object": "list", "data": [...]}`
JSON body with the actual model id in it. A ~70GB checkpoint (e.g.
`Qwen/Qwen3-Omni-30B-A3B-Instruct`) can take several minutes past
"Running" to finish loading before it answers for real.

Weight downloads are cached across sessions on MPCDF's side -- the
first spawn of a given model can take up to an hour if the weights
aren't cached yet; respawning the *same* model afterward (even a fresh
session/job) took as little as ~5 minutes.

## A session's reliability can crater mid-run, and lowering client concurrency does not fix it

Observed 2026-08-23/24 backfilling `numind/NuExtract3` and
`mistralai/Mistral-Small-3.2-24B-Instruct-2506` readings onto ~300+
already-ground-truthed books each (`cli/generate_ground_truth.py`'s
`--keys-file` backfill mode, see `AGENTS.md`) -- both sessions started
healthy and got noticeably less reliable over the following hours, but
in two different shapes worth telling apart:

- **`Mistral-Small-3.2-24B-Instruct-2506`'s failure rate was not a
  fixed set of hard books -- it was the session itself.** Four
  successive passes over the same shrinking leftover pool, at
  `--concurrency` 4, 4, 2, then serial 1, gave real-attempt success
  rates of **97.6% (83/85), 18.4% (41/223), 31.3% (57/182), 16.0%
  (20/125)** -- almost every failure past the first pass was
  `_call_with_retry`'s 450s hard wall-clock timeout, not a request
  error. If specific books were simply hard, success should climb
  monotonically as the pool shrinks to just the easy remainder; instead
  round 3 (31.3%) beat round 2 (18.4%) on a strictly harder residual
  pool, and cutting concurrency all the way to 1 (round 4) didn't
  recover round-1 reliability either. That non-monotonic, concurrency-
  insensitive pattern points at the vLLM server itself having entered a
  degraded state early in the session (a wedged worker or fragmented
  KV cache from continuous batching over many large multi-image
  requests is the likely mechanism, though not confirmed against
  server-side logs) rather than at any particular book. **Direct probe
  requests bypassing the CLI's retry logic still succeeded fine during
  the degraded period** (a trivial text-only completion in ~9s, a real
  6-page vision request with the production prompt and
  `max_tokens=4096` in ~41s) -- so the session wasn't fully dead, just
  unreliable at the concurrency/request pattern the real batch used.
  No client-side fix was found; the practical takeaway is to budget for
  a full session respawn via the dashboard if a long batch's success
  rate craters partway through, rather than trusting a concurrency
  knob or more retries to recover it.
- **`numind/NuExtract3`'s decline, by contrast, tracked genuinely hard
  books.** Three passes (concurrency 4, 2, then serial 1) gave
  monotonically *dropping* success as the pool shrank to its hardest
  residual: 76.5% (189/247), 51.7% (30/58), 10.7% (3/28) -- consistent
  with a fixed subset of books the model can't reliably read regardless
  of load, not a session-health signal. The 25 books that still failed
  after all three passes (mostly the same 450s timeout, some
  `InternalServerError`) were recorded via `cli/skip_list.py` rather
  than chased further -- see AGENTS.md step 6 and this repo's existing
  10 pre-2026-08-24 `NuExtract3` skip-list entries for the same pattern
  on other books.

## Spawn-form fields that matter

- **Machine**: `Viper` for the AMD MI300A path (this is what all the
  notes below assume).
- **Framework**: `vLLM` (this doc doesn't cover the Ollama path).
- **Framework image reference**: a DockerHub `image_name:tag`, editable
  -- not locked to whatever's pre-filled. Default at time of writing:
  `rocm/vllm:rocm7.0.0_vllm_0.11.2_20251210` (vLLM **0.11.2**). See
  "Picking a different image" below before changing it.
- **Framework CLI arguments**: a single free-text field appended to the
  container's `vllm serve` invocation. **Every quote character
  (single or double) appears to get stripped before the value reaches
  the shell** -- confirmed by two independent failed attempts passing
  `--hf-overrides` a JSON value both single-quoted and
  backslash-escaped-double-quoted; both arrived at vLLM with every `"`
  and `'` gone (`--hf-overrides {architectures: [X]}`, not valid JSON).
  **Any CLI flag whose value must contain a quoted string (JSON,
  spaces-requiring paths, etc.) cannot be passed through this field on
  this launcher.** This directly blocks `deepseek-ai/deepseek-vl2` (see
  below) and would block anything else needing `--hf-overrides` or
  similar.
- **GPUs**: 2 worked fine for every 30-40B-class model tried here
  (`--tensor-parallel-size=2` must match).
- **Time**: session lifetime; 1-3h was enough for every model here once
  weights were already cached, but budget up to the full 8h max for a
  cold, large-checkpoint first spawn.

## Confirmed-working models (this session, `--tensor-parallel-size=2`, 2 GPUs)

- `Qwen/Qwen2.5-VL-7B-Instruct` -- small smoke-test pairing.
- `OpenGVLab/InternVL2_5-8B` -- small smoke-test pairing.
- `Qwen/Qwen3-Omni-30B-A3B-Instruct` -- production-quality pairing.
- `OpenGVLab/InternVL2_5-38B` -- production-quality pairing.

All four launched with just
`--model=<repo-id> --tensor-parallel-size=2 --trust-remote-code` in the
CLI-arguments field and served real `/v1/models`/chat-completions
traffic against the default `rocm7.0.0_vllm_0.11.2_*` image.

**Quality note, not a launch-mechanics finding but worth keeping next
to these two names:** across ~40 real title-level disagreements
resolved by hand during ground-truth arbitration (`docs/history.md`'s
2026-08-19 entry), `Qwen3-Omni-30B-A3B-Instruct` was right in nearly
every case; `InternVL2_5-38B` had frequent word-level misreadings and at
least two outright hallucinated strings. Worth factoring into future
model-pairing choices on this service, not just whether a model boots.

## Models that failed to launch on the default `vllm_0.11.2` image, and why

- **`Qwen/Qwen3.6-35B-A3B`**: crashes with
  `pydantic_core._pydantic_core.ValidationError: ... model type
  'qwen3_5_moe' but Transformers does not recognize this architecture`.
  `--trust-remote-code` does **not** help -- vLLM's own log says so
  explicitly: `The argument 'trust_remote_code' is to be used with Auto
  classes. It has no effect here and is ignored.` Architecture
  validation against vLLM's internal registry happens before any
  remote-code loading path is even reached. This is a genuine
  version-gap (this specific vLLM/transformers build predates the
  architecture), not a flag/config problem -- a newer image (see below)
  is the only real fix.
- **`deepseek-ai/deepseek-vl2`**: fails with `pydantic_core...
  ValidationError: ... No model architectures are specified`. Root
  cause: the model's own `config.json` (upstream, still true as of this
  session) has no top-level `architectures` field, and vLLM has never
  auto-inferred it for this model -- it needs
  `--hf-overrides '{"architectures": ["DeepseekVLV2ForCausalLM"]}'`
  passed explicitly, on any vLLM version. Given the CLI-arguments
  field's quote-stripping above, **this specific model cannot currently
  be launched through this dashboard at all**, regardless of image.
  Confirmed via GitHub issue history that this is a long-standing,
  still-open upstream gap, not something a newer vLLM release patches
  away.
- **`zai-org/GLM-4.1V-9B-Thinking`**: not attempted directly, but ruled
  out ahead of time -- vLLM's model registry maps it to
  `Glm4vForConditionalGeneration`, which requires **vLLM ≥0.12.0**; the
  default image ships 0.11.2. Would need the newer image below.
  (`zai-org/GLM-4.5V`, by contrast, maps to
  `Glm4vMoeForConditionalGeneration`, which only needs vLLM ≥0.10.2 --
  already satisfied by the default 0.11.2 image, no image change
  needed. Not yet actually launched/verified this session -- next thing
  to try.)

## Picking a different "Framework image reference"

The pre-filled image is not the only option -- any `rocm/vllm` tag from
DockerHub works, but AMD's tagging convention changed partway through
2026: everything after the `rocm7.0.0_vllm_0.11.2_*` generation is
**GPU-architecture-specific**, not a generic ROCm7 build. Picking the
wrong architecture suffix silently gets you an image that won't run on
this hardware at all.

**MI300A is `gfx942`** (confirmed via AMD's own docs and `rocminfo`
output references -- MI300A and MI300X share the same `gfx942` LLVM
target), which falls under the `gfx94X` grouping in these tags. Do
**not** pick a `gfx950-dcgpu` tag -- that's AMD's next-generation chip,
a different, incompatible architecture.

Listed via `curl -s "https://hub.docker.com/v2/repositories/rocm/vllm/tags?page_size=100"`
(public DockerHub API, no auth needed) -- as of this session, the
newest tag actually matching MI300A's `gfx94X` family:

```
rocm/vllm:rocm7.13.0_gfx94X-dcgpu_ubuntu24.04_py3.13_pytorch_2.10.0_vllm_0.19.1
```

vLLM 0.19.1 clears both the `Qwen3.6-35B-A3B` (`qwen3_5_moe`) and
`GLM-4.1V-9B-Thinking` (needs ≥0.12.0) version gaps above -- **not yet
actually spawned/verified this session**, so treat as "should work
based on version numbers," not confirmed. Re-check DockerHub for a
newer `gfx94X-dcgpu` tag before using this, since AMD publishes new
ones regularly (this list already had an even newer `rocm7.14.0`
generation without a `gfx94X` variant published yet, and a
`rocm7.13.0_gfx94X-dcgpu` one that did exist at the time of writing --
tag availability for a given GPU family lags the newest overall
release).

## `.endpoints` fields controlling extraction API style

Two optional fields, read by `inference.py`'s endpoint-file parser
(both the JSON-array and plain-text pasted-session-table formats):

- `extraction_api` -- `"nuextract"` routes this endpoint through
  `nuextract.py`'s template-mode extraction (an explicit JSON schema
  plus optional prose instructions, sent via
  `extra_body={"chat_template_kwargs": {...}}`) instead of the ordinary
  free-text-prompt path every other model uses. Empty/absent (the
  default) means the ordinary path, unchanged.
- `extraction_instructions` -- `"false"`/`"0"`/`"no"` omits the
  `instructions` field from the template-mode request (needed for a
  finetuned NuExtract2-family checkpoint, whose own chat template
  doesn't accept a separate instructions parameter the way NuExtract3's
  does). Defaults to `true`; only meaningful when
  `extraction_api == "nuextract"`.

**`numind/NuExtract3` gets `extraction_api: "nuextract"` and
`extraction_instructions: true` by convenience default when neither
field is set explicitly** -- a one-off carve-out for that exact model id
only, so the endpoint already documented above works without editing
`.endpoints`. Any other NuExtract-family checkpoint (e.g. a finetuned
NuExtract2) must declare both fields explicitly, since a finetuned
checkpoint can be renamed to anything.

A free-text prompt sent as ordinary chat content does NOT work against
`numind/NuExtract3` -- confirmed empirically: the model either ignores
it (falling back to its own trained document-to-markdown default) or
returns a degenerate single-entry response. Template mode with an
explicit JSON schema is the only reliable path. See design spec
`docs/superpowers/specs/2026-08-22-nuextract-template-mode-integration-design.md`
for the full empirical comparison (vision vs. text input, with vs.
without the `instructions` field).
