# MPCDF LLM Inference Service -- empirical notes

Web UI at `https://llm.mpcdf.mpg.de`. Distinct from the Raven/Viper-GPU
SLURM batch-job HPC access documented in `evaluation/hpc/README.md` --
this is a separate, self-service web dashboard that spins up a
per-user, per-session vLLM (or Ollama) server on Viper-GPU (or DAIS) as
a SLURM job under the hood, for a user-chosen duration (up to 8h),
exposing an OpenAI-compatible chat-completions API. Written up here
because none of this is documented anywhere obvious and every item
below was learned the hard way, live, against real sessions
(2026-08-18/19) integrating it as a second inference endpoint alongside
KISSKI for `evaluation/refresh_llm_cache.py`'s `--endpoint`/
`--config-file` flags (see `evaluation/inference_endpoints.py`'s module
docstring for the design rationale).

## Endpoint URL convention

Each spawned session gets its own base URL of the form
`https://llm.mpcdf.mpg.de/<session-hash>/v1` and its own API key --
**the `/v1` suffix is required and not obvious from the dashboard UI**;
confirmed by curling the bare session-hash URL first (404) then
`/v1/models` (real JSON). Set `<ALIAS>_BASE_URL` to the `/v1`-suffixed
form when populating this project's `--endpoint` env vars.

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
