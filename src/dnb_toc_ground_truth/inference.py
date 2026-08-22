"""Generic OpenAI-compatible inference-endpoint resolution --
forked from chapter-segmentation's evaluation/inference_endpoints.py with
all provider-specific auto-discovery removed. Every model must be named
explicitly via --use-vision/--use-text and resolved against an
--endpoints-file; see design spec
docs/superpowers/specs/2026-08-21-dnb-toc-ground-truth-extraction-design.md
"Endpoint and config system"."""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from openai import AsyncOpenAI

DEFAULT_TIMEOUT = 90.0
DEFAULT_ENDPOINTS_FILENAME = ".endpoints"
DEFAULT_CONFIG_FILENAME = ".config.json"

_MODEL_ARG_RE = re.compile(r"--model[= ](\S+)")


@dataclass(frozen=True)
class ModelEndpoint:
    """One ready-to-call (client, model_id) pair, plus which extraction
    path it was requested for ("vision" or "text"). `label` is the
    resolved model id, used only for log/print output."""

    label: str
    model_id: str
    kind: str
    client: AsyncOpenAI


class OpenAICompatibleLLMClient:
    """Minimal LLMClient wrapping an already-built AsyncOpenAI client --
    callers construct the client themselves (via resolve_model_endpoints
    below), so this class has no provider-specific knowledge at all."""

    def __init__(self, model: str, client: AsyncOpenAI):
        self._client = client
        self._model = model

    async def generate(
        self, prompt: str, *, max_tokens: int, temperature: float,
        is_valid: Optional[Callable[[str], bool]] = None,
    ) -> str:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""


@dataclass(frozen=True)
class _EndpointEntry:
    """One parsed row from an --endpoints-file, before resolution against
    a requested model id. `status` is the JSON format's raw status string
    ("Running", "Stopped", ...) used only to break a multi-match tie --
    always "" for the plain-text format, which has no equivalent field.
    `extraction_api` ("" or "nuextract") and `extraction_instructions`
    select and configure the NuExtract-family template-mode extraction
    path in nuextract.py -- see _resolve_extraction_fields below and
    design spec docs/superpowers/specs/2026-08-22-nuextract-template-
    mode-integration-design.md."""

    base_url: str
    api_key: str
    model: str
    status: str = ""
    extraction_api: str = ""
    extraction_instructions: bool = True


def _normalize_base_url(url: str) -> str:
    return url if url.rstrip("/").endswith("/v1") else url.rstrip("/") + "/v1"


_NUEXTRACT_CONVENIENCE_MODEL = "numind/NuExtract3"


def _resolve_extraction_fields(fields: dict, model: str) -> tuple[str, bool]:
    """Resolves (extraction_api, extraction_instructions) for one endpoint
    entry from its raw parsed fields dict -- works identically for the
    JSON-row dict and the plain-text session-block dict, both plain
    str-keyed dicts by the time this is called. An explicitly-PRESENT
    "extraction_api" key always wins, even an explicit "" override --
    only an ABSENT key falls back to the numind/NuExtract3 convenience
    default (the endpoint already running before this field existed
    keeps working without editing .endpoints). Every other model with an
    absent key defaults to "" (today's free-text-prompt path, unchanged).
    "extraction_instructions" defaults to True unless explicitly set to
    "false"/"0"/"no" (case-insensitive) -- only meaningful when
    extraction_api == "nuextract". See design spec 2026-08-22-nuextract-
    template-mode-integration-design.md."""
    if "extraction_api" in fields:
        extraction_api = str(fields["extraction_api"]).strip()
    elif model == _NUEXTRACT_CONVENIENCE_MODEL:
        extraction_api = "nuextract"
    else:
        extraction_api = ""
    extraction_instructions = str(fields.get("extraction_instructions", "")).strip().lower() not in ("false", "0", "no")
    return extraction_api, extraction_instructions


def _parse_session_block(block: str) -> dict[str, str]:
    """Parses one pasted dashboard session table (tab-separated
    `field<TAB>value` lines, exactly as copied from a provider's UI) into
    a dict."""
    fields: dict[str, str] = {}
    for line in block.splitlines():
        if "\t" not in line:
            continue
        key, _, value = line.partition("\t")
        key = key.strip()
        if key:
            fields[key] = value.strip()
    return fields


def _model_from_fields(fields: dict[str, str]) -> str:
    model = fields.get("model", "").strip()
    if model:
        return model
    match = _MODEL_ARG_RE.search(fields.get("framework_args", ""))
    return match.group(1) if match else ""


def _parse_plain_text_endpoints(text: str) -> list[_EndpointEntry]:
    """Legacy pasted-session-table format (backward-compatible
    alternative to the JSON array format): one or more blocks separated
    by a blank line, each with `url`/`key`/`framework_args` (or `model`)
    fields. A block missing url/key/model is skipped."""
    entries = []
    for block in (b for b in re.split(r"\n\s*\n", text.strip()) if b.strip()):
        fields = _parse_session_block(block)
        url = fields.get("url", "").strip()
        api_key = fields.get("key", "").strip()
        model = _model_from_fields(fields)
        if not (url and api_key and model):
            continue
        extraction_api, extraction_instructions = _resolve_extraction_fields(fields, model)
        entries.append(_EndpointEntry(
            base_url=_normalize_base_url(url), api_key=api_key, model=model,
            extraction_api=extraction_api, extraction_instructions=extraction_instructions,
        ))
    return entries


def _parse_json_endpoints(data: list[dict]) -> list[_EndpointEntry]:
    """Officially-supported endpoints-file format: a JSON array of
    objects as pasted from a provider dashboard. Consumes only `url`,
    `key`, and the model id (from `model` if present, else parsed out of
    `framework_args`'s `--model=...` token), plus `status` for
    tie-breaking -- every other field (framework, gpus, job_id, ...) is
    ignored except `extraction_api`/`extraction_instructions`, see
    _resolve_extraction_fields. An entry missing url/key/model is
    skipped."""
    entries = []
    for row in data:
        url = str(row.get("url", "")).strip()
        api_key = str(row.get("key", "")).strip()
        model = str(row.get("model", "")).strip()
        if not model:
            match = _MODEL_ARG_RE.search(str(row.get("framework_args", "")))
            model = match.group(1) if match else ""
        if not (url and api_key and model):
            continue
        extraction_api, extraction_instructions = _resolve_extraction_fields(row, model)
        entries.append(_EndpointEntry(
            base_url=_normalize_base_url(url), api_key=api_key, model=model,
            status=str(row.get("status", "")),
            extraction_api=extraction_api, extraction_instructions=extraction_instructions,
        ))
    return entries


def load_endpoint_entries(path: Path) -> list[_EndpointEntry]:
    """Parses an --endpoints-file, auto-detecting the JSON-array format
    vs. the plain-text pasted-session-table format by trying JSON first.
    Raises ValueError naming the path if it doesn't exist or contains no
    usable entry -- meant to be diagnosed directly by whoever set up the
    file, not a bare empty-list surprise downstream."""
    if not path.exists():
        raise ValueError(f"--endpoints-file {path} does not exist")
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = None
    entries = _parse_json_endpoints(data) if isinstance(data, list) else _parse_plain_text_endpoints(text)
    if not entries:
        raise ValueError(f"--endpoints-file {path} has no usable endpoint entries")
    return entries


def resolve_model_endpoints(
    model_ids: list[str], kind: str, entries: list[_EndpointEntry], *, timeout: float = DEFAULT_TIMEOUT,
) -> list[ModelEndpoint]:
    """Resolves each of `model_ids` (in order; duplicates allowed -- the
    same model id may be requested twice for two independent reads
    against endpoints that happen to share a model id) against `entries`
    by exact model-id match. More than one entry matching the same id is
    resolved by preferring the one whose `status` is "Running"; if that
    still leaves more than one candidate (or none of the matches report
    status at all -- the plain-text format never does), raises ValueError
    naming the ambiguous id so the caller can fix the endpoints file.
    Raises ValueError naming the id if no entry matches at all."""
    resolved = []
    for model_id in model_ids:
        matches = [e for e in entries if e.model == model_id]
        if not matches:
            raise ValueError(f"no endpoint found for model {model_id!r} in the endpoints file")
        if len(matches) > 1:
            running = [e for e in matches if e.status == "Running"]
            if len(running) == 1:
                matches = running
            else:
                raise ValueError(
                    f"model {model_id!r} matches {len(matches)} endpoint entries and exactly one \"Running\" "
                    f"entry could not be identified to disambiguate -- fix the endpoints file"
                )
        entry = matches[0]
        client = AsyncOpenAI(base_url=entry.base_url, api_key=entry.api_key, timeout=timeout)
        resolved.append(ModelEndpoint(label=entry.model, model_id=entry.model, kind=kind, client=client))
    return resolved


def load_config(path: Path) -> dict:
    """Parses a --config-file (JSON) into a dict of CLI-flag defaults.
    Returns {} if the file doesn't exist -- config is optional, every
    value has its own script-level default."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
