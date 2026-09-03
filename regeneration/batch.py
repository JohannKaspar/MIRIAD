"""Batch API request building, submission and response extraction.

Three deviations from data_generation/generate_dataset.py, all deliberate and
all recorded in the run manifest so they can be stated in the write-up:

  transport         the original made one synchronous call per passage; this
                    submits one batch job. Same prompt, same one-passage-per-
                    request granularity, half the token price.

  max_output_tokens the original capped visible output at 1000 tokens. Reasoning
                    tokens are billed and counted as output, so the same cap
                    would truncate answers before the third pair. Raised, with
                    truncation reported per request rather than assumed absent.

  temperature       the original set temperature=0. Reasoning models do not take
                    it, so it is omitted. The arms therefore differ in sampling
                    as well as in model, which is a stated limitation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

from . import prompt as prompt_module
from .sample import Passage

ENDPOINT = "/v1/responses"


@dataclass
class GenerationConfig:
    model: str = "gpt-5.6-sol"
    reasoning_effort: str = "low"     # explicit: the API default is "medium"
    max_output_tokens: int = 4000
    completion_window: str = "24h"
    prompt_sha256: str = field(default_factory=prompt_module.upstream_sha256)


def build_request(passage: Passage, config: GenerationConfig) -> dict:
    return {
        "custom_id": passage.passage_id,
        "method": "POST",
        "url": ENDPOINT,
        "body": {
            "model": config.model,
            "input": prompt_module.build(passage.text),
            "reasoning": {"effort": config.reasoning_effort},
            "max_output_tokens": config.max_output_tokens,
        },
    }


def write_requests(passages: list[Passage], config: GenerationConfig, path: Path) -> Path:
    with path.open("w") as handle:
        for passage in passages:
            handle.write(json.dumps(build_request(passage, config)) + "\n")
    return path


def submit(requests_path: Path, config: GenerationConfig) -> dict:
    from openai import OpenAI

    client = OpenAI()
    uploaded = client.files.create(file=requests_path.open("rb"), purpose="batch")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint=ENDPOINT,
        completion_window=config.completion_window,
        metadata={"task": "T-215", "model": config.model, "effort": config.reasoning_effort},
    )
    return {"batch_id": batch.id, "input_file_id": uploaded.id, "status": batch.status,
            "config": asdict(config)}


def status(batch_id: str) -> dict:
    from openai import OpenAI

    batch = OpenAI().batches.retrieve(batch_id)
    counts = batch.request_counts
    return {
        "status": batch.status,
        "completed": getattr(counts, "completed", None),
        "failed": getattr(counts, "failed", None),
        "total": getattr(counts, "total", None),
        "output_file_id": batch.output_file_id,
        "error_file_id": batch.error_file_id,
    }


def fetch(batch_id: str, out_path: Path) -> Path:
    from openai import OpenAI

    client = OpenAI()
    batch = client.batches.retrieve(batch_id)
    if batch.output_file_id is None:
        raise RuntimeError(f"batch {batch_id} has no output yet (status {batch.status})")
    out_path.write_bytes(client.files.content(batch.output_file_id).read())
    return out_path


def extract_text(body: dict) -> str:
    """Pull the assistant text out of one Responses API body."""
    if isinstance(body.get("output_text"), str):
        return body["output_text"]
    chunks = []
    for item in body.get("output", []):
        if item.get("type") != "message":
            continue
        for part in item.get("content", []):
            if part.get("type") == "output_text":
                chunks.append(part.get("text", ""))
    return "".join(chunks)


def read_results(results_path: Path):
    """Yield (passage_id, text, meta) for each line of a batch output file."""
    with results_path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            passage_id = record.get("custom_id")
            response = record.get("response") or {}
            body = response.get("body") or {}
            usage = body.get("usage") or {}
            details = usage.get("output_tokens_details") or {}
            meta = {
                "status_code": response.get("status_code"),
                "response_status": body.get("status"),
                "incomplete_reason": (body.get("incomplete_details") or {}).get("reason"),
                "input_tokens": usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"),
                "reasoning_tokens": details.get("reasoning_tokens"),
                "error": record.get("error"),
            }
            yield passage_id, extract_text(body), meta
