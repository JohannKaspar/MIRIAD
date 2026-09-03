"""The generation prompt, read from the released pipeline rather than copied.

`data_generation/generate_dataset.py` holds the prompt as two module-level
triple-quoted strings. Importing that module is not an option: it builds an
OpenAI client at import time. So the source is read and the two strings are
extracted, and the file's hash is pinned. If upstream edits the prompt, this
raises instead of silently changing what the two arms are being compared on.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

UPSTREAM = Path(__file__).resolve().parents[1] / "data_generation" / "generate_dataset.py"

# sha256 of data_generation/generate_dataset.py at commit f89f8d3.
PINNED_SHA256 = "63cee6b41f6d40a5f2e712931d22d08bd4a77732b4e2d5442727d089e14be467"

_TRIPLE = r"^{name}\s*=\s*'''(.*?)'''"


def _extract(source: str, name: str) -> str:
    match = re.search(_TRIPLE.format(name=name), source, re.S | re.M)
    if match is None:
        raise RuntimeError(f"{name} not found in {UPSTREAM}")
    return match.group(1)


def load(check_hash: bool = True) -> tuple[str, str]:
    """Return (main_prompt, negative_examples) exactly as released."""
    raw = UPSTREAM.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if check_hash and PINNED_SHA256 != "PENDING" and digest != PINNED_SHA256:
        raise RuntimeError(
            f"{UPSTREAM} changed upstream.\n"
            f"  pinned {PINNED_SHA256}\n  actual {digest}\n"
            "Review the diff, then update PINNED_SHA256 deliberately."
        )
    source = raw.decode("utf-8")
    return _extract(source, "main_prompt"), _extract(source, "negative_examples")


def build(passage: str) -> str:
    """The exact string the original pipeline sent for one passage."""
    main_prompt, negative_examples = load()
    return main_prompt + passage + negative_examples


def upstream_sha256() -> str:
    return hashlib.sha256(UPSTREAM.read_bytes()).hexdigest()
