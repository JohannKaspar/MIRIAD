"""The generation prompt, read from the released pipeline rather than copied.

`data_generation/generate_dataset.py` holds the prompt as two module-level
triple-quoted strings. Importing that module is not an option: it builds an
OpenAI client at import time. So the source is parsed and the two string
literals are evaluated exactly as Python would, and the file's hash is pinned.
If upstream edits the prompt, this raises instead of silently changing what the
two arms are being compared on.

The first version read the raw characters between the triple quotes, which
left escape sequences unprocessed: `main_prompt` ends in a literal `\\n` in the
source, so a backslash and an `n` were sent where the original script sent a
line break. Evaluating the literal removes the class of error, not just that
instance.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

UPSTREAM = Path(__file__).resolve().parents[1] / "data_generation" / "generate_dataset.py"

# sha256 of data_generation/generate_dataset.py at commit f89f8d3.
PINNED_SHA256 = "63cee6b41f6d40a5f2e712931d22d08bd4a77732b4e2d5442727d089e14be467"


def _extract(source: str, name: str) -> str:
    """Return the value of the module-level string assignment `name`."""
    for node in ast.parse(source).body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and getattr(node.targets[0], "id", None) == name
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)):
            return node.value.value
    raise RuntimeError(f"{name} not found as a string assignment in {UPSTREAM}")


def load(check_hash: bool = True) -> tuple[str, str]:
    """Return (main_prompt, negative_examples) exactly as the original script saw them."""
    raw = UPSTREAM.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if check_hash and digest != PINNED_SHA256:
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
