"""Recovering QA pairs from raw model output, and the released keyword filter.

Two parsers are kept on purpose. `parse_released` is the regex the original
pipeline used; `parse_tolerant` also accepts the shapes a current model returns
for the same prompt (markdown emphasis, headings, Q:/A: abbreviations). Both are
run over both arms so the effect of the parser is a reported number rather than
a silent confound: the released regex returns nothing at all for output in
another shape, which would otherwise read as the model producing no pairs.
"""

from __future__ import annotations

import re

RELEASED_PATTERN = re.compile(r"Question \d+: (.*?)\nAnswer \d+: (.*?)(?=\n\n|$)", re.S)

# Same anchors, but tolerant of leading markers (#, *, -, digits), of markdown
# emphasis around the label, of "Q"/"A" abbreviations, and of a single newline
# instead of a blank line between pairs.
_LABEL = r"[ \t]*[#*\-\d.]*[ \t]*[*_]{0,2}(?:Question|Q)[ \t]*\d*[*_]{0,2}[ \t]*[:.\)][ \t]*"
_ANSWER = r"[ \t]*[#*\-\d.]*[ \t]*[*_]{0,2}(?:Answer|A)[ \t]*\d*[*_]{0,2}[ \t]*[:.\)][ \t]*"
TOLERANT_PATTERN = re.compile(
    rf"^{_LABEL}(?P<question>.+?)\s*^{_ANSWER}(?P<answer>.+?)(?=^{_LABEL}|\Z)",
    re.S | re.M | re.I,
)

# quality_control/keyword_filter.py, verbatim: two phrases, matched against the
# answer only. The question is never tested.
PASSAGE_REFERENCE_PHRASES = [r"the passage", r"the study"]
PASSAGE_REFERENCE_PATTERN = re.compile("|".join(PASSAGE_REFERENCE_PHRASES), re.IGNORECASE)


def _clean(text: str) -> str:
    return re.sub(r"[*_`]+$", "", text.strip()).strip()


def parse_released(response: str) -> list[tuple[str, str]]:
    """The original parser, unchanged."""
    return RELEASED_PATTERN.findall(response)


def parse_tolerant(response: str) -> list[tuple[str, str]]:
    """Accepts the released shape plus the common markdown variants."""
    pairs = []
    for match in TOLERANT_PATTERN.finditer(response):
        question = _clean(match.group("question"))
        answer = _clean(match.group("answer"))
        if question and answer:
            pairs.append((question, answer))
    return pairs


def survives_keyword_filter(answer: str) -> bool:
    """True when the released filter would keep this pair."""
    return PASSAGE_REFERENCE_PATTERN.search(answer) is None


def apply_keyword_filter(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    return [(q, a) for q, a in pairs if survives_keyword_filter(a)]
