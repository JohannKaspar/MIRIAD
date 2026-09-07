"""Countable measurements over both arms, and the stratified screen selection.

Arm A comes from `original_arm.jsonl` (stored GPT-3.5 output for the sampled
passages), arm B from `pairs.jsonl` written by `run.py parse`. Everything here
is descriptive and needs no evaluator. `stratify` picks the screens for the
human evaluation in draw order within each stratum, as pre-registered.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from itertools import combinations
from pathlib import Path

from .parse import PASSAGE_REFERENCE_PATTERN

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def answer_in_passage(answer: str, passage: str, n: int = 4) -> float:
    """Share of the answer's word 4-grams that occur verbatim in the passage.

    A crude, transparent proxy for grounding: 1.0 means every 4-gram of the
    answer is lifted from the passage, 0.0 means none is. Paraphrase scores low
    without being wrong, so this is reported, not judged.
    """
    grams = _ngrams(_tokens(answer), n)
    if not grams:
        return 0.0
    return len(grams & _ngrams(_tokens(passage), n)) / len(grams)


_STOP = set("the a an of and or in on to for with by is are was were be been that this these "
            "those as at from it its their which who whom into than then also may can not no such "
            "more most other some any each both between over under during after before within "
            "without about".split())


def content_word_overlap(answer: str, passage: str) -> float:
    """Share of the answer's content words (stopwords and short tokens removed)
    that occur anywhere in the passage. Tolerant of paraphrase, so low values
    point to content the passage does not contain rather than to rewording.
    """
    words = {w for w in _tokens(answer) if w not in _STOP and len(w) > 2}
    if not words:
        return 0.0
    return len(words & {w for w in _tokens(passage) if w not in _STOP and len(w) > 2}) / len(words)


def near_duplicate(a: str, b: str, threshold: float = 0.7) -> bool:
    ta, tb = set(_tokens(a)), set(_tokens(b))
    return bool(ta and tb) and len(ta & tb) / len(ta | tb) >= threshold


def load_arm_a(path: Path) -> dict[str, dict]:
    arm = {}
    for line in path.read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            arm[rec["passage_id"]] = rec
    return arm


def load_arm_b(path: Path) -> dict[str, dict]:
    arm = {}
    for line in path.read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            rec["kept"] = len(rec["pairs"])
            arm[rec["passage_id"]] = rec
    return arm


def _arm_stats(records: list[dict], passages: dict[str, str]) -> dict:
    raw = kept = trigger = 0
    kept_hist = Counter()
    ain, cwo, qlen, alen = [], [], [], []
    dup_pairs = dup_total = 0
    for rec in records:
        raw_pairs = rec["raw_pairs"] if "raw_pairs" in rec else rec["pairs"]
        raw += len(raw_pairs)
        kept_here = 0
        for pr in raw_pairs:
            survives = pr.get("survives_filter",
                              PASSAGE_REFERENCE_PATTERN.search(pr["answer"]) is None)
            trigger += 0 if survives else 1
            kept_here += 1 if survives else 0
            if survives:
                ain.append(answer_in_passage(pr["answer"], passages[rec["passage_id"]]))
                cwo.append(content_word_overlap(pr["answer"], passages[rec["passage_id"]]))
                qlen.append(len(_tokens(pr["question"])))
                alen.append(len(_tokens(pr["answer"])))
        kept += kept_here
        kept_hist[min(kept_here, 3)] += 1
        surviving = [pr for pr in raw_pairs
                     if pr.get("survives_filter", PASSAGE_REFERENCE_PATTERN.search(pr["answer"]) is None)]
        for x, y in combinations(surviving, 2):
            dup_total += 1
            dup_pairs += near_duplicate(x["question"] + " " + x["answer"],
                                        y["question"] + " " + y["answer"])
    n = len(records)

    def med(xs):
        xs = sorted(xs)
        return xs[len(xs) // 2] if xs else None

    return {
        "passages": n,
        "raw_pairs": raw,
        "raw_pairs_per_passage": round(raw / n, 3) if n else None,
        "trigger_phrase_rate": round(trigger / raw, 4) if raw else None,
        "filter_survival": round(kept / raw, 4) if raw else None,
        "kept_pairs": kept,
        "kept_per_passage_hist": {str(k): kept_hist[k] for k in range(4)},
        "kept_per_passage_share": {str(k): round(kept_hist[k] / n, 4) for k in range(4)} if n else {},
        "coverage": round(sum(v for k, v in kept_hist.items() if k > 0) / n, 4) if n else None,
        "answer_in_passage_median": round(med(ain), 3) if ain else None,
        "content_word_overlap_median": round(med(cwo), 3) if cwo else None,
        "answers_under_half_content_in_passage": round(sum(x < 0.5 for x in cwo) / len(cwo), 4) if cwo else None,
        "question_tokens_median": med(qlen),
        "answer_tokens_median": med(alen),
        "near_duplicate_rate": round(dup_pairs / dup_total, 4) if dup_total else None,
    }


def compute(arm_a: dict, arm_b: dict, passages: dict[str, str], order: list[str]) -> dict:
    ids = [i for i in order if i in arm_a and i in arm_b]
    a = _arm_stats([arm_a[i] for i in ids], passages)
    b = _arm_stats([arm_b[i] for i in ids], passages)
    recovered = sum(1 for i in ids if arm_a[i]["kept"] == 0 and arm_b[i]["kept"] > 0)
    lost = sum(1 for i in ids if arm_a[i]["kept"] > 0 and arm_b[i]["kept"] == 0)
    return {
        "paired_passages": len(ids),
        "arm_a_original": a,
        "arm_b_regenerated": b,
        "coverage_recovered_by_b": recovered,
        "coverage_lost_by_b": lost,
        "coverage_recovered_share": round(recovered / len(ids), 4) if ids else None,
    }


def stratify(arm_a: dict, arm_b: dict, order: list[str],
             n_comparison: int = 90, n_recovery: int = 30) -> dict:
    """Screens in draw order within each stratum, as pre-registered.

    Comparison: A kept all three. Recovery: A kept none. In both, B must have
    at least one surviving pair, otherwise there is nothing to rate; passages
    skipped for that reason are counted and reported.
    """
    comparison, recovery, skipped = [], [], Counter()
    for pid in order:
        if pid not in arm_a or pid not in arm_b:
            continue
        ka, kb = arm_a[pid]["kept"], arm_b[pid]["kept"]
        if ka == 3 and len(comparison) < n_comparison:
            if kb == 0:
                skipped["comparison_b_empty"] += 1
            else:
                comparison.append(pid)
        elif ka == 0 and len(recovery) < n_recovery:
            if kb == 0:
                skipped["recovery_b_empty"] += 1
            else:
                recovery.append(pid)
        if len(comparison) >= n_comparison and len(recovery) >= n_recovery:
            break
    return {"comparison": comparison, "recovery": recovery, "skipped": dict(skipped)}


def print_report(m: dict) -> None:
    a, b = m["arm_a_original"], m["arm_b_regenerated"]
    rows = [
        ("passages", "passages"), ("raw pairs", "raw_pairs"),
        ("raw pairs / passage", "raw_pairs_per_passage"),
        ("trigger-phrase rate", "trigger_phrase_rate"),
        ("filter survival", "filter_survival"),
        ("coverage (>=1 kept)", "coverage"),
        ("answer 4-grams in passage, median", "answer_in_passage_median"),
        ("content words in passage, median", "content_word_overlap_median"),
        ("answers <50% content in passage", "answers_under_half_content_in_passage"),
        ("question tokens, median", "question_tokens_median"),
        ("answer tokens, median", "answer_tokens_median"),
        ("near-duplicate rate", "near_duplicate_rate"),
    ]
    print(f"{'':28}{'GPT-3.5 (A)':>14}{'GPT-5.6 (B)':>14}")
    for label, key in rows:
        print(f"{label:28}{str(a[key]):>14}{str(b[key]):>14}")
    print(f"{'kept 0/1/2/3 per passage':28}"
          f"{'/'.join(str(a['kept_per_passage_hist'][k]) for k in '0123'):>14}"
          f"{'/'.join(str(b['kept_per_passage_hist'][k]) for k in '0123'):>14}")
    print(f"\ncoverage recovered by B: {m['coverage_recovered_by_b']} passages "
          f"({m['coverage_recovered_share']:.1%}); lost by B: {m['coverage_lost_by_b']}")
