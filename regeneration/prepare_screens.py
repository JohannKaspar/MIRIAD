"""Build the blinded rating screens from the clean rated-subset run.

Reads the sample (passage text), arm A (stored GPT-3.5 output), arm B for the
rerun candidates, and the candidate lists. Writes two files:

  work/screens_blinded.json   what the app shows. Sets are called "Set 1" and
                              "Set 2"; no arm identity, no passage id, no kind.
  work/screens_key.json       the mapping back: passage id, which arm sits in
                              which position, which screens are planted and
                              how. Analysis only. Never opened while rating.

Selection follows the pre-registration: first 90 comparison-stratum passages
(A kept three) and first 30 recovery-stratum passages (A kept none) in draw
order, each requiring B to have at least one surviving pair; passages skipped
for an empty B are counted. Twelve planted screens are built from the spare
candidates beyond those: ten two-set screens from comparison spares and two
one-set screens from recovery spares, half with a trigger phrase inserted into
one answer, half with a contradiction written into one answer. Contradictions
are authored by hand into `work/planted_contradictions.json` before this runs,
so the same party writes and verifies them and the evaluator never sees them
in advance.

Blinding uses seed 4711, separate from the sampling seed: set order per screen,
then the order of screens, so planted and recovery screens are interleaved.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

from .metrics import load_arm_a, load_arm_b

W = Path("regeneration/work")
BLIND_SEED = 4711
N_COMPARISON, N_RECOVERY = 90, 30
N_PLANT_TWO_SET, N_PLANT_ONE_SET = 10, 2

_WS = re.compile(r"\s+")


def clean(text: str) -> str:
    """Normalise formatting so it cannot identify an arm.

    Arm A's questions carry a trailing newline and arm B's answers have
    different spacing habits; both are collapsed to single-spaced text.
    """
    return _WS.sub(" ", text).strip()


def insert_trigger(answer: str) -> str:
    """Rewrite so the released keyword filter would drop it."""
    head = answer[0].lower() + answer[1:] if answer else answer
    return "According to the study, " + head


def build(planted_path: Path = W / "planted_contradictions.json") -> dict:
    sample = json.load(open(W / "sample.json"))
    passages = {p["passage_id"]: p["text"] for p in sample["passages"]}
    A = load_arm_a(W / "original_arm.jsonl")
    B = load_arm_b(W / "pairs.jsonl")
    cand = json.load(open(W / "rerun_candidates.json"))

    def pairs_a(pid):
        return [{"question": clean(p["question"]), "answer": clean(p["answer"])}
                for p in A[pid]["pairs"] if p["survives_filter"]]

    def pairs_b(pid):
        return [{"question": clean(p["question"]), "answer": clean(p["answer"])}
                for p in B[pid]["pairs"]]

    skipped = {"comparison_b_empty": 0, "recovery_b_empty": 0}
    comparison, comp_spare = [], []
    for pid in cand["comparison_candidates"]:
        if pid not in B or B[pid]["kept"] == 0:
            skipped["comparison_b_empty"] += 1
            continue
        (comparison if len(comparison) < N_COMPARISON else comp_spare).append(pid)
    recovery, rec_spare = [], []
    for pid in cand["recovery_candidates"]:
        if pid not in B or B[pid]["kept"] == 0:
            skipped["recovery_b_empty"] += 1
            continue
        (recovery if len(recovery) < N_RECOVERY else rec_spare).append(pid)
    if len(comparison) < N_COMPARISON or len(recovery) < N_RECOVERY:
        raise RuntimeError(f"not enough candidates: {len(comparison)} comparison, {len(recovery)} recovery")
    if len(comp_spare) < N_PLANT_TWO_SET or len(rec_spare) < N_PLANT_ONE_SET:
        raise RuntimeError(f"not enough spares for planted screens: {len(comp_spare)} / {len(rec_spare)}")

    contradictions = json.load(open(planted_path)) if planted_path.exists() else {}
    plan = plan_planted(comp_spare, rec_spare)
    rng = random.Random(BLIND_SEED)
    screens, key = [], {}

    def add(kind, pid, sets_with_arms, planted=None):
        # sets_with_arms: list of (arm_label, pairs); order randomised here
        sets = list(sets_with_arms)
        if len(sets) == 2:
            rng.shuffle(sets)
        sid = f"tmp{len(screens):04d}"
        screens.append({"screen_id": sid, "format": "two_sets" if len(sets) == 2 else "one_set",
                        "passage": passages[pid], "sets": [{"pairs": s[1]} for s in sets]})
        key[sid] = {"passage_id": pid, "kind": kind, "arms_in_order": [s[0] for s in sets],
                    "planted": planted}

    for pid in comparison:
        add("comparison", pid, [("A", pairs_a(pid)), ("B", pairs_b(pid))])
    for pid in recovery:
        add("recovery", pid, [("B", pairs_b(pid))])

    for pl in plan:
        pid, fmt, defect, side_arm, j = pl["pid"], pl["format"], pl["defect"], pl["arm"], pl["pair_index"]
        target = [dict(p) for p in (pairs_a(pid) if side_arm == "A" else pairs_b(pid))]
        j = min(j, len(target) - 1)
        original = target[j]["answer"]
        if defect == "trigger":
            target[j]["answer"] = insert_trigger(original)
        else:
            edited = contradictions.get(pid, {}).get(str(j))
            if not edited:
                raise RuntimeError(f"no hand-written contradiction for {pid} pair {j}; "
                                   f"run targets() and author it in {planted_path} first")
            target[j]["answer"] = clean(edited)
        planted = {"defect": defect, "arm": side_arm, "pair_index": j, "original_answer": original}
        if fmt == "two":
            other = ("B", pairs_b(pid)) if side_arm == "A" else ("A", pairs_a(pid))
            add(f"planted_{defect}", pid, [(side_arm, target), other], planted)
        else:
            add(f"planted_{defect}", pid, [(side_arm, target)], planted)

    # interleave everything, then assign final ids in shown order
    order = list(range(len(screens)))
    rng.shuffle(order)
    final_screens, final_key = [], {}
    for n, idx in enumerate(order, 1):
        sid = f"s{n:03d}"
        s = dict(screens[idx]); s["screen_id"] = sid
        final_screens.append(s)
        final_key[sid] = key[screens[idx]["screen_id"]]

    json.dump({"blind_seed": BLIND_SEED, "screens": final_screens},
              open(W / "screens_blinded.json", "w"), indent=1)
    json.dump({"blind_seed": BLIND_SEED, "skipped": skipped, "key": final_key},
              open(W / "screens_key.json", "w"), indent=1)
    kinds = {}
    for v in final_key.values():
        kinds[v["kind"]] = kinds.get(v["kind"], 0) + 1
    return {"screens": len(final_screens), "kinds": kinds, "skipped": skipped,
            "planted": [(pl["pid"], pl["format"], pl["defect"]) for pl in plan]}


def plan_planted(comp_spare: list[str], rec_spare: list[str]) -> list[dict]:
    """Decide the planted screens deterministically from the spare candidates.

    Its own RNG, so `targets()` can reproduce the choices without building the
    screens: ten two-set plants from comparison spares and two one-set plants
    from recovery spares, alternating trigger and contradiction, with the
    defective side and pair chosen at random. pair_index is chosen in 0..2 and
    clipped to the set length at build time.
    """
    rng = random.Random(BLIND_SEED + 1)
    plan = []
    picks = [(pid, "two") for pid in comp_spare[:N_PLANT_TWO_SET]] + \
            [(pid, "one") for pid in rec_spare[:N_PLANT_ONE_SET]]
    for i, (pid, fmt) in enumerate(picks):
        plan.append({"pid": pid, "format": fmt,
                     "defect": "trigger" if i % 2 == 0 else "contradiction",
                     "arm": rng.choice(["A", "B"]) if fmt == "two" else "B",
                     "pair_index": rng.randrange(3)})
    return plan


def targets() -> list[dict]:
    """The answers that need a hand-written contradiction, with their passages.

    Run after arm B for the rerun candidates exists and before build(). Prints
    what to author into work/planted_contradictions.json as
    {passage_id: {pair_index: edited_answer}}.
    """
    sample = json.load(open(W / "sample.json"))
    passages = {p["passage_id"]: p["text"] for p in sample["passages"]}
    A = load_arm_a(W / "original_arm.jsonl")
    B = load_arm_b(W / "pairs.jsonl")
    cand = json.load(open(W / "rerun_candidates.json"))
    comp_ok = [pid for pid in cand["comparison_candidates"] if pid in B and B[pid]["kept"] > 0]
    rec_ok = [pid for pid in cand["recovery_candidates"] if pid in B and B[pid]["kept"] > 0]
    out = []
    for pl in plan_planted(comp_ok[N_COMPARISON:], rec_ok[N_RECOVERY:]):
        if pl["defect"] != "contradiction":
            continue
        pid, arm = pl["pid"], pl["arm"]
        pairs = ([p for p in A[pid]["pairs"] if p["survives_filter"]] if arm == "A" else B[pid]["pairs"])
        j = min(pl["pair_index"], len(pairs) - 1)
        out.append({"passage_id": pid, "arm": arm, "pair_index": j,
                    "question": clean(pairs[j]["question"]), "answer": clean(pairs[j]["answer"]),
                    "passage": passages[pid]})
    return out



if __name__ == "__main__":
    import sys as _sys
    if len(_sys.argv) > 1 and _sys.argv[1] == "targets":
        for t in targets():
            print(f"=== {t['passage_id']} arm {t['arm']} pair {t['pair_index']} ===")
            print("PASSAGE:", t["passage"][:900], "...")
            print("Q:", t["question"]); print("A:", t["answer"]); print()
    else:
        print(json.dumps(build(), indent=1))
