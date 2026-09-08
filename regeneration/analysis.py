"""The pre-registered numbers, from ratings and judgements.

    python -m regeneration.analysis

Reads work/screens_key.json, work/ratings.jsonl, and if present
work/judge_screens.jsonl and work/judge_stratum.jsonl. Everything here is
fixed by PREREGISTRATION.md; nothing is chosen after looking at the data.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

W = Path("regeneration/work")
MARGIN = (0.35, 0.65)
CRITERIA = ("factual", "grounded", "relevant")
GATE_CATCH = 9   # of 12 planted screens


def wilson(k: int, n: int, z: float) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def positional_to_arm(choice: int | None, arms: list[str]) -> str | None:
    if choice is None:
        return None
    if choice == 0:
        return "tie"
    return arms[choice - 1]


def criterion_rates(ratings: list[dict], key: dict, kind: str) -> dict:
    """Share of pairs checked per criterion, per arm, on screens of `kind`."""
    checked, total = defaultdict(int), defaultdict(int)
    for r in ratings:
        k = key[r["screen_id"]]
        if k["kind"] != kind:
            continue
        for name, val in r["criteria"].items():
            si = int(name.split("_")[0][3:])          # set index, 1-based
            crit = name.rsplit("_", 1)[1]
            arm = k["arms_in_order"][si - 1]
            total[(arm, crit)] += 1
            checked[(arm, crit)] += 1 if val else 0
    return {f"{arm}_{crit}": {"checked": checked[(arm, crit)], "of": total[(arm, crit)],
                              "rate": round(checked[(arm, crit)] / total[(arm, crit)], 4)}
            for (arm, crit) in sorted(total)}


def planted_catch(ratings: list[dict], key: dict) -> dict:
    """A planted screen is caught when the defective pair is left unchecked on
    at least one criterion. Reported against the unchecked rate on non-planted
    pairs, since a rater who unchecks everything would 'catch' everything."""
    caught = seen = 0
    not_preferred = two_set = 0
    for r in ratings:
        k = key[r["screen_id"]]
        if not k["kind"].startswith("planted"):
            continue
        seen += 1
        p = k["planted"]
        si = k["arms_in_order"].index(p["arm"]) + 1
        pi = p["pair_index"] + 1
        flags = [r["criteria"].get(f"set{si}_pair{pi}_{c}", True) for c in CRITERIA]
        if not all(flags):
            caught += 1
        if r["format"] == "two_sets":
            two_set += 1
            arm_pref = positional_to_arm(r["forced_choice"], k["arms_in_order"])
            if arm_pref != p["arm"]:
                not_preferred += 1
    # base rate: share of non-planted pairs with any criterion unchecked
    any_unchecked = pairs = 0
    for r in ratings:
        if key[r["screen_id"]]["kind"].startswith("planted"):
            continue
        by_pair = defaultdict(list)
        for name, val in r["criteria"].items():
            by_pair[name.rsplit("_", 1)[0]].append(val)
        for vals in by_pair.values():
            pairs += 1
            any_unchecked += 0 if all(vals) else 1
    return {"caught": caught, "of": seen, "gate": GATE_CATCH,
            "gate_passed": seen >= 12 and caught >= GATE_CATCH,
            "defective_set_not_preferred": not_preferred, "of_two_set": two_set,
            "base_rate_any_unchecked_nonplanted": round(any_unchecked / pairs, 4) if pairs else None}


def forced_choice(ratings: list[dict], key: dict) -> dict:
    c = Counter()
    for r in ratings:
        k = key[r["screen_id"]]
        if k["kind"] != "comparison":
            continue
        c[positional_to_arm(r["forced_choice"], k["arms_in_order"])] += 1
    decided = c["A"] + c["B"]
    lo90, hi90 = wilson(c["B"], decided, 1.645)
    lo95, hi95 = wilson(c["B"], decided, 1.96)
    inside = decided > 0 and lo90 >= MARGIN[0] and hi90 <= MARGIN[1]
    return {"screens": sum(c.values()), "A_wins": c["A"], "B_wins": c["B"], "ties": c["tie"],
            "tie_rate": round(c["tie"] / max(sum(c.values()), 1), 4),
            "B_win_rate_decided": round(c["B"] / decided, 4) if decided else None,
            "ci95": [round(lo95, 4), round(hi95, 4)], "ci90": [round(lo90, 4), round(hi90, 4)],
            "margin": list(MARGIN), "equivalence_within_margin": inside,
            "verdict": ("no large difference (90% CI inside the margin)" if inside
                        else "material difference or not resolved (90% CI crosses the margin)")}


def judge_agreement(ratings: list[dict], key: dict, judged: list[dict]) -> dict | None:
    if not judged:
        return None
    by_screen = {j["item_id"]: j for j in judged}
    agree = n = agree_decided = n_decided = 0
    table = Counter()
    for r in ratings:
        j = by_screen.get(r["screen_id"])
        if j is None or key[r["screen_id"]]["kind"] != "comparison" or j["verdict"] is None:
            continue
        h, m = r["forced_choice"], j["verdict"]
        n += 1; agree += h == m
        table[(h, m)] += 1
        if h != 0 and m != 0:
            n_decided += 1; agree_decided += h == m
    # Cohen's kappa over the 3-way positional labels
    obs = agree / n if n else float("nan")
    hm = Counter(h for (h, _), c in table.items() for _ in range(c))
    mm = Counter(m for (_, m), c in table.items() for _ in range(c))
    exp = sum(hm[x] * mm[x] for x in (0, 1, 2)) / (n * n) if n else float("nan")
    kappa = (obs - exp) / (1 - exp) if n and exp < 1 else float("nan")
    return {"screens_compared": n, "agreement": round(obs, 4) if n else None,
            "kappa": round(kappa, 4) if n else None,
            "agreement_when_both_decided": round(agree_decided / n_decided, 4) if n_decided else None,
            "of_decided": n_decided}


def judge_stratum(judged: list[dict]) -> dict | None:
    if not judged:
        return None
    c = Counter()
    for j in judged:
        c[positional_to_arm(j["verdict"], j["arms_in_order"])] += 1
    decided = c["A"] + c["B"]
    lo, hi = wilson(c["B"], decided, 1.96)
    return {"items": sum(c.values()), "A_wins": c["A"], "B_wins": c["B"], "ties": c["tie"],
            "unparsed": c[None], "B_win_rate_decided": round(c["B"] / decided, 4) if decided else None,
            "ci95": [round(lo, 4), round(hi, 4)]}


def run() -> dict:
    key = json.load(open(W / "screens_key.json"))["key"]
    ratings = load_jsonl(W / "ratings.jsonl")
    out = {
        "screens_rated": len(ratings),
        "planted": planted_catch(ratings, key),
        "primary_forced_choice": forced_choice(ratings, key),
        "criteria_comparison_stratum": criterion_rates(ratings, key, "comparison"),
        "criteria_recovery_stratum": criterion_rates(ratings, key, "recovery"),
        "judge_agreement_on_screens": judge_agreement(ratings, key, load_jsonl(W / "judge_screens.jsonl")),
        "judge_full_stratum": judge_stratum(load_jsonl(W / "judge_stratum.jsonl")),
    }
    if not out["planted"]["gate_passed"]:
        out["primary_forced_choice"]["withdrawn"] = ("planted-defect gate not met" if out["planted"]["of"] >= 12
                                                     else "not all planted screens rated yet")
    return out


if __name__ == "__main__":
    print(json.dumps(run(), indent=1))
