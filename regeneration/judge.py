"""The LLM judge: the pre-registered forced choice, by a model that is not the
generator, served on the Swiss AI Research Platform.

    python -m regeneration.judge screens     # the human's screens, same blinded order
    python -m regeneration.judge stratum     # every comparison-stratum passage in a run

`screens` judges exactly what the evaluator saw, in the same set order, so the
agreement figure is not confounded by position. `stratum` judges every
comparison-stratum passage of a full run, set order randomised under its own
seed. Both are resumable and write one JSON line per item.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

W = Path("regeneration/work")
BASE_URL = "https://api.swissai.svc.cscs.ch/v1"
MODEL = "CSCS-Inference/zai-org/GLM-5.2"
STRATUM_SEED = 4712

QUESTION = "Which set would you rather have in a retrieval corpus for medical question answering?"
SYSTEM = ("You compare two sets of question-answer pairs, each set generated from the same "
          "passage of a medical paper, and say which set you would rather have in a retrieval "
          "corpus for medical question answering. Judge the sets as a whole. Reply with exactly "
          "one character and nothing else: 1 for Set 1, 2 for Set 2, 0 for no difference.")


def _client():
    from openai import OpenAI
    key = os.environ.get("SWISS_AI_RESEARCH")
    if not key:
        raise RuntimeError("SWISS_AI_RESEARCH not set; source ~/.api_keys")
    return OpenAI(base_url=BASE_URL, api_key=key)


def render(passage: str, sets: list[list[dict]]) -> str:
    parts = ["PASSAGE:", passage.strip(), ""]
    for i, pairs in enumerate(sets, 1):
        parts.append(f"SET {i}:")
        for j, p in enumerate(pairs, 1):
            parts.append(f"Q{j}: {p['question']}")
            parts.append(f"A{j}: {p['answer']}")
        parts.append("")
    parts.append(QUESTION)
    parts.append("Answer with exactly one character: 1, 2, or 0.")
    return "\n".join(parts)


def ask(client, prompt: str, model: str, retries: int = 6, max_tokens: int = 1024) -> tuple[int | None, str]:
    delay = 3
    for attempt in range(retries):
        try:
            # GLM-5.2 thinks before it answers and the thinking counts against
            # max_tokens; a cap of 8 returned nothing at all. Leave room, keep
            # thinking on, and take the last standalone digit of the visible reply.
            r = client.chat.completions.create(
                model=model, temperature=0, max_tokens=max_tokens,
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": prompt}])
            text = (r.choices[0].message.content or "").strip()
            digits = re.findall(r"(?<![\d.])[012](?![\d.])", text)
            used = r.usage.completion_tokens if r.usage else None
            return (int(digits[-1]) if digits else None), json.dumps({"text": text[-200:], "completion_tokens": used, "finish": r.choices[0].finish_reason})
        except Exception as e:  # noqa: BLE001
            print(f"  retry {attempt}: {str(e)[:100]}", file=sys.stderr, flush=True)
            time.sleep(delay); delay = min(delay * 2, 60)
    return None, "GAVE UP"


def _done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(l)["item_id"] for l in path.read_text().splitlines() if l.strip()}


def run_items(items: list[dict], out: Path, model: str, workers: int,
              max_tokens: int = 1024, redo_unparsed: bool = False) -> None:
    if redo_unparsed and out.exists():
        kept = [l for l in out.read_text().splitlines() if l.strip() and json.loads(l)["verdict"] is not None]
        out.write_text("\n".join(kept) + ("\n" if kept else ""))
    done = _done(out)
    todo = [it for it in items if it["item_id"] not in done]
    print(f"{len(todo)} to judge, {len(done)} already done  -> {out}", flush=True)
    client = _client()
    with out.open("a") as fh, ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(ask, client, render(it["passage"], it["sets"]), model, 6, max_tokens): it for it in todo}
        for n, fut in enumerate(as_completed(futs), 1):
            it = futs[fut]
            verdict, raw = fut.result()
            rec = {k: v for k, v in it.items() if k not in ("passage", "sets")}
            rec.update({"verdict": verdict, "raw": raw, "model": model})
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            if n % 10 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)}", flush=True)


def items_from_screens(swap: bool = False) -> list[dict]:
    """The evaluator's two-set screens, in the evaluator's blinded order.

    `swap` reverses the set order on every screen; judging both orders and
    counting disagreements measures the judge's position bias directly.
    """
    screens = json.load(open(W / "screens_blinded.json"))["screens"]
    key = json.load(open(W / "screens_key.json"))["key"]
    items = []
    for s in screens:
        if s["format"] != "two_sets" or key[s["screen_id"]]["kind"] != "comparison":
            continue
        arms = list(key[s["screen_id"]]["arms_in_order"]); sets = [x["pairs"] for x in s["sets"]]
        if swap:
            arms, sets = arms[::-1], sets[::-1]
        items.append({"item_id": s["screen_id"], "screen_id": s["screen_id"],
                      "passage_id": key[s["screen_id"]]["passage_id"],
                      "arms_in_order": arms, "swapped": swap,
                      "passage": s["passage"], "sets": sets})
    return items


def items_from_stratum(pairs_path: Path) -> list[dict]:
    """Every comparison-stratum passage of a run: A kept three, B non-empty."""
    from .metrics import load_arm_a, load_arm_b
    from .prepare_screens import clean
    sample = json.load(open(W / "sample.json"))
    passages = {p["passage_id"]: p["text"] for p in sample["passages"]}
    order = [p["passage_id"] for p in sample["passages"]]
    A, B = load_arm_a(W / "original_arm.jsonl"), load_arm_b(pairs_path)
    rng = random.Random(STRATUM_SEED)
    items = []
    for pid in order:
        if pid not in A or pid not in B or A[pid]["kept"] != 3 or B[pid]["kept"] == 0:
            continue
        sets = [("A", [{"question": clean(p["question"]), "answer": clean(p["answer"])}
                       for p in A[pid]["pairs"] if p["survives_filter"]]),
                ("B", [{"question": clean(p["question"]), "answer": clean(p["answer"])}
                       for p in B[pid]["pairs"]])]
        rng.shuffle(sets)
        items.append({"item_id": pid, "passage_id": pid, "arms_in_order": [s[0] for s in sets],
                      "passage": passages[pid], "sets": [s[1] for s in sets]})
    return items


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["screens", "stratum"])
    ap.add_argument("--pairs", default=str(W / "pairs.jsonl"), help="stratum mode: arm B pairs file")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=1024, help="room for thinking; raise if verdicts come back unparsed")
    ap.add_argument("--redo-unparsed", action="store_true", help="drop unparsed verdicts and judge those items again")
    ap.add_argument("--swap", action="store_true", help="screens mode: reverse set order on every screen (position-bias control)")
    args = ap.parse_args()
    items = items_from_screens(swap=args.swap) if args.mode == "screens" else items_from_stratum(Path(args.pairs))
    if args.limit:
        items = items[: args.limit]
    out_name = f"judge_{args.mode}{'_swapped' if args.swap else ''}.jsonl"
    run_items(items, W / out_name, args.model, args.workers,
              max_tokens=args.max_tokens, redo_unparsed=args.redo_unparsed)


if __name__ == "__main__":
    main()
