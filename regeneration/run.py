"""Command line for the regeneration arm.

    python -m regeneration.run sample --frame step4 --source ../preprocessing/medicine_passages --size 2000 --seed 20260829
    python -m regeneration.run build  --limit 120
    python -m regeneration.run submit
    python -m regeneration.run status
    python -m regeneration.run fetch
    python -m regeneration.run parse

`sample` draws the full 2000 once; `build --limit` takes a prefix of it, so a
120-passage pilot now and the rest later form one pooled sample.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from . import batch as batch_module
from . import parse as parse_module
from . import sample as sample_module

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_dotenv() -> None:
    """Read OPENAI_API_KEY and friends from a gitignored .env at the repo root.

    The key is not committed and an already-exported value always wins, so this
    only fills in what the shell did not provide.
    """
    import os

    path = REPO_ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


WORK = Path("regeneration/work")
SAMPLE = WORK / "sample.json"
REQUESTS = WORK / "requests.jsonl"
BATCH = WORK / "batch.json"
RESULTS = WORK / "results.jsonl"
PAIRS = WORK / "pairs.jsonl"


def cmd_sample(args) -> None:
    source = Path(args.source)
    if args.frame == "generated":
        passages = sample_module.iter_generated(source)
    elif args.frame == "step4":
        passages = sample_module.iter_step4(source)
    else:
        passages = sample_module.iter_released(str(args.source))
    drawn = sample_module.draw(passages, args.size, args.seed)
    WORK.mkdir(parents=True, exist_ok=True)
    sample_module.write(drawn, SAMPLE, args.frame, args.seed)
    print(f"drew {len(drawn)} passages from frame '{args.frame}' with seed {args.seed}")
    print(f"  -> {SAMPLE}")
    print(f"  first three: {[p.passage_id for p in drawn[:3]]}")


def cmd_build(args) -> None:
    meta, passages = sample_module.read(SAMPLE)
    selected = passages[: args.limit] if args.limit else passages
    config = batch_module.GenerationConfig(
        model=args.model, reasoning_effort=args.effort, max_output_tokens=args.max_output_tokens
    )
    batch_module.write_requests(selected, config, REQUESTS)
    input_tokens = sum(len(batch_module.prompt_module.build(p.text)) for p in selected) / 4.0
    print(f"built {len(selected)} requests of {meta['size']} sampled  -> {REQUESTS}")
    print(f"  model {config.model}, effort {config.reasoning_effort}, "
          f"max_output_tokens {config.max_output_tokens}")
    print(f"  rough input volume: {input_tokens/1e6:.2f}M tokens")


def cmd_submit(args) -> None:
    config = batch_module.GenerationConfig(
        model=args.model, reasoning_effort=args.effort, max_output_tokens=args.max_output_tokens
    )
    record = batch_module.submit(REQUESTS, config)
    BATCH.write_text(json.dumps(record, indent=1))
    print(f"submitted {record['batch_id']} (status {record['status']})  -> {BATCH}")


def cmd_status(args) -> None:
    batch_id = json.loads(BATCH.read_text())["batch_id"]
    print(json.dumps(batch_module.status(batch_id), indent=1))


def cmd_fetch(args) -> None:
    batch_id = json.loads(BATCH.read_text())["batch_id"]
    batch_module.fetch(batch_id, RESULTS)
    print(f"wrote {RESULTS}")


def cmd_parse(args) -> None:
    counts = Counter()
    usage = Counter()
    n = 0
    with PAIRS.open("w") as out:
        for passage_id, text, meta in batch_module.read_results(RESULTS):
            n += 1
            released = parse_module.parse_released(text)
            tolerant = parse_module.parse_tolerant(text)
            kept = parse_module.apply_keyword_filter(tolerant)
            counts["released_parsed"] += len(released)
            counts["tolerant_parsed"] += len(tolerant)
            counts["kept_after_filter"] += len(kept)
            counts["released_zero"] += 1 if not released else 0
            counts["tolerant_zero"] += 1 if not tolerant else 0
            counts["all_three_lost"] += 1 if tolerant and not kept else 0
            counts["truncated"] += 1 if meta.get("incomplete_reason") else 0
            counts["errored"] += 1 if meta.get("error") else 0
            for key in ("input_tokens", "output_tokens", "reasoning_tokens"):
                usage[key] += meta.get(key) or 0
            out.write(json.dumps({
                "passage_id": passage_id,
                "pairs": [{"question": q, "answer": a} for q, a in kept],
                "released_parsed": len(released),
                "tolerant_parsed": len(tolerant),
                "meta": meta,
            }) + "\n")

    print(f"{n} passages  -> {PAIRS}\n")
    print(f"  pairs, released parser      {counts['released_parsed']:>7}  "
          f"({counts['released_parsed']/max(n,1):.3f} per passage)")
    print(f"  pairs, tolerant parser      {counts['tolerant_parsed']:>7}  "
          f"({counts['tolerant_parsed']/max(n,1):.3f} per passage)")
    print(f"  kept after keyword filter   {counts['kept_after_filter']:>7}  "
          f"({counts['kept_after_filter']/max(counts['tolerant_parsed'],1):.1%} survival)")
    print(f"  passages parsing to zero    released {counts['released_zero']}, "
          f"tolerant {counts['tolerant_zero']}")
    print(f"  passages losing every pair  {counts['all_three_lost']}")
    print(f"  truncated / errored         {counts['truncated']} / {counts['errored']}")
    if usage["output_tokens"]:
        print(f"\n  tokens: {usage['input_tokens']/1e6:.3f}M in, "
              f"{usage['output_tokens']/1e6:.3f}M out "
              f"(of which {usage['reasoning_tokens']/1e6:.3f}M reasoning)")
        cost = usage["input_tokens"] / 1e6 * 2.00 + usage["output_tokens"] / 1e6 * 10.00
        print(f"  cost at gpt-5.6-sol batch rates: ${cost:,.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="regeneration.run", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("sample", help="draw the full passage sample once")
    p.add_argument("--frame", choices=["generated", "step4", "released"], required=True)
    p.add_argument("--source", required=True,
                   help="generated: the shard_*.json directory. step4: medicine_passages. "
                        "released: a parquet glob")
    p.add_argument("--size", type=int, default=2000)
    p.add_argument("--seed", type=int, required=True)
    p.set_defaults(func=cmd_sample)

    for name, func, helptext in (("build", cmd_build, "write the batch request file"),
                                 ("submit", cmd_submit, "upload and start the batch")):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("--model", default="gpt-5.6-sol")
        p.add_argument("--effort", default="low")
        p.add_argument("--max-output-tokens", dest="max_output_tokens", type=int, default=4000)
        if name == "build":
            p.add_argument("--limit", type=int, default=None,
                           help="generate only the first N of the sample")
        p.set_defaults(func=func)

    for name, func, helptext in (("status", cmd_status, "poll the batch"),
                                 ("fetch", cmd_fetch, "download batch output"),
                                 ("parse", cmd_parse, "parse, filter and report survival")):
        p = sub.add_parser(name, help=helptext)
        p.set_defaults(func=func)

    args = parser.parse_args()
    load_dotenv()
    args.func(args)


if __name__ == "__main__":
    main()
