"""Drawing the passage sample, reproducibly and in one fixed order.

The sample is drawn once at full size and stored as an ordered list, so any
prefix is itself a valid sample. Generating the first 120 now and the remaining
1,880 later therefore yields one pooled sample under one pre-registration,
rather than two samples that cannot be combined.

Two frames are supported:

  step4     preprocessing/medicine_passages/medicine_passages_<shard>.json,
            the chunked passages before any model saw them. This is the frame
            the design wants: it still contains the 1,251,824 passages (35.2%)
            whose three pairs were all removed by the keyword filter.

  released  the published corpus. Only passages that kept at least one pair are
            present, so this frame silently excludes the cases where the
            original model failed hardest. Usable, but the restriction has to
            be declared.

Generation did not run on every shard: generate_dataset.py returns early when
`start_shard % 10 == 0 or start_shard == 21`. Passages exist for those shards
but no original arm does, so they are excluded from the step4 frame.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path


def shard_was_generated(shard: int) -> bool:
    """Mirrors the early return in data_generation/generate_dataset.py."""
    return not (shard % 10 == 0 or shard == 21)


@dataclass(frozen=True)
class Passage:
    passage_id: str          # "<shard>_<paper_id>_<passage_index>", as in qa_id
    shard: int
    paper_id: str
    passage_index: int
    text: str


def iter_step4(passages_dir: Path, shards: list[int] | None = None):
    """Yield every passage from the step-4 files, generated shards only."""
    paths = sorted(passages_dir.glob("medicine_passages_*.json"))
    for path in paths:
        shard = int(path.stem.rsplit("_", 1)[1])
        if not shard_was_generated(shard):
            continue
        if shards is not None and shard not in shards:
            continue
        for paper in json.loads(path.read_text()):
            paper_id = str(paper["paper"]["metadata"]["paper_id"])
            for index, text in enumerate(paper["passages"]):
                yield Passage(f"{shard}_{paper_id}_{index}", shard, paper_id, index, text)


def iter_released(corpus_glob: str):
    """Yield every distinct passage from the published parquet corpus."""
    import duckdb

    rows = duckdb.sql(
        f"""
        SELECT any_value(qa_id) AS qa_id, any_value(passage_text) AS passage_text
        FROM read_parquet('{corpus_glob}')
        GROUP BY regexp_replace(qa_id, '_[0-9]+$', '')
        """
    ).fetchall()
    for qa_id, text in rows:
        shard, paper_id, index, _ = qa_id.split("_", 3)
        yield Passage(f"{shard}_{paper_id}_{index}", int(shard), paper_id, int(index), text)


def draw(passages, size: int, seed: int) -> list[Passage]:
    """Shuffle the frame under `seed` and take the first `size`.

    Sorting first makes the result independent of filesystem ordering, so the
    same seed and frame always give the same sample in the same order.
    """
    ordered = sorted(passages, key=lambda p: p.passage_id)
    if size > len(ordered):
        raise ValueError(f"asked for {size} passages, frame holds {len(ordered)}")
    random.Random(seed).shuffle(ordered)
    return ordered[:size]


def write(sample: list[Passage], path: Path, frame: str, seed: int) -> None:
    path.write_text(
        json.dumps(
            {
                "frame": frame,
                "seed": seed,
                "size": len(sample),
                "passages": [asdict(p) for p in sample],
            },
            indent=1,
        )
    )


def read(path: Path) -> tuple[dict, list[Passage]]:
    blob = json.loads(path.read_text())
    meta = {k: v for k, v in blob.items() if k != "passages"}
    return meta, [Passage(**p) for p in blob["passages"]]
