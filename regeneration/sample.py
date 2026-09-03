"""Drawing the passage sample, reproducibly and in one fixed order.

The sample is drawn once at full size and stored as an ordered list, so any
prefix is itself a valid sample. Generating the first 120 now and the remaining
1,880 later therefore yields one pooled sample under one pre-registration,
rather than two samples that cannot be combined.

Three frames are supported, best first:

  generated  the raw generation output, shard_<n>.json, holding every passage
             that was actually sent to GPT-3.5-Turbo together with its raw
             llm_output and parsed pairs. This is the frame the design wants.
             It is unconditioned on the keyword filter, so it still contains
             the passages whose three pairs were all removed, and because it
             carries llm_output it supports a raw-against-raw comparison of
             parse and filter survival rather than raw-against-filtered.

  step4      medicine_passages_<shard>.json, the chunked passages before any
             model saw them. Much larger than the generated frame: generation
             ran on a subset of papers, so most step4 passages have no original
             arm at all. Use only to characterise what was left out.

  released   the published corpus. Only passages that kept at least one pair
             are present, so this frame excludes the cases where the original
             model failed hardest. Usable, but the restriction has to be
             declared.

Which shards were generated is read from the generation output rather than
inferred from the code. data_generation/generate_dataset.py returns early when
`start_shard % 10 == 0 or start_shard == 21`, but the output on the group share
holds shards 0 to 89 with no such gaps, so the released code does not describe
the run that produced the corpus.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path


def generated_shards(generated_dir: Path) -> set[int]:
    """The shards generation actually produced, read from disk.

    Deliberately not derived from the early return in generate_dataset.py: that
    would exclude shards 0, 10, 20, 21, 30 and so on, which the output shows
    were generated after all.
    """
    return {int(p.stem.rsplit("_", 1)[1]) for p in generated_dir.glob("shard_*.json")}


@dataclass(frozen=True)
class Passage:
    passage_id: str          # "<shard>_<paper_id>_<passage_index>", as in qa_id
    shard: int
    paper_id: str
    passage_index: int
    text: str


def iter_generated(generated_dir: Path, shards: list[int] | None = None):
    """Yield every passage that was actually sent to the original model.

    Each shard_<n>.json maps paper_id -> {"passage_info": {passage_id: {...}}},
    where each entry carries passage_text, the parsed qa pairs and the raw
    llm_output. Only passage_text is returned here; the original arm is read
    separately so the two arms stay clearly apart.
    """
    for path in sorted(generated_dir.glob("shard_*.json"),
                       key=lambda p: int(p.stem.rsplit("_", 1)[1])):
        shard = int(path.stem.rsplit("_", 1)[1])
        if shards is not None and shard not in shards:
            continue
        for paper_id, paper in json.loads(path.read_text()).items():
            for passage_id, entry in paper.get("passage_info", {}).items():
                index = int(passage_id.rsplit("_", 1)[1])
                yield Passage(passage_id, shard, str(paper_id), index, entry["passage_text"])


def iter_step4(passages_dir: Path, shards: list[int] | None = None):
    """Yield every chunked passage, including those generation never reached."""
    for path in sorted(passages_dir.glob("medicine_passages_*.json")):
        if path.name.startswith("._"):
            continue
        shard = int(path.stem.rsplit("_", 1)[1])
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
