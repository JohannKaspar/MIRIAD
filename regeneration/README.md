# Regeneration arm (T-215)

Regenerates QA pairs from MIRIAD source passages with a newer model, using the
released prompt unchanged, so the two arms differ in the model and not in the
pipeline around it.

## Why each piece exists

The pair count is not a free measurement. `data_generation/generate_dataset.py`
asks for exactly three questions per passage, and 10,677,724 raw pairs over
3,560,470 passages is 2.999, so generation yield is capped by construction. Only
two things can move the released pair count:

- the **parser**, `re.findall(r'Question \d+: ...')`, which returns nothing for
  output in another shape;
- the **keyword filter**, two case-insensitive phrases (`the passage`,
  `the study`) matched against the answer only.

Both respond to output formatting and phrasing habit rather than to content, so
`run.py parse` reports them separately and runs both parsers over both arms.

The filter also behaves as a passage-level selector rather than a pair-level
one: of the 3,560,470 generated passages, 35.2% lost all three pairs and 42.9%
kept all three, where independent per-pair filtering at the observed 54.5%
survival rate would give 9.4% and 16.2%. A newer model that avoids the two
phrases recovers much of that, which looks like a large coverage gain and is a
phrasing effect.

## Sampling frame

`--frame step4` reads `preprocessing/medicine_passages/medicine_passages_<shard>.json`,
the chunked passages before any model saw them. This frame still holds the
1,251,824 passages (35.2%) whose pairs were all removed, which is where the
original model failed hardest.

`--frame released` reads the published corpus, which only contains passages that
kept at least one pair. Usable, but it restricts the comparison to ground the
original model already handled, and that has to be declared.

Shards where `shard % 10 == 0 or shard == 21` are excluded: generation returned
early on them, so passages exist but no original arm does.

## Deviations from the released pipeline

Recorded here because they belong in the write-up, not because they are hidden.

| | released | here |
| --- | --- | --- |
| transport | one synchronous call per passage | Batch API, same granularity, half price |
| `max_output_tokens` | 1000 | 4000, since reasoning tokens count as output |
| `temperature` | 0 | omitted; reasoning models do not accept it |
| parser | `Question \d+:` only | both, reported separately |
| prompt | unchanged | unchanged, hash-pinned in `prompt.py` |

## Usage

```bash
python -m regeneration.run sample --frame step4 \
    --source ../preprocessing/medicine_passages --size 2000 --seed 20260829
python -m regeneration.run build --limit 120      # prefix of the same sample
python -m regeneration.run submit
python -m regeneration.run status
python -m regeneration.run fetch
python -m regeneration.run parse
```

`sample` draws the full 2,000 once and stores them in a fixed order, so any
prefix is itself a valid sample. Generating 120 now and the remaining 1,880
later gives one pooled sample under one pre-registration.
