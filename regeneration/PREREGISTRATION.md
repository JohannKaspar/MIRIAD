# Pre-registration: GPT-3.5-Turbo against GPT-5.6 on MIRIAD passages

Written before any pair was read. Committed on the date in the git history; that
commit is the timestamp. Deviations are appended at the bottom with their dates,
never edited into the text above.

## Question

Referee #4 asked whether a newer model would produce materially different
question-answer pairs from the same source passages. This registers the
pair-level comparison that answers it. No downstream retrieval or
question-answering measurement is part of this study.

## Design

One passage set, two arms, everything else held fixed.

- **Arm A, original.** GPT-3.5-Turbo output already generated for these
  passages, taken from the raw generation output rather than the released
  corpus, so it is unconditioned on the keyword filter.
- **Arm B, regenerated.** `gpt-5.6-sol`, reasoning effort `low`, Batch API,
  `max_output_tokens` 4000, the prompt from `data_generation/generate_dataset.py`
  unchanged and hash-pinned. `temperature` is omitted because reasoning models do
  not accept it, where the original set `temperature=0`; the arms therefore
  differ in sampling as well as in model.

Both arms pass through one parser and the released keyword filter
(`quality_control/keyword_filter.py`: `the passage` and `the study`,
case-insensitive, matched against the answer only).

## Sample

- **Frame.** The retained generation run on the group share,
  `bsse_group_mmoor/projects/miriad/data/gpt_scaled_up_dataset/`, 1,135,954
  passages. This is 32% of the 3,560,470 passages behind the release, so the
  frame is the retained portion of that run, not the whole of it. It is
  unconditioned on the keyword filter and so retains the passages whose pairs
  were all removed.
- **Draw.** 2,000 passages, seed `20260829`, ids sorted then shuffled, stored in
  one fixed order. Any prefix is itself a valid sample. Drawn on 2026-08-29; the
  ordered ids are in `sample_ids.json` with
  `sha256=2e2634c0391956cde8ecc8ad3443d0f672a82de671beb670c42843defcb8ea16` over the newline-joined
  list, so the draw is verifiable without redistributing the passage text.
- **Drawn sample.** 2,000 passages over 1,994 distinct papers and 89 shards,
  median passage length 4,451 characters.
- **Rated subset.** The first 120 of that order, spanning 42 shards.
- **Everything else** is a prefix of the same draw, so a later extension pools
  with this one under this document.

## Human evaluation

One evaluator. The instrument is `quality_control/streamlit_app/`, the app used
for the published expert review, with its three criteria unchanged:

- **Factual** — the answer should be factually correct and accurate.
- **Grounded in Passage** — the answer should be fully supported by the passage text.
- **Relevant** — the Q&A should refer to medically relevant content. It is
  irrelevant if it contains specific details about a study's experimental design,
  statistical analysis methods, tables or figures, study dates, locations,
  funding sources, or other details that are not essential for understanding the
  key medical knowledge.

Per screen: the passage, then both pair sets, each rated on the three criteria.
Then one forced choice: **set A better / set B better / no difference**.

### Primary endpoint

The forced-choice win rate among decided comparisons, reported with a binomial
95% interval and tested against the equivalence margin below.

### Equivalence margin

**15 percentage points.** Equivalence is concluded when the 90% interval on the
win rate lies inside 35% to 65%.

The margin is set by what 120 comparisons can resolve, and is stated here rather
than discovered later. Expecting roughly 40% ties leaves about 72 decided
comparisons; at a 50% win rate the 90% interval is about 40% to 60%, inside a
15-point margin and outside a 10-point one. **This study can therefore support
"no large difference" and cannot support "no small difference."** A result inside
the margin is reported in exactly those words.

### Secondary endpoints

Paired per-passage rates for each of the three criteria, arm against arm.
Descriptive; no equivalence claim is attached to them.

## Validity controls

- **Blinding.** Arm order is randomised per screen under seed `4711`, kept
  separate from the sampling seed. Formatting is normalised across arms. The
  evaluator does not know which arm is which, and the write-up states whether
  that held.
- **Planted defects, 12 items (10%).** One side carries an injected defect: in
  half, a fact contradicting the passage; in half, an answer rewritten to contain
  a trigger phrase. The catch rate is reported. **If fewer than 9 of 12 are
  caught, the human ratings are reported as unreliable** and the forced-choice
  result is withdrawn.
- **Repeat items, 18 items (15%).** Re-shown later in shuffled order. Agreement
  with the first rating is reported as the single-rater substitute for
  inter-rater agreement. Reported, not gated on.
- **Total screens.** 120 real, 12 planted, 18 repeats, 150 in all.

## Analysis, fixed in advance

1. All 150 screens are rated before any analysis is run. No interim looks.
2. Planted items are excluded from the endpoints and reported only as the catch
   rate. Repeat items count once, first rating.
3. Ties are reported and excluded from the win-rate denominator; the tie rate is
   itself reported.
4. No subgroup analysis by specialty, year or length. Any such analysis run later
   is exploratory and labelled so.

## Measurements needing no evaluator

Computed over all 2,000 passages, descriptive, not covered by the margin above:
parse survival under both parsers, keyword-filter survival, pairs per passage,
the distribution of pairs kept per passage, answer content present in the source
passage, length distributions, and near-duplicate rate within a passage.

The original arm's baseline on this frame is measured, not taken from the paper:
72.2% keyword-filter survival over 3,407,217 pairs, and 18.2% / 7.4% / 14.1% /
60.3% of passages keeping 0 / 1 / 2 / 3 pairs. The published 54.5% is not the
keyword filter's rate and is not used.

## What would change the conclusion

A win rate outside 35% to 65% is a material difference and is reported as one,
with the regeneration cost stated. A result inside the margin is reported as no
large difference at this sample size. Neither outcome is described as a mandate
to rebuild or not to rebuild the corpus.

## Deviations

None yet.
