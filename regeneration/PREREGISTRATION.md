# Pre-registration: GPT-3.5-Turbo against GPT-5.6 on MIRIAD passages

Written before any regenerated pair existed or was read. The commit that adds
this file is the timestamp. Deviations are appended at the bottom, dated, and
never edited into the text above.

## Question

Referee #4 asked whether a newer model would produce materially different
question-answer pairs from the same passages. This is a pair-level comparison.
No retrieval or question-answering measurement is part of it.

## Arms

- **A, original.** GPT-3.5-Turbo output as generated for the corpus, read from
  the raw generation output on the group share, so it is unconditioned on the
  keyword filter.
- **B, regenerated.** `gpt-5.6-sol`, reasoning effort `low`, Batch API,
  `max_output_tokens` 4000, prompt from `data_generation/generate_dataset.py`
  unchanged and hash-pinned. `temperature` omitted, since reasoning models do not
  accept it; the original set 0, so the arms differ in sampling as well as model.

Both arms pass through the same two parsers, reported separately, and the
released keyword filter (`the passage`, `the study`, case-insensitive, answer
only).

## Sample

2,000 passages from the retained generation run on the group share (1,135,954
passages, 32% of the run behind the release), seed `20260829`, ids sorted then
shuffled, fixed order. Ordered ids in `sample_ids.json`, checksum
`sha256=2e2634c0391956cde8ecc8ad3443d0f672a82de671beb670c42843defcb8ea16`.
Any prefix of the order, within a stratum, is a valid sample.

Arm B is generated for all 2,000.

## Measurements needing no evaluator, all 2,000

Parse survival under each parser; rate of raw answers containing a trigger
phrase; keyword-filter survival; pairs kept per passage; passage coverage
recovered; answer content present in the source passage; question and answer
length; near-duplicate rate within a passage. Descriptive, reported for both
arms. Arm A's baseline on this frame is measured, not taken from the paper:
72.2% filter survival, and 18.2% / 7.4% / 14.1% / 60.3% of passages keeping
0 / 1 / 2 / 3 pairs.

## Human evaluation, one evaluator

Instrument: `quality_control/streamlit_app/` with its three criteria unchanged.

- **Factual** — the answer should be factually correct and accurate.
- **Grounded in Passage** — the answer should be fully supported by the passage text.
- **Relevant** — the Q&A should refer to medically relevant content; it is
  irrelevant if it contains specific details about a study's experimental design,
  statistical analysis methods, tables or figures, dates, locations, funding, or
  other details not essential to the key medical knowledge.

Two strata, taken in draw order within each:

| stratum | passages | what is rated |
| --- | --- | --- |
| A kept all three pairs | first 90 | both sets on the three criteria, then one forced choice |
| A kept no pair | first 30 | B's set on the three criteria |

Forced choice, verbatim: *Which set would you rather have in a retrieval corpus
for medical question answering? A / B / no difference.*

**Blinding.** Set order randomised per screen under seed `4711`, formatting
normalised. The evaluator does not know which arm is which.

**Planted defects.** 12 extra screens, mixed in blind: in six, one answer
contradicts the passage; in six, one answer is rewritten to contain a trigger
phrase. Catch rate reported. **Fewer than 9 of 12 caught withdraws the
forced-choice result.**

## Endpoints

**Primary.** Forced-choice win rate among decided comparisons, binomial 95%
interval, tested against a 15-point margin: equivalence is concluded when the
90% interval lies inside 35% to 65%. About 90 screens with roughly 40% ties
leave about 54 decisions and an interval near 11 points wide, so **this supports
"no large difference" and cannot support "no small difference."** A result
inside the margin is reported in those words.

**Secondary.** Rubric rates per arm on the comparison stratum, reported in the
paper's units as a guard against gross regression; they sit near a ceiling and
are **not a test**. Rubric rates on the recovery stratum, reported as a
single-arm rate anchored against the evaluator's own comparison-stratum rate.
Tie rate.

**LLM judge.** The same forced choice over every comparison-stratum passage in
the 2,000, by a model that is not the generator. Its win rate is reported with
its agreement against the evaluator's 90 screens. It extends the sample; it does
not replace the human result.

## Analysis, fixed

1. All screens rated before any analysis. No interim looks.
2. Planted screens excluded from endpoints and reported only as the catch rate.
3. Ties reported and excluded from the win-rate denominator.
4. No subgroup analysis. Anything run later is exploratory and labelled so.

## What is out of scope, and why

Downstream retrieval, because it needs a corpus index and a stratum design the
referee's question does not require. Other models, MedGemma included: the
referee wrote "and/or", and closed-weight against open-weight is a separate
question. A GPT-3.5 arm with a corrected prompt, because no rebuild would use
GPT-3.5, and the phrasing mechanism is already visible in the trigger-phrase
rate.

## Reporting

A material difference is reported as one, with the regeneration cost stated. No
result is described as a mandate to rebuild or not to rebuild. The blinded
screens, the ratings and both seeds are released.


## Observations logged before any rating, 2026-08-29

Recorded for transparency about what was known when rating began. Nothing above
is changed by them.

The countable measurements over all 2,000 passages show arm B's answers overlap
their source passage far less than arm A's: median verbatim 4-gram overlap 0.029
against 0.284, and even on a paraphrase-tolerant content-word measure 34.0% of
B's answers have fewer than half their content words anywhere in the passage,
against 3.3% for A. The text above assumes the three criteria sit near a ceiling
on both arms; for Grounded in Passage on arm B that assumption may not hold.
The reporting rule for the criteria is unchanged: descriptive, in the paper's
units, not a test. If the groundedness rate on arm B is well below ceiling, that
is a finding, and it will be reported as one rather than as a preference.

## Deviations

**2026-08-29, prompt not byte-identical in the first generation run.** The
prompt extractor read the raw source characters between the triple quotes
instead of evaluating the literal, so the escape `\n` at the end of
`main_prompt` was sent as a backslash and an `n` followed by the line break,
where the original script sent the line break alone. Every other byte of both
prompt parts was identical. All 2,000 requests in batch
`batch_6a9eb6adc4448190a0ed43611f50ff4a` carried this one stray token
immediately before the passage. The extractor now evaluates the literal and is
checked against Python's own evaluation. Whether arm B is regenerated under the
exact prompt, or the run is kept with this caveat, is recorded below when
decided.
