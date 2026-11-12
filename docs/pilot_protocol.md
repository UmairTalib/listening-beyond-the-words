# Pilot protocol

This page fixes how the pilot runs and how its result is read. It is committed here before the first training run; the commit date shows it was fixed in advance. Nothing on this page changes after test scores are seen. The only allowed changes are listed below and recorded in the settings change log.

## Design

| Item | Pilot |
| --- | --- |
| Question | Does CPG beat B0 on normal audio, and without words? |
| Data | E-DAIC sessions 300–492, 189 participants. Sessions 600–718 stay unopened. |
| Arms | Zero-shot Voxtral (scoring only), B0, B1 and CPG |
| Design | 2 rounds × 5 folds, split by participant, stratified by label and gender. 30 training runs. |
| Test views | `x0` (original audio) and `rev` (each 250 ms window played backwards, so the words are removed) |
| Main numbers | Δ = CPG − B0 in mean per-fold AUROC on each test view, over 10 fold-runs, with 80% and 95% intervals |
| Also reported | B1 − B0 and CPG − B1, values per round, zero-shot AUROC, Spearman correlation with the PHQ-8 totals |
| Compute | About 100 V100 GPU-hours |

## Speech used

Fixed on 12 November 2026, before any training. The settings are in `configs/pilot/data.yaml`.

- **Transcripts.** The pilot uses the E-DAIC transcripts, because the DAIC-WOZ human transcripts had not arrived when the segments were built. They come from automatic speech recognition, hold the participant's lines and contain almost no fillers such as "um" and "uh". The words-only view `x1` therefore has no fillers, while `x0` keeps them. This is reported as a limitation.
- **Labels.** A participant is positive if the PHQ-8 total in the official split files is 10 or more. For every pilot participant this total equals the sum of the eight item scores. 57 of the 189 participants are positive. The binary column disagrees for 15 participants and is not used. A third label file gives a different total for 2 participants, which changes no label. Gender comes from the split files (87 female, 102 male); `detailed_lables.csv` codes it differently for 3 participants.
- **Interviewer filter.** The interviewer is mostly inaudible in these recordings, but leaks into some participant lines, and an assistant can be heard at the start of a few sessions. A transcript line is dropped if at least half of its speech, cut into 1.5 s windows, sounds more like the interviewer than like the participant to a speaker-verification model (WavLM x-vectors, pinned in `configs/pilot/models.yaml`), or if it contains one of ten known interviewer phrases. A line that lasts at least 1.5 s but holds no 0.75 s of continuously detected speech is dropped as well, because neither the model nor a listener has judged it; such lines hold 1.9% of the words and are balanced between the PHQ-8 classes. In a blind listening check of 35 lines, another voice was heard in an estimated 3.6% of all lines before the filter and in about 1.3% after it. The rest is reported as a limitation.
- **Word times.** A line of 30 s or more with fewer than one word per second has broken timestamps (a few words spread over minutes) and is dropped. The words of every kept line are then placed on the audio by forced alignment with wav2vec2-large-960h-lv60-self (pinned in `configs/pilot/models.yaml`); digits are spelled out for the aligner only. A line that cannot be aligned is dropped and counted.
- **Checks before training.** The share of dropped words must not differ between PHQ-8-positive and PHQ-8-negative participants (two-sided Mann–Whitney test, p ≥ 0.05); otherwise the filter is reviewed before any training. Twenty random segments are listened to before training; if another voice is heard in more than two of them, the filter is reviewed.

## Fixed settings

These values are fixed before the first training run. Only the two allowed changes below may alter m or the number of epochs.

| Setting | Value |
| --- | --- |
| Model | Voxtral Mini 3B at a pinned revision, fp16 with fp32 master weights and loss scaling. The audio encoder is frozen and its output cached once per view. |
| LoRA | r = 16, α = 32, dropout 0.05 on q, k, v, o, gate, up and down |
| Optimiser | AdamW, learning rate 1.0e-4 (adapter 2.0e-5), cosine schedule with 5% warm-up, weight decay 0.01, gradient clip 1.0 |
| Epochs | 3; the final checkpoint is used |
| Prompt | "Based on this speech, does the speaker have elevated depressive symptoms? Answer Yes or No." |
| Score | z = logit("Yes") − logit("No"); a participant's score is the mean z over their segments |
| Segments and bags | Exactly 40 words, 300 ms gaps. A bag is 4 segments of one participant; 4 bags per step (2 positive, 2 negative); class weights 1. |
| Views | `x0` original; `x1` re-voiced (Kokoro-82M, same gender, words only); `x2` muffled (400 Hz low-pass); `rev` for testing only |
| Loss | CE(x0) + 0.25 CE(x1) + 0.5 CE(x2) + λG · gain term |
| Gain term | Margin m = 0.3, softness s = 0.1, λG = 1 ramped up over the first 20% of steps, computed in fp32 |
| Word check (F4) | faster-whisper large-v3 and wav2vec2-large-960h-lv60-self |
| Intervals | Participant bootstrap stratified by label, 10,000 resamples |

## Feasibility checks

All five checks must pass before the main numbers are read. A failed check is fixed and the affected step rerun. A failed check is not a result.

| Check | Passes if |
| --- | --- |
| F1 Runs finish | All 30 runs finish with no NaN or inf in the loss or the gradients |
| F2 Gain term active | In the CPG runs, the hinge is active on 20–80% of bags in epochs 2 and 3, after at most the one allowed change of m. If it is still outside 20–80%, this is reported as a finding and decided with the supervisor. |
| F3 B0 learns | B0's mean AUROC on `x0` is at least 0.60. Below that, the fallback for a B0 near chance (a linear readout head) is discussed with the supervisor before any effect is read |
| F4 Words removed | On `rev`, both ASR systems reach a WER of at least 0.80, and a text classifier on their output predicts the label with an AUROC of at most 0.60. Otherwise the whole-turn backup (each whole turn played backwards) is used, checked the same way |
| F5 Cost | A three-view run takes at most 4 GPU-hours. If not, every arm uses 8 instead of 16 segments per participant per epoch, and B1's round 2 is cut first. |

## Decision rule

The decision uses the point estimates of Δ on `x0` and on `rev`. The 80% intervals and the values per round are reported next to them as a check, not as a test.

| Outcome | Rule | Decision |
| --- | --- | --- |
| Green: beats B0 | Δ on `x0` ≥ +0.03 | Go |
| Green: robust without words | −0.04 < Δ on `x0` < +0.03, and Δ on `rev` ≥ +0.03 | Go. The thesis leads with robustness. |
| Amber: no visible effect | −0.04 < Δ on `x0` < +0.03, and Δ on `rev` < +0.03 | Decide with the supervisor, using the diagnostics |
| Red: worse on normal audio | Δ on `x0` ≤ −0.04 | No-go for the recipe as it stands |

## Allowed setting changes

Besides the pre-set fallbacks in the feasibility checks, two changes are allowed. Each comes from training logs only and is made once, before the remaining runs start. It applies to all arms, is dated in the settings change log below and is sent to the supervisor in one line.

- **Gain-alive rule.** One CPG run (round 1, fold 1) runs before any other CPG run. If the hinge is active on fewer than 20% of bags in epochs 2 and 3, m becomes 0.5. If it is active on more than 80%, m becomes 0.1.
- **4-epoch rule.** If the training loss of that same CPG run is still clearly falling in epoch 3, every arm trains 4 epochs, and B0's round 1 is rerun with 4 epochs.

If m or the number of epochs changes, that first CPG run is redone with the new setting.

## If time runs short

Cut in this order: B1's round 2, then B1, then round 2 of B0 and CPG. The minimum version (B0 and CPG, round 1 only, both test views) is read with the same rule and a wider error. The rule is never read before the five checks pass.

## What the main study inherits

- The main study treats every pilot result as seen.
- It draws new partitions with new seeds and reuses no pilot run.
- E-DAIC sessions 600–718 stay unopened until the external test.

## Settings change log

Every change to a frozen setting goes here, with its date and the training-log evidence. Test scores never count as evidence.

| Date | Study | Setting | Old | New | Evidence | Commit |
| --- | --- | --- | --- | --- | --- | --- |
| – | – | – | – | – | – | – |
