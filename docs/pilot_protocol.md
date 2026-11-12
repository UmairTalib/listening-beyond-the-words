# Pilot protocol

This page fixes how the pilot runs and how its result is read. It is agreed with the supervisor by e-mail and committed here before the first training run. Nothing on this page changes after test scores are seen. The only allowed changes are listed below and recorded in the settings change log.

## Design

| Item | Pilot |
| --- | --- |
| Question | Does CPG beat B0 on normal audio, and without words? |
| Data | E-DAIC sessions 300–492, 189 participants. Sessions 600–718 stay unopened. |
| Arms | Zero-shot Voxtral (scoring only), B0, B1 and CPG |
| Design | 2 rounds × 5 folds, split by participant, stratified by label and gender. 30 training runs. |
| Test views | `x0` (original audio) and `rev` (reversed speech, words removed) |
| Main numbers | Δ = CPG − B0 in mean per-fold AUROC on each test view, over 10 fold-runs, with 80% and 95% intervals |
| Also reported | B1 − B0 and CPG − B1, values per round, zero-shot AUROC, Spearman correlation with the PHQ-8 totals |
| Compute | About 100 V100 GPU-hours |

## Feasibility checks

All five checks must pass before the main numbers are read. A failed check is fixed and the affected step rerun. A failed check is not a result.

| Check | Passes if |
| --- | --- |
| F1 Runs finish | All 30 runs finish with no NaN or inf in the loss or the gradients |
| F2 Gain term active | In the CPG runs, the hinge is active on 20–80% of bags in epochs 2 and 3, after at most the one allowed change of m |
| F3 B0 learns | B0's mean AUROC on `x0` is at least 0.60. Below that, the fallback for a B0 near chance (a linear readout head) is discussed with the supervisor before any effect is read |
| F4 Words removed | On `rev`, both ASR systems reach a WER of at least 0.80, and a text classifier on their output predicts the label with an AUROC of at most 0.60. Otherwise the whole-turn backup is used, checked the same way |
| F5 Cost | A three-view run takes at most 4 GPU-hours |

## Decision rule

The decision uses the point estimates of Δ on `x0` and on `rev`. The 80% intervals and the values per round are reported next to them as a check, not as a test.

| Outcome | Rule | Decision |
| --- | --- | --- |
| Green: beats B0 | Δ on `x0` ≥ +0.03 | Go |
| Green: robust without words | −0.04 < Δ on `x0` < +0.03, and Δ on `rev` ≥ +0.03 | Go. The thesis leads with robustness. |
| Amber: no visible effect | −0.04 < Δ on `x0` < +0.03, and Δ on `rev` < +0.03 | Decide with the supervisor, using the diagnostics |
| Red: worse on normal audio | Δ on `x0` ≤ −0.04 | No-go for the recipe as it stands |

## Allowed setting changes

Two changes are allowed. Each comes from training logs only and is made once, before the remaining runs start. It applies to all arms, is dated in the settings change log below and is sent to the supervisor in one line.

- **Gain-alive rule.** One CPG run (round 1, fold 1) runs first. If the hinge is active on fewer than 20% of bags in epochs 2 and 3, m becomes 0.5. If it is active on more than 80%, m becomes 0.1.
- **Four-epoch rule.** If the training loss is still clearly falling in epoch 3, every arm trains 4 epochs, and B0's round 1 is rerun with 4 epochs.

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
