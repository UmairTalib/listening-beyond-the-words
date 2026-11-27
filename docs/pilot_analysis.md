# Pilot analysis

This page fixes how the pilot's test scores are analysed. It fills only the points that the [pilot protocol](pilot_protocol.md) leaves open and changes none of its rules; where the two seem to differ, the protocol holds. It is committed before any AUROC of any arm is computed, so the commit date shows that it was fixed in advance. It comes later than the protocol aimed for: the protocol wanted every setting written before any test score existed (section "Encoder caches and zero-shot scores", bullet "Zero-shot scores"), and the zero-shot and first trained scores exist since 2026-11-26. Every number seen so far from these scores needs no label (counts, means, spreads, percentiles), so the rule that still holds is the protocol's: nothing is tuned after results that use the labels (section "Clip versions", bullet "Reading the rules"). No rule on this page changes once the first AUROC is computed (section "Order").

## The main numbers
- **Scores.** A participant's score on a view is the mean z over all their segments of that view, in the scores of the run whose test fold holds them; z = logit("Yes") − logit("No"). Scores of different fold models are never pooled or averaged, across folds or rounds, because each model has its own scale.
- **AUROC.** For each arm, view and round, the AUROC is computed in each of the five test folds, with ties counted as one half, and the five are averaged; the two rounds are then averaged, so each arm's number is the mean of 10 per-fold AUROCs. The zero-shot arm is read the same way, fold by fold in both rounds, on `x0` and `rev`.
- **Differences.** Δ = CPG − B0 is taken fold by fold on the same participants and averaged over the 10 fold-runs. B1 − B0 and CPG − B1 are computed the same way.
- **Exact values.** Participant scores and fold AUROCs are computed as exact fractions, so a value that lies on a bound of a rule is never a rounding artefact. Floating-point numbers are used only inside the bootstrap and for the last square root of a Spearman correlation, which no rule reads.
- **A fold with equal scores.** If every participant of a fold has the same score, the fold's AUROC is 1/2 and its Spearman correlation is not defined; a mean or interval that needs it reads "not defined" and names the fold.

## Intervals
- **Draws.** Participants are drawn with replacement within each PHQ-8 class over all 189: 57 positive and 132 negative draws in each of 10,000 resamples, from seed 109 (the stream `item_seed(109, "evaluate", "bootstrap")`, a seed no other step uses). A participant drawn k times counts k times in their test fold of each round. One resample serves both rounds, every arm and both views, so every difference is paired. A resample that leaves any of the 10 test folds without a positive or a negative participant is drawn again; the redraws are counted and reported. The resamples need only the labels and the partitions, so they are drawn before any score is read, and the record keeps their SHA-256.
- **Recomputing.** On each resample, every fold's AUROC and Spearman correlation is computed again on the participants drawn into that fold; a difference is taken fold by fold, then averaged, as for the point estimate.
- **Intervals.** Percentile intervals with NumPy's linear percentiles, as the word check uses (section "Clip versions", bullet "The `x0` control"): the 80% interval runs from the 10th to the 90th percentile of the 10,000 values, the 95% interval from the 2.5th to the 97.5th. Every mean over fold-runs gets both: each arm's AUROC on each view, the three differences, each arm's mean Spearman correlation with the PHQ-8 totals, and the agreement of B0's and CPG's scores. Each difference also gets its bootstrap standard error, the standard deviation (ddof 1) of its 10,000 values. Per-round and per-fold values are point estimates only.
- **What they cover.** The intervals cover the sampling of participants, not training noise; the two rounds' separate values show the training noise.

## Checks first
All five feasibility checks must pass before the main numbers are read (protocol, section "Feasibility checks"). The analysis job keeps to this order:
1. The checks that read no score. F1 from the record of every kept run; F2 from the record of every CPG run, with the pooled share: the summed active bags over the summed bags of the kept CPG runs, not a mean of shares; F4 from the pinned word-check run; F5 as read on the first CPG run of round 1, fold 1 (section "Training", bullet "Check F5"). A record of F1, F4 or F5 that does not say that its check passed, or any record that lacks its key, is refused, never read as a fail. F2 is known from the training logs before any AUROC exists; if it fails, the researcher decides what follows then (section "Feasibility checks", F2), before any AUROC.
2. The zero-shot and B0 scores: their own numbers and check F3, B0's mean AUROC on `x0` of at least 0.60.
3. Only when all five checks pass are B1's and CPG's scores opened, and Δ, the other differences and the outcome computed.

A failed F2 or F3 writes the checks and the zero-shot and B0 numbers, prints "F<n> FAIL: reported as a finding; the rule is not read" and stops: no Δ and no outcome are computed.

## Decision rule
- The rule is the protocol's (section "Decision rule"), read on the point estimates of Δ on `x0` and `rev`. Values are compared exactly, unrounded, and printed with 4 decimals; a value that rounds to a bound is also printed unrounded. The same reading applies to F3. The 80% intervals and the per-round values are printed beside Δ, as a check, never as a test.
- The job prints the outcome and the protocol's decision for it, word for word.
- **Where a gain comes from.** The study design, which is not in this repository, adds one line to the outcome "Green: beats B0": if B1 − B0 also reaches +0.03 and CPG − B1 stays below +0.02, both on `x0`, the gain comes mainly from the extra clips. It is read only in that outcome, and it changes the framing, not the decision. In any other outcome the job prints "framing: not read", and without B1 "not available (cut: no B1)".

## Also reported
- Per round and per fold: every arm's AUROC and the three differences, on `x0` and `rev`.
- Per test fold: the Spearman correlation of each arm's participant scores (zero-shot included) with the PHQ-8 totals, on both views, and its mean over the fold-runs. Tied values share their average rank.
- Numbers for planning the main study: B0's AUROC on `x0`; Δ's bootstrap standard error; the difference between the two rounds' Δ; the Spearman correlation of B0's and CPG's participant scores within each fold, and its mean.

## Runs and pins
- **Kept runs.** A run counts if its record says done, it was not started with `--allow-dirty`, it passed F1, it used the pinned labels and partitions, and it came from commit 271cce6 (only B0's round 1 and the first CPG run of round 1, fold 1) or from the newest `v0.1.x` tag of its checkout, as its record shows. The timing run never counts. If two done runs share a fold-run, the earlier one counts and the other is reported.
- **Pins.** After training, `configs/pilot/evaluate.yaml` pins each kept run by its folder and the SHA-256 of its scores and of its record; the job checks every pin before it reads anything.
- **Versions.** The pinned set must be one of the protocol's versions (section "If time runs short"): all 30 runs; 25 (without B1's round 2); 20 (without B1); or 10 (B0 and CPG, round 1 only). A cut is made only when time runs short, decided and dated before the pins. Without B1's round 2, B1's numbers and its differences use round 1's five fold-runs, and CPG − B0 on the same five fold-runs is shown beside them. In a cut version, a value that cannot exist reads "not available (cut: <version>)", never blank.

## Order
1. This page and the analysis settings are committed before any AUROC is computed.
2. The code is tested on synthetic data only.
3. A trial, which the study design allows, computes the zero-shot numbers over the 10 fold-runs and B0's over round 1's five, with no difference and no rule, and runs the checks that read no score; these numbers count as seen. A fault it finds is fixed with a test and no rule change.
4. After all runs end and their checks are read, the full analysis runs once, from the release that holds its code (`v0.1.x`; the job refuses any other checkout).
5. No rule on this page changes after the trial. A later fault, meaning code that does not do what this page says, shown by a synthetic test, is fixed and released as the next `v0.1.x`; the job then runs again into a new folder, both records stay, the rerun's numbers replace the first ones whichever way they move, and the result line names the fix. A flaw in a rule itself is reported as a limitation: the reading under the frozen rule stays the result, and any other reading appears only beside it, labelled post hoc.

## Data protection
The job runs on OMNI, on a CPU. Participant scores, labels and PHQ-8 totals stay there; the job checks that every label equals "PHQ-8 total of 10 or more". It writes only aggregate tables (per-fold AUROCs, means, intervals, counts) and its record; the table writer refuses any column that identifies a participant or a segment. The tables are copied into `reports/pilot/` and committed.

## Not part of this analysis
- Main-study methods: DeLong tests, AUROCs pooled over folds, calibration measures, Holm corrections and one-sided tests.
- Later steps: the pooled-encoder probe, the summary of the gain term's logs over the CPG runs, the main study's power simulated again with the pilot's numbers, and the pilot report.
- A zero-shot AUROC on `x1`, which is not a test view.
