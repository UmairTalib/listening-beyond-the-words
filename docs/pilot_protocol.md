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
| Compute | About 100 V100 GPU-hours; the pilot runs once, on speech with the interviewer cut out (section "Interviewer detector") |

## Speech used

Fixed on 12 November 2026, before any training. The settings are in `configs/pilot/data.yaml`.

- **Transcripts.** The pilot takes its words from the E-DAIC transcripts, because the DAIC-WOZ human transcripts had not arrived when the segments were built. They come from automatic speech recognition, hold the participant's lines and contain almost no fillers such as "um" and "uh". The words-only view `x1` therefore has no fillers, while `x0` keeps them. This is reported as a limitation. The DAIC-WOZ transcripts, which arrived on 2026-11-18, are used only for the times of the interviewer's and the participant's turns, never for their words (section "Interviewer detector").
- **Labels.** A participant is positive if the PHQ-8 total in the official split files is 10 or more. For every pilot participant this total equals the sum of the eight item scores. 57 of the 189 participants are positive. The binary column disagrees for 15 participants and is not used. A third label file gives a different total for 2 participants, which changes no label. Gender comes from the split files (87 female, 102 male); `detailed_lables.csv` codes it differently for 3 participants.
- **Interviewer filter.** The interviewer is mostly inaudible in these recordings, but leaks into some participant lines, and an assistant can be heard at the start of a few sessions. A transcript line is dropped if at least half of its speech, cut into 1.5 s windows, sounds more like the interviewer than like the participant to a speaker-verification model (WavLM x-vectors, pinned in `configs/pilot/models.yaml`), or if it contains one of 70 lines of her script in her own wording (10 until 2026-11-12, when 1.6% of the segments were found to still hold her questions). A line that lasts at least 1.5 s but holds no 0.75 s of continuously detected speech is dropped as well, because neither the model nor a listener has judged it; such lines hold 1.9% of the words and are balanced between the PHQ-8 classes. In a blind listening check of 35 lines, another voice was heard in an estimated 3.6% of all lines before the filter and in about 1.3% after it. The rest is reported as a limitation.
- **Word times.** A line of 30 s or more with fewer than one word per second has broken timestamps (a few words spread over minutes) and is dropped. The words of every kept line are then placed on the audio by forced alignment with wav2vec2-large-960h-lv60-self (pinned in `configs/pilot/models.yaml`); digits are spelled out for the aligner only. A line that cannot be aligned is dropped and counted.
- **Cut.** The original clip `x0` of a segment is cut from the session recording at its word times. Each part of the segment (its words from one line) keeps up to 0.1 s of audio before its first word and after its last word, because the aligner's word times end a little early. The margin never reaches more than halfway to the nearest other kept participant word, which is often less than 0.1 s away in the same line, and never past the recording. On all 6,207 segments, no margin passes the halfway point to a dropped line. With the aligner's padding, a part can still reach up to 0.3 s past the times of its transcript line, into audio that the interviewer filter did not judge (35 minutes over all segments); the listening check hears it. Each part fades in and out over 10 ms, so that the joins do not click, and the parts are joined by exactly 300 ms of silence. With full margins, the last word of one part and the first word of the next are therefore 0.5 s apart. The margin and the fade are set in `configs/pilot/data.yaml`.
- **Checks before training.** The share of dropped words must not differ between PHQ-8-positive and PHQ-8-negative participants (two-sided Mann–Whitney test, p ≥ 0.05); otherwise the filter is reviewed before any training. Twenty segments, drawn at random from all segments (seed 101), are cut as original clips and listened to before training, in random order and without their labels. Each clip is scaled to a common loudness and played over headphones at a fixed, clearly audible volume. Another voice means any sound made by anyone but the participant, speech or not: the interviewer, an assistant, background speech, a laugh or a "mhm", however faint. A clip that is still unclear after a second listen counts as another voice, and the listener notes whether it is heard at the start or end of a part or inside the speech. If another voice is heard in more than two of the 20 clips, the interviewer filter and the cut are reviewed. Result on 2026-11-12: another voice in 4 of the 20 clips (once at the edge of a part, three times inside the speech), so the filter and the cut are under review before any training.

### Interviewer detector

Pre-registered on 2026-11-13, before any run. Its settings are in `configs/pilot/data.yaml` (`ellie_detector`).

The speaker-verification filter misses the interviewer when she is short, faint or speaks in a pause of the participant. A test that scored each short stretch of speech on its own found her in only 2 of the 4 clips where she was heard and flagged 4 of the 16 clean clips, so it failed the rule set before its run. A dedicated detector is therefore added. It rests on one fact: the interviewer's lines are fixed recordings played in every session, so a sound that recurs almost identically in other sessions is her, while a participant's speech never recurs. Short sound events in the gaps between lines and in the pauses inside lines are matched across sessions, and those that recur in at least three sessions become templates of her clips. Every session is then scanned with the templates. A hit needs a match score above a threshold that is set only from decoys (the templates shifted up and down by 3 semitones) and a level that fits the interviewer's loudness in that session. A hit marks the template's full length as a zone. The detector uses no feature of the participant's voice.

Zones are applied to the speech in three ways. A line is dropped if zones cover at least half of it. A word that comes within 50 ms of a zone is dropped. A line is split where a zone lies between two kept words, so that the two pieces are joined by the 300 ms silence like two turns; the margins of the cut stop 20 ms before a zone. A rule for an assistant's voice before the interview (lines that end before the first zone of at least 1.5 s within the first two minutes are dropped) is used only if it drops at most 0.5% of the kept words.

The gates run in this order, and a failed gate stops the chain.

| Gate | Passes if |
| --- | --- |
| P1 Recurrence | The same interviewer phrase in two sessions is told apart from the same common word said by two participants with an AUROC of at least 0.95, and at least half of the same-phrase pairs score above the 99.9th percentile of the word pairs |
| P2 Library | At least 80 templates, after at most one rerun with a lower event threshold |
| D1 Known lines | At least 80% of the 266 lines that consist mostly of interviewer phrases are at least half covered by zones |
| D2 Decoys | Some threshold in the grid keeps the decoys' cost to at most 0.5% of the kept words on average and 0.75% for each decoy family; the smallest such threshold is used |
| D3 Planted copies | Copies of her clips from other sessions, planted at her level, are found in at least 90% of pauses, 90% of turn edges and 50% of partial overlaps |
| D4 Cost | At most 1% of common participant words touch a zone, and the zones remove at most 3% of the kept words |
| V4 Precision | In a blind set of 20 detections mixed with 20 controls, 2 s each, another voice is heard in at least 12 detections and at most 2 controls |
| V5 First clips | Zones meet at least 3 of the 4 clips where the interviewer was heard and at most 3 of the 16 clean clips |
| V6 Balance | After the rebuild, the dropped share of words does not differ between PHQ-8 classes (two-sided Mann–Whitney test, p ≥ 0.05) |
| V7 New clips | In 20 new segments (seed 102, none overlapping the first 20 clips), another voice in at most 2 |

Exactly one fallback follows a failure. If P1 or P2 fails, templates are built within each session instead of across sessions. If a later gate fails with the misses mostly over words or quiet, a small learned frame detector on frozen WavLM states is added on top of the zones, trained on the zones and on her clips planted over words; its threshold is the largest score that finds at least 70% of planted copies over words while dropping at most 1% more of the kept words. The fallback repeats the gates once, and V7 then uses seed 104. If V6 fails while the decoys' cost is balanced between the classes, the difference is real interviewer speech, and the researcher decides the case before training. If the fallback fails too, the researcher decides whether the pilot waits for the DAIC-WOZ transcripts with the interviewer's own times or runs with the leak reported as a limitation. No setting is ever tuned on a listened clip, and no rule is loosened.

Result on 2026-11-13, gate P: P1 failed. The same interviewer phrase in two sessions scored far above unrelated pairs (median 0.22 against 0.03 for different phrases), but not reliably above the same word said by two participants: AUROC 0.929 for the fine-structure variant (0.857 for the whitened spectrum), and only 23% of same-phrase pairs above the null's 99.9th percentile (40%). P2 passed with 587 templates, but the chain stops at P1.

Decision on 2026-11-13, before any training. The fallback is not run: none of the 266 known interviewer lines has its phrase transcribed three or more times in its own session, so a library built within sessions could not meet D1. The pilot therefore runs twice, with the same settings.

- **First run, preliminary.** It uses the speech rebuilt with the 70 scripted lines dropped. The interviewer can still be heard in some segments (in 4 of the 20 listened clips); this is reported as a limitation.
- **Second run, the main result.** When the DAIC-WOZ transcripts with the interviewer's own times arrive, every stretch where she speaks is cut out and the speech is rebuilt. A new blind listening check of 20 segments (seed 102, none overlapping the first 20 clips) must find another voice in at most 2. Then all runs are repeated.
- **Fixed now.** The decision rule is read on the second run; the first run only measures how much the leak changed the result. Settings fixed during the first run (the margin m and the number of epochs) carry over unchanged, with no new tuning.
- **Deadline.** If the transcripts have not arrived by 2026-11-27, an interviewer finder that re-transcribes the audio with the pinned Whisper model is tried first, with its own rules fixed before it runs.

Decision on 2026-11-18, before any training. The DAIC-WOZ human transcripts arrived, so the pilot runs once, on the speech with the interviewer cut out; this replaces the two runs above. The preliminary run on the current speech and the Whisper finder are dropped.

- **Source.** The transcripts come from the official DAIC-WOZ release of USC ICT. Each is checked against the CRC-32 in its archive, and every recording in the release is byte-identical to the E-DAIC recording of the same session, so the transcript times fit the E-DAIC audio exactly as they fit the DAIC-WOZ audio; four sessions whose times are offset from their recording are shifted by the cut rule below. The official archive of session 440 is cut off on the USC server, so its transcript comes from a public copy whose other 188 transcripts are byte-identical to the official files; seven further copies agree with it.
- **Cut.** Every stretch where the interviewer speaks is cut out by her times in these transcripts. Sessions 451, 458 and 480 have no interviewer turns in the release; for them only the words inside the participant's own turns are kept. The exact rule, including how known timing offsets and interruptions are handled, is fixed on this page before the speech is rebuilt.
- **Check.** A new blind listening check of 20 segments (seed 102, none overlapping the first 20 clips) must find another voice in at most 2 before any training.

Cut rule, fixed on 2026-11-18 before the speech is rebuilt. Its settings are in `configs/pilot/data.yaml` (`daicwoz_cut`). The interviewer filter of the section "Speech used" stays as it is: the words of the lines it keeps are aligned again, and then this rule decides which of them remain. A turn is one row of a DAIC-WOZ transcript with valid times; rows that are blank or hold only a sync marker are not turns.

- **Participant's turns.** A word is kept only if its middle lies inside one of the participant's turns and it does not come within 200 ms of one of the interviewer's turns. A word from s to e comes within 200 ms of a turn from a to b if s < b + 200 ms and e > a − 200 ms. Keeping only the participant's turns, instead of only removing the interviewer's, also drops words in stretches that no participant turn covers. The guard is wider than the interviewer detector's 50 ms, because the detector's zones already reached 150 ms past the sound they matched, while the transcript's turns have no such padding. Sessions 451, 458 and 480 have no interviewer turns, so only the participant's turns decide for them.
- **Interview only.** A word that begins before the interviewer's first turn starts or ends after her last turn ends is dropped. Before her first turn the participant may talk with the research staff, and an assistant can be heard at the start of a few sessions; after her last turn the interview is over. On the word times of 2026-11-12, this drops 1 word before her first turn and 132 words after her last turn (in 70 sessions) that the other conditions keep. The rule does not apply to sessions 451, 458 and 480.
- **Offsets.** In sessions 318, 321, 341 and 362 the transcript times are offset from the recording. Every time in these four transcripts is shifted by the offset that the timing check found (`scripts/check_daicwoz_timing.py`: the shift, in 0.1 s steps from −90 s to +90 s, that puts the most word middles inside the participant's turns): +34.1 s, +3.5 s, +5.9 s and +16.7 s. With these, 95.7–99.1% of the aligned words fall inside the participant's turns, close to the median of 97.5% over all sessions. The public preprocessing code [daic_woz_process](https://github.com/adbailey1/daic_woz_process/blob/8b5f8ff5c510df9904e0750f1a806c29e649c5cf/config_files/config_process.py), which does not say how its offsets were found, adds +34.32, +3.84, +6.19 and +16.86 s instead (82.0–93.3%). The shifts that put the fewest words inside the interviewer's turns (+34.48, +3.72, +6.24 and +15.70 s) are not used, because the real overlaps between the two speakers, which the transcripts mark, also move them. The public offsets lie within 0.34 s of ours, and those shifts within 0.38 s in three of the four sessions, so the guard in these four sessions is 400 ms. No other session has a best shift of more than 0.5 s in either direction, and no other session is shifted.
- **Interruptions.** The DAIC-WOZ documentation names two interruptions in which another person enters the room: session 373 from about 5:52 to 7:00 and session 444 from about 4:46 to 6:27. These times are approximate, so each interruption is widened by 10 s on each side and treated like an interviewer turn. The widened windows also contain the windows in the public code above (373: 6:35–7:08; 444: 4:46–6:27).
- **Parts and margins.** A part of a segment is a run of its kept words from one transcript line with no dropped word and no interviewer turn between them; parts are joined by the 300 ms silence as before. The audio between two kept words of one part stays in the clip, so the participant's pauses are kept. In sessions 451, 458 and 480, whose interviewer turns are missing, a part also ends where one participant turn ends and the next begins, so that no audio from a gap between two participant turns is used there. For the same reason, a margin there also stops at the edge of the participant's turn, or at the edge of a kept word that runs past it (added on 2026-11-18, before the listening check, because the first build kept 10.5 s of margin audio in these gaps). A margin stops 20 ms short of an interviewer turn on either side and never reaches more than halfway to a dropped word. Everything else in the cut stays as in the section "Speech used".
- **Cost.** On the word times of 2026-11-12, which were aligned while the interviewer filter still used 10 of her lines, the rule drops 4.77% of the aligned words, with no significant difference between the PHQ-8 classes (two-sided Mann–Whitney test, p = 0.75) or the genders (p = 0.27). Every participant keeps at least 129 words; one keeps fewer than 160 (4 segments, one bag) and 38 keep fewer than 640 (16 segments). How their segments are used is fixed in the row "Segments and bags" of the fixed settings, before training.
- **Checks.** On the rebuilt speech, the balance test of the section "Speech used" is repeated on the share of each participant's transcript words that are dropped in all (by the filter, the word times and this rule), and the number of parts per segment is reported for each PHQ-8 class. The new listening check (seed 102) replaces the first one, which is not repeated.
- **If a check fails.** If the new listening check hears another voice in more than 2 of the 20 clips, the guard is widened once to 500 ms in every session, the speech is rebuilt, and 20 new segments (seed 104, none overlapping the 40 clips heard before) are checked the same way. If that check fails too, the researcher decides before any training whether the pilot runs with the leak reported as a limitation. If the dropped share differs between the PHQ-8 classes (p < 0.05), the rule is kept and the difference is reported by cause: the interviewer's turns, stretches outside the participant's turns, the interview's limits and the interruptions. No setting of this rule is tuned on a listened clip, and no rule is loosened.

Result on 2026-11-18, on the speech rebuilt with this rule (250,016 words aligned from the selection with 70 of her lines): the rule drops 4.48% of the aligned words (by her turns 6,970 words, outside the participant's turns 3,920, at the interview's limits 133, by the interruptions 183). The share of each participant's transcript words dropped in all does not differ between the PHQ-8 classes (two-sided Mann–Whitney test, p = 0.50) or the genders (p = 0.98), so the balance check passes. A segment has a mean of 4.06 parts for negative and 4.24 for positive participants (means per participant, p = 0.28). All 189 participants keep segments, 5,883 in all (3 to 103 each); one has fewer than 4 and 39 have fewer than 16. One x0 clip is longer than 30 s (37.4 s). The new listening check (seed 102, 20 clips from 19 sessions, none sharing audio with the clips heard before) found another voice in none of the 20 clips, so it passes and the speech is used as built.

## Clip versions

Fixed on 2026-11-20, before any clip version is built. The settings are in `configs/pilot/data.yaml` (`views` and `word_check`). `x0` is cut as in the section "Speech used" and by the cut rule of 2026-11-18 (section "Interviewer detector"), and the other views are made from the same parts. The settings that only `x1` has, such as its voices and the text it reads, are added here and to `configs/pilot/data.yaml` before `x1` is built. The rules of the word check, including the checks of `x1`, are added before any clip is transcribed.

- **The part is the unit.** Every view is built part by part, and its parts are joined by exactly 300 ms of silence, as in `x0`. `x1` makes one Kokoro call per part, and `x2` filters each part of `x0` on its own. For `rev`, each part's audio, as cut for `x0` but before its fades, is played backwards in 250 ms windows that start at the part's first sample; a shorter last window is reversed too. The part then gets the same 10 ms fades as `x0`, so that it starts and ends at exactly 0. The backup of check F4 plays each whole part backwards, also before the fades. The study design, which is not in this repository, said "turn" for `x1`'s calls and for this backup, and so did check F4 below for the backup until 2026-11-20. A part is meant there, and a turn stays one row of a DAIC-WOZ transcript. With the part as the unit, every view has the same pieces and the same silent gaps, and no filter echo or reversed window reaches into a gap.
- **Re-voiced clip.** Fixed on 2026-11-22, before `x1` is built. Kokoro-82M (pinned in `configs/pilot/models.yaml`; kokoro 0.9.4, with misaki 0.9.4 as its text front end; US English, speed 1.0, evaluation mode, so no random dropout) reads each part's words in one of two voices of the participant's gender: af_heart or af_bella for women, am_michael or am_fenrir for men. Within each gender, participants are ordered by PHQ-8 class (negative first) and then by ID, and alternate between the two voices along that order, starting with af_heart and am_michael. Each voice thus gets half of the negative and half of the positive participants of its gender, give or take one, so that the voice is balanced against the label. The study design, which is not in this repository, assigned the voices by participant ID alone. The segments store each word in lower case, with apostrophes dropped and every other character that is not a letter or a digit turned into a space, so "I'm" is stored as "im" and "x-ray" as "x" and "ray". Kokoro reads the words as the E-DAIC transcript writes them instead: capitals and apostrophes are kept (a typographic apostrophe becomes a plain one), every other mark becomes a space, and Kokoro reads digits itself. A word as written is used only when all its stored words lie in the part; otherwise the part's stored words are used, for example when "x-ray" is split at a part edge. Turning the text back into stored words gives exactly the part's words. Words outside Kokoro's lexicon get their sounds from espeak-ng 1.52.0. Each part's call is seeded from seed 106, the segment and the part's position. Kokoro's 24 kHz output is resampled to 16 kHz with soxr. Each part is then trimmed at both ends where it is more than 40 dB below its peak, keeping 10 ms beyond the first and last louder sample, and gets the same 10 ms fades as `x0`, which fall in those 10 ms. The parts are joined by 300 ms of silence, and the clip's RMS, gaps included, is set to 0.05 before the peak guard (bullet "Levels"). Before any clip is rendered, a dry run of Kokoro's text front end on every part must give every word a sound; if it does not, only this text mapping may change, dated here. Then 20 segments outside the word-check sample (seed 106) are rendered, and the researcher listens to 10 of them. A listen can expose a fault in the code, which is fixed for all clips, but no setting of `x1` is tuned on a listened clip.
- **Muffled clip.** The filter of `x2` is an 8th-order Butterworth low-pass whose −3 dB point is at 400 Hz. It runs forwards and then backwards, so that nothing shifts in time; the two passes together are 6 dB down at 400 Hz. After the filter, each part gets the same 10 ms fades as `x0` again (added on 2026-11-22, before any clip version is built), because the filter spreads each part's edges by a few milliseconds: without them the muffled parts would no longer start and end at exactly 0, and each join with a silent gap would get a small step that can click. The clip's RMS is then set to that of `x0`.
- **Levels.** `x0` is never rescaled: it keeps the recording's level. `x1` is set to an RMS of 0.05 and `x2` to that of `x0`. If the peak of an `x1` or `x2` clip is then above 0.99, the whole clip is scaled down to a peak of 0.99, so that no sample reaches full scale. Such clips are counted, and they stay below their target RMS. `rev` only reorders the recording's own samples and fades each part like `x0`, so its peak never exceeds the recording's and it needs no such guard.
- **Files.** Every view is stored in the workspace as 16 kHz mono 16-bit FLAC, with a manifest that gives each file's length, RMS, peak and SHA-256 and the start and end sample of each part. Each built view is pinned in `configs/pilot/data.yaml` by the SHA-256 of its manifest.
- **Word-check sample.** The word check (check F4 and the checks of `x1`) transcribes five segments per participant, drawn at random with seed 105, or all of a participant's segments if there are fewer: at most 943 segments. The sample is drawn and pinned before any clip is transcribed.

Result on 2026-11-22 (job 12557091, 9.7 minutes): `x0`, `x2` and `rev` each hold 5,883 clips, 23.65 hours. Before anything was written, `x0` of the 20 clips of the listening check with seed 102 was rebuilt, and the listening copy of each one equalled the heard file, sample for sample. In every segment the three views have the same parts and silent gaps, every part starts and ends at exactly 0, and every file reads back as written. The peak guard scaled down 22 `x2` clips, from 9 sessions. 12 clips of `x0`, and the same 12 of `rev`, reach full scale, because their recordings do. One clip is longer than 30 s (37.4 s). The word-check sample holds 943 segments: five from each of 188 participants and all three of the participant who has three.

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
| Segments and bags | Exactly 40 words, 300 ms gaps. A bag is 4 segments of one participant; 4 bags per step (2 positive, 2 negative); class weights 1. In each epoch a participant gives as many whole bags as fit in at most 16 of their segments, drawn at random; segments left over wait for a later epoch, and no segment is repeated. A participant with fewer than 4 segments (one, with 3) gives no training bag but is scored when in a test fold (fixed on 2026-11-18, before training). |
| Views | `x0` original; `x1` re-voiced (Kokoro-82M, same gender, words only); `x2` muffled (400 Hz low-pass); `rev` for testing only; all built part by part (section "Clip versions") |
| Loss | CE(x0) + 0.25 CE(x1) + 0.5 CE(x2) + λG · gain term |
| Gain term | Margin m = 0.3, softness s = 0.1, λG = 1 ramped up over the first 20% of steps, computed in fp32 |
| Word check (F4 and the checks of `x1`) | faster-whisper large-v3 and wav2vec2-large-960h-lv60-self |
| Intervals | Participant bootstrap stratified by label, 10,000 resamples |

## Feasibility checks

All five checks must pass before the main numbers are read. A failed check is fixed and the affected step rerun. A failed check is not a result.

| Check | Passes if |
| --- | --- |
| F1 Runs finish | All 30 runs finish with no NaN or inf in the loss or the gradients |
| F2 Gain term active | In the CPG runs, the hinge is active on 20–80% of bags in epochs 2 and 3, after at most the one allowed change of m. If it is still outside 20–80%, this is reported as a finding and decided with the supervisor. |
| F3 B0 learns | B0's mean AUROC on `x0` is at least 0.60. Below that, the fallback for a B0 near chance (a linear readout head) is discussed with the supervisor before any effect is read |
| F4 Words removed | On `rev`, both ASR systems reach a WER of at least 0.80, and a text classifier on their output predicts the label with an AUROC of at most 0.60. Otherwise the whole-part backup (each whole part played backwards; "turn" until 2026-11-20, section "Clip versions") is used, checked the same way |
| F5 Cost | A three-view run takes at most 4 GPU-hours. If not, every arm uses 8 instead of 16 segments per participant per epoch, and B1's round 2 is cut first. |

## Decision rule

The decision uses the point estimates of Δ on `x0` and on `rev`. The 80% intervals and the values per round are reported next to them as a check, not as a test. The rule is read on the single run on the speech with the interviewer cut out (section "Interviewer detector").

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
