# Documentation

These pages explain how the project runs and how its results are checked. More pages (setup, data, pipeline and method) arrive together with their code.

| Page | Read it to |
| --- | --- |
| [Pilot protocol](pilot_protocol.md) | see how the pilot runs and how its result is read |
| [Engineering conventions](engineering_conventions.md) | write code, configs, jobs, commits and docs in the project's style |

The thesis gives the motivation, the related work and the complete study design.

## Glossary

| Term | Meaning |
| --- | --- |
| Adapter | The small network in Voxtral that maps the audio encoder's output into the language model's input space. It trains at a low learning rate. |
| Arm | One recipe that is compared: zero-shot (no training), B0, B1 or CPG. |
| ASR | Automatic speech recognition: software that turns speech into text. |
| AUROC | The chance that a random PHQ-8-positive participant scores higher than a random PHQ-8-negative one. 0.5 is chance. |
| B0 | Standard LoRA fine-tuning on the original audio (`x0`). |
| B1 | Fine-tuning on `x0`, `x1` and `x2` with cross-entropy only (gain weight 0). |
| Bag | Four segments of one participant. The gain term is computed per bag. |
| Cache | The frozen encoder's output, saved once per segment and view. |
| CPG | Counterfactual paralinguistic gain: B1 plus the gain term. "Paralinguistic" means everything in speech beyond the words, such as pitch, timing, loudness and voice quality. |
| CV | Cross-validation: train on some folds, test on the one left out, and repeat. |
| eGeMAPS | A standard set of 88 voice measures, such as pitch, loudness and their variability, computed with opensmile. The checks of `x1` use them. |
| Fold, round, run | A fold is one fifth of the participants. A round is one partition into five folds. A run (or fold-run) trains on four folds and tests on the fifth. |
| Gain (G_b) | How much more correct the model is on the real audio than on the words alone, within one bag. |
| Hinge active | G_b is below the margin m, so the gain term pushes in this bag. |
| ICC | Intraclass correlation. The checks of `x1` use ICC(C,1), the Shrout–Fleiss ICC(3,1): how closely a voice measure on `x1` follows the same measure on `x0`, segment by segment. 1 means it follows exactly, apart from a constant shift; 0 or less means not at all. |
| Leakage AUROC | How well a text classifier, trained on other participants' words, predicts the label from the ASR output of a view. 0.5 is chance. |
| Logit | The model's raw score for a token before it is turned into a probability. |
| LoRA | Low-rank adaptation: small trainable matrices added to the frozen weight matrices of the language model. |
| Muffled clip | `x2`: the original clip through a 400 Hz low-pass filter, part by part. Pitch, loudness and timing stay; most words become unclear. |
| Original clip | `x0`: the parts of a segment, cut from the session recording at their word times and joined by 300 ms of silence. |
| Participant | One interviewed person, with one interview session. Some pages say person. |
| Participant score | The mean of z over all of a participant's segments. |
| Part | A run of a segment's words from one transcript line with nothing cut out between them, and their audio. In a clip, parts are joined by 300 ms of silence. |
| PHQ-8 | An eight-item depression questionnaire. A total of 10 or more is the positive label here. |
| Re-voiced clip | `x1`: the segment's words read part by part by Kokoro-82M, a text-to-speech model, in a synthetic voice of the participant's gender. The words stay; the participant's voice is gone. |
| Reversed clip | `rev`: like the original clip, but each part is played backwards in 250 ms windows before it gets its fades, so that the words are removed (check F4). Pitch and voice quality stay. Used for testing only. |
| Segment | Exactly 40 consecutive words of one participant, with their audio. |
| Stop-gradient | A value that the loss uses but that training does not change through this term. |
| Study | A set of runs with one protocol: the pilot now, the main study later. |
| Token | A unit of text that the language model reads or writes. "Yes" and "No" must each be one token. |
| Turn | One row of a DAIC-WOZ transcript with valid times: the interviewer or the participant speaking. Blank rows and sync markers are not turns. |
| View | One version of a segment's audio: `x0` original, `x1` re-voiced, `x2` muffled, `rev` reversed. Also called a clip version. |
| WER | Word error rate: the share of words that speech recognition gets wrong. |
| Word check | The check of the views on five random segments per participant. Two ASR systems transcribe each of these segments in every view. `rev` must lose the words (check F4), `x0` is the control, and `x1` must keep the words but lose the participant's voice. |
| Workspace | The storage folder on OMNI that holds all data, caches and runs. It expires unless extended. |
| z | A segment's score: logit("Yes") − logit("No"). |
