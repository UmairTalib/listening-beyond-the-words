# Listening Beyond the Words

**Counterfactual training for depression detection with audio LLMs**

Master's thesis project, M.Sc. Computer Science, University of Siegen.

Author: Umair Talib · Supervisor: Ugonna Oleh

[![CI](https://github.com/UmairTalib/listening-beyond-the-words/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/UmairTalib/listening-beyond-the-words/actions/workflows/ci.yml)

> **Status (November 2026):** pilot study analysed. All 30 training runs finished and all five feasibility checks pass, and the analysis fixed in advance gives the outcome "Green: beats B0" (decision: go). The pilot's diagnostics and report come next, then the main study.

---

## The problem

Audio large language models (audio LLMs) can listen to an interview and answer questions such as *"does this speaker show signs of depression?"*. In practice they lean heavily on **what** is said and largely ignore **how** it is said: the voice, timing, pauses and prosody that clinicians also listen to. As a result they fail when the words are missing, uninformative or misleading.

## The idea

This project adds a new training objective, the **counterfactual paralinguistic gain (CPG)**, to standard LoRA fine-tuning. For each person the model sees three versions of the same speech:

| View | What it contains |
| --- | --- |
| `x0` original | words **and** voice |
| `x1` re-voiced | the same words read by a neutral synthetic voice (words only) |
| `x2` muffled | a 400 Hz low-pass of the original (mostly voice) |

The gain term rewards the model when its score on the real audio is more correct than its score on the words-only copy, so it is pushed to use evidence that only the voice carries:

$$
\mathcal{L} = \mathrm{CE}(x_0) + \lambda_L\,\mathrm{CE}(x_1) + \lambda_P\,\mathrm{CE}(x_2) + \lambda_G\, s\log\!\left(1 + e^{(m - G_b)/s}\right),
\qquad
G_b = (2y-1)\,\hat z_0 - \max\!\left(0,\ (2y-1)\,\hat z_1\right)
$$

where $\hat z$ are standardised per-bag scores (running statistics with stop-gradient) and $\hat z_1$ is a fixed, stop-gradient reference.

## Approach

- **Model:** [Voxtral Mini 3B](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507) with LoRA; frozen audio encoder, cached once per view.
- **Data:** DAIC-WOZ / E-DAIC clinical interviews (licensed; not included in this repository).
- **Baselines:** standard fine-tuning (B0), the same three views without the gain term (B1), and classical voice and text baselines.
- **Evaluation:** person-level AUROC with cross-validation by person; tests on normal audio and on audio where the words are removed or misleading.
- **Compute:** University of Siegen OMNI cluster (NVIDIA V100).

## Roadmap

- [ ] Pilot study: can CPG beat standard fine-tuning? (B0 vs B1 vs CPG, 2 × 5-fold CV)
- [ ] Main study: unseen word masks, conflict set, controlled shortcut test
- [ ] External test on held-out E-DAIC sessions
- [ ] Thesis and paper

## Results

**Pilot study** (189 E-DAIC participants, 2 rounds × 5 folds split by person; rules fixed in advance in [docs/pilot_protocol.md](docs/pilot_protocol.md) and [docs/pilot_analysis.md](docs/pilot_analysis.md); tables in [`reports/pilot/`](reports/pilot/)):

| Arm | AUROC, original audio (`x0`) | AUROC, words removed (`rev`) |
| --- | --- | --- |
| Zero-shot Voxtral | 0.714 | 0.516 |
| B0: standard LoRA fine-tuning | 0.657 | 0.541 |
| B1: B0 with the re-voiced and muffled views | 0.648 | 0.551 |
| CPG: B1 with the counterfactual paralinguistic gain | 0.705 | 0.550 |

Each value is the mean of 10 per-fold AUROCs. CPG − B0 is +0.047 on the original audio and +0.009 with the words removed. The rule reads the point estimate on the original audio: at least +0.03 gives **Green: beats B0**. The 95% intervals (+0.018 to +0.079 and −0.032 to +0.049) cover the sampling of participants, not training noise, and are a check, not a test. The gain over B0 does not come mainly from the extra views: B1 − B0 is −0.010 on the original audio, while CPG − B1 is +0.057. On the original audio CPG (0.705) stays within 0.01 of zero-shot Voxtral (0.714), while B0 and B1 lie 0.056 and 0.066 below it (point estimates; no interval is computed for differences from the zero-shot arm); with the words removed every arm is close to chance. The pilot decides whether to go ahead; the main study, on new partitions, tests the claim.

## Repository layout

```text
src/listening_beyond_words/   library code: data pipeline, clip versions, encoder caches, the CPG loss, training, the analysis of the test scores and the pilot's diagnostics
tests/                        unit tests on synthetic data
configs/                      settings and pins: data.yaml, models.yaml, the shared training settings, one config per arm, the analysis settings (evaluate.yaml) and those of the diagnostics (probe.yaml, power.yaml)
scripts/                      one Python script per pipeline step and shell scripts for OMNI; scripts/slurm/ holds the Slurm job files
docs/                         pilot protocol, pilot analysis, pilot diagnostics, engineering conventions and glossary
```

## Development

The environment is locked with [uv](https://docs.astral.sh/uv/) and uses Python 3.12. On a laptop:

```bash
make env      # create .venv with the CPU build of PyTorch
make hooks    # install the git hooks
make check    # format check, lint, type check and tests (CI also runs the git hooks)
```

On the OMNI cluster, `scripts/setup_omni_env.sh` builds the GPU environment in the workspace. The rules for code, configs, data, Slurm jobs, commits and releases are in [`docs/engineering_conventions.md`](docs/engineering_conventions.md).

## Data and ethics

The interview data are distributed by USC ICT under a research licence and contain sensitive health information. **No audio, transcripts, derived clips, features or per-person scores are stored in this repository**; only code and aggregate results. This is a research project, not a diagnostic tool.

## Related publication

An earlier paper on the same corpus measured how much interviewer speech ends up in participant transcripts, the problem that the interviewer filter and the cut by the interviewer's turns in this pipeline address:

U. Oleh, U. Talib, M. A. Alam, and R. Obermaisser, "Leakage-Resilient Diarization and Feature Extraction Framework for Robust Multimodal Mental Health Corpora," in *Proc. 12th Int. Conf. Control, Decision and Information Technologies (CoDIT)*, Bari, Italy, Jul. 2026, pp. 2508–2512, doi: [10.1109/CoDIT70676.2026.11630977](https://doi.org/10.1109/CoDIT70676.2026.11630977).

## Documentation

See [`docs/README.md`](docs/README.md) for the pilot protocol, the pilot analysis, the pilot diagnostics, the engineering conventions and a glossary.

## Licence

© 2026 Umair Talib. All rights reserved for now; an open-source licence will follow after the thesis is graded. Nothing here licenses the DAIC-WOZ or E-DAIC data: USC ICT grants access to them under its own agreement.
