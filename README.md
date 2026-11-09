# Listening Beyond the Words

**Counterfactual training for depression detection with audio LLMs**

Master's thesis project, M.Sc. Computer Science, University of Siegen.
Author: Umair Talib · Supervisor: Ugonna Oleh

> **Status (November 2026):** method designed, pilot study in preparation. No results yet.

---

## The problem

Audio large language models (audio LLMs) can listen to an interview and answer questions such as *"does this speaker show signs of depression?"*. In practice they lean heavily on **what** is said and largely ignore **how** it is said: the voice, timing, pauses and prosody that clinicians also listen to. As a result they fail when the words are missing, uninformative or misleading.

## The idea

This project adds a new training objective, the **counterfactual prosody gain (CPG)**, to standard LoRA fine-tuning. For each person the model sees three versions of the same speech:

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
- [ ] Full study: unseen word masks, conflict set, controlled shortcut test
- [ ] External test on held-out E-DAIC sessions
- [ ] Thesis and paper

## Repository layout

```
src/listening_beyond_words/   library code (data pipeline, views, CPG loss, training, evaluation)
tests/                        unit tests (loss values, gradients, data splits)
configs/                      experiment configurations
scripts/slurm/                job scripts for the OMNI cluster
docs/                         methodology, pilot plan, proposal and reference scripts
```

## Data and ethics

The interview data are distributed by USC ICT under a research licence and contain sensitive health information. **No audio, transcripts, derived clips, features or per-person scores are stored in this repository**; only code and aggregate results. This is a research project, not a diagnostic tool.

## Documentation

See [`docs/README.md`](docs/README.md) for the methodology, the pilot plan and the analysis plan.

## License

To be decided.
