# Changelog

All notable changes to this project are recorded in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Changes to method settings are listed under **Changed**, with their date and the reason.

## [Unreleased]

### Added

- Engineering conventions in `docs/engineering_conventions.md`, enforced by Ruff, mypy, pytest, pre-commit hooks and a CI workflow.
- Documentation index with a glossary in `docs/README.md`.
- Pilot protocol in `docs/pilot_protocol.md`: the design, the five feasibility checks, the go / no-go rule and the settings change log.
- Locked Python 3.12 environment managed with uv: the CPU build of PyTorch for the Mac and CI, the CUDA 12.6 build with V100 kernels for OMNI.
- Settings module that reads every data path from `LBW_WORKSPACE`.
- Slurm helpers: a shared job setup and a submit script that keeps logs in the workspace.
- Script that sets up the GPU environment on OMNI and checks the PyTorch build.
- Download script that fetches E-DAIC straight into the OMNI workspace, checks file sizes and records SHA-256 checksums.
- Unpack job for the audio and transcripts of the 189 pilot sessions (300–492).
- Voxtral Mini 3B pinned to a Hugging Face commit in `configs/pilot/models.yaml`, a script that fetches pinned models on an OMNI login node, and a GPU smoke test job: the model loads in fp16, "Yes" and "No" are single tokens, and synthetic clips get finite scores.
- The 189 pilot session IDs with a guard that refuses every other session, an E-DAIC transcript reader, and a transcript check that prints aggregate numbers only: lines, words, coverage, timing, confidence, fillers, tags and known interviewer phrases.
- Interviewer filter: a pinned speaker-verification model (WavLM x-vectors) scores short speech windows inside the participant's lines against the interviewer's voice, with a GPU job that prints aggregate checks and prepares a blind listening set on OMNI.
- Labels of the pilot: PHQ-8 total of 10 or more from the official split files, checked against the item scores, keeping only the 189 pilot sessions; pinned with its SHA-256 in `configs/pilot/data.yaml`.
- Line selection of the pilot: the interviewer filter applied to every transcript line, with a check that the dropped share does not differ between PHQ-8 classes or genders.
- Pilot protocol section "Speech used": the transcripts, the labels and the interviewer filter, fixed before any training.
- Word times: lines with broken timestamps are dropped, and the words of every kept line are placed on the audio by forced alignment with a pinned wav2vec2 model (GPU job).
- Segments of exactly 40 consecutive participant words, with lines joined by 300 ms gaps.
- Partitions: 2 rounds of 5 folds (seeds 101 and 102), split by participant and stratified by label and gender.
- Cut of the original clip (`x0`): each part of a segment keeps up to 0.1 s of audio around its words, never more than halfway to the nearest other kept word or past the recording, fades in and out over 10 ms and is joined to the next part by 300 ms of silence. The margin and the fade are set in `configs/pilot/data.yaml`.
- Listening check: a script on OMNI cuts 20 random segments as original clips at a common loudness, numbered in random order, with a key to open only after listening.
- Chunk check of the interviewer filter: a GPU job scores every short stretch of speech inside the listened clips on its own, with a control on known interviewer lines.
- Interviewer detector, first part: spectral fingerprints of short sound events, the search for each event's best match in every other session, and a GPU job that checks the premise (gate P1) and builds the template libraries of the interviewer's recurring clips (gate P2).
- Interviewer detector, second part: a GPU scan of every session with every template (normalised cross-correlation, on all cells and on the quiet frames), pitch decoys that set the threshold, zones, and a job that checks gates D1-D4 with planted copies and reports what the zones would cost by class and gender.

### Changed

- 2026-11-12: the interviewer filter knows 70 of Ellie's scripted lines instead of 10, in her own wording only, because 1.6% of the segments still held her questions in their words.
- 2026-11-13: an interviewer detector is pre-registered in the pilot protocol, with its settings, gates and one fallback in `configs/pilot/data.yaml`, because the listening check heard the interviewer in 4 of 20 clips that the speaker-verification filter had kept.
- 2026-11-13: the pilot runs twice, because the interviewer detector failed its premise gate: a preliminary run on the current speech, then the main run once the interviewer's times from the DAIC-WOZ transcripts allow her to be cut out. Settings fixed in the first run carry over.
