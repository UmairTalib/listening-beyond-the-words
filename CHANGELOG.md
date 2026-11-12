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
