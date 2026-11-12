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
- Download script that fetches E-DAIC straight into the OMNI workspace and checks sizes and checksums.
- Unpack job for the audio and transcripts of the 189 pilot sessions (300–492).
