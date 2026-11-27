# Engineering conventions

This file says how the repository is organised and how work moves from the Mac to OMNI and into git. Tools enforce most rules: `make check`, the git hooks and CI. When a rule changes, change the tool settings and this file in the same commit.

1. [Repository layout](#1-repository-layout)
2. [Environment](#2-environment)
3. [Everyday commands](#3-everyday-commands)
4. [Code style](#4-code-style)
5. [Configuration](#5-configuration)
6. [Paths and the workspace](#6-paths-and-the-workspace)
7. [Data protection](#7-data-protection)
8. [Reproducibility](#8-reproducibility)
9. [Slurm jobs](#9-slurm-jobs)
10. [Tests](#10-tests)
11. [Commits](#11-commits)
12. [Branches and pull requests](#12-branches-and-pull-requests)
13. [Versions, changelog and releases](#13-versions-changelog-and-releases)
14. [Writing documentation](#14-writing-documentation)

## 1. Repository layout

```text
.github/                     CI workflow and pull request template
configs/<study>/             settings and pins: data.yaml (data, cut, clip versions, file
                             hashes) and models.yaml (model revisions)
docs/                        pilot protocol, pilot analysis, these conventions and the glossary
reports/<study>/             aggregate results only: tables and figures (from the first results)
scripts/                     one Python script per pipeline step, and shell scripts for OMNI
                             (download, unpack, environment)
scripts/slurm/               Slurm job files, common.sh and submit.sh
scripts/check_commit_msg.py  the commit-msg hook
src/listening_beyond_words/  the package, one flat module per topic
tests/                       pytest suite, synthetic data only
```

Inside the package:

```text
settings.py      workspace paths, read from LBW_WORKSPACE
runinfo.py       time, commit and tags, per-item seeds, the seeds of a process, repeatable
                 PyTorch kernels, file hashes, environment and GPU, run.json and error
                 records without messages
sessions.py      the 189 pilot sessions and the guard that refuses every other session
labels.py        PHQ-8 labels of the pilot
partitions.py    participants split into folds
transcripts.py   E-DAIC transcripts and the word normalisation
daicwoz.py       DAIC-WOZ transcripts: the speakers' turns and their times
interviewer.py   the interviewer filter (speaker verification)
ellie.py         the interviewer detector of 2026-11-13, stopped at its first gate
alignment.py     forced alignment of the words to the audio
cut.py           the cut rule: which words stay, and where a part ends
segments.py      40-word segments and what lies around their parts
views.py         the clip versions: x0, x2 and rev, their levels and 16-bit samples
revoice.py       the re-voiced clip x1: its text, voices and audio, and the Kokoro wrapper
clips.py         the files around the clips: recordings, transcript lines, pinned inputs,
                 listening keys, FLAC
wordcheck.py     the word check (check F4 and the checks of x1): its sample, the text rule
                 and WER, the leakage classifiers and their interval, and the verdicts
voicecheck.py    the voice checks of x1: voice measures per part, ICC and voice probe
recognisers.py   the word check's two ASR systems, faster-whisper and wav2vec2
models.py        pinned model revisions, the record of a snapshot's files and the check of
                 loaded weights
voxtral.py       Voxtral loading, prompt and score, the cache path (the encoder's output,
                 the pass-through, the score from a stored output) and the overflow watch
cache.py         the stored encoder outputs: .npy files, their manifest, the check's
                 sample and the pinned cache runs with their checks
zeroshot.py      the zero-shot scores: their settings, the check's sample and the reader
                 of the pinned scores
scoring.py       the scoring loop that every test-fold score comes from, the embargo before
                 any score, and the label-free numbers of the scores
config.py        the training configs: an arm file with the train.yaml it extends, checked
                 into frozen dataclasses, and the SHA-256 of the resolved config
loss.py          cross-entropy on z, the CPG gain term, its running statistics, the ramp of
                 λG and the logged measures of a step
bags.py          the bag sampler: the training participants' bags of each epoch and step
training.py      the model that trains (LoRA and the adapter copy), AdamW and the loss
                 scaler, one training step, the saved weights, the hand-over to the scoring
                 model, and the measures of the checks and rules read on the training logs
espeak.py        the workspace espeak-ng for Kokoro's text front end
```

- Library modules hold functions and classes. Most take arrays, tables and dataclasses, not file paths, so tests can call them with synthetic data. A few named readers and writers take paths, such as `sessions.session_files`, `clips.read_session`, `clips.write_view` and `transcripts.read_transcript`.
- Each pipeline step is one script in `scripts/`, named after what it does: `build_segments.py`, `build_views.py`, `draw_wordcheck_sample.py`. A step reads `configs/pilot/data.yaml`, calls the library, writes a new folder in the workspace named after what it holds (`derived/segments/run-<UTC time>/`, `views/<view>/run-<job id>/`) and leaves a `run.json` there (section 8). It prints aggregate numbers only.
- A step that needs a GPU, or that runs for more than a few minutes on a CPU, also has a job file of the same name, `scripts/slurm/<step>.sbatch`, with the job name `lbw-<step>` in hyphens (`lbw-build-views`). Shorter steps run on a login node; their docstring says so.
- The re-voiced clip `x1` takes three steps. `check_revoice.py` (a job) runs the dry run of Kokoro's text front end and the test render, whose clips the researcher hears. `build_revoiced.py` (an array job of 20 tasks, all at once) writes the clips into `views/x1/run-<array job id>/<session>/` and each task's records into `tasks/<t>/` there. `merge_revoiced.py` (on a login node) checks every task and file and only then writes the run's `manifest.parquet`, `parts.parquet` and, last, its `run.json`, so a run folder with `run.json` is merged. Unlike other steps (section 8), the merge writes `run.json` only at the end, and a refused merge writes nothing. If any task fails, nothing is merged, and the whole array runs again into a new run folder; the merge refuses task folders from another array job.
- The word check is one GPU job, `check_views.py` (job file `scripts/slurm/check_views.sbatch`, job name `lbw-check-views`), which writes `derived/wordcheck/run-<job id>/`. Its `transcripts.parquet` holds the ASR systems' text, and its other tables hold numbers per segment and per participant; only the researcher opens them, on OMNI. The script is not called `wordcheck.py`, because the tests of that module are `tests/test_wordcheck.py`, which a script of that name would claim too (section 10). It has three modes, whose options name a pinned view or an earlier run, never a setting (section 5):
  - by default, it reads the pinned word-check sample in the views of `word_check.transcribed_views`;
  - `--views <view> ...` reads only the named pinned views on the sample, under the rules of check F4, for the backup of F4. It needs the word check's own run pinned (`wordcheck_run`) and made from the same pinned inputs. Once a leakage rerun of that run is pinned (`leakage_rerun`), it reads their leakage on all segments, as the rerun read that of `rev`; while a rerun that the run's `x0` control called for is not pinned, it refuses to run;
  - `--rerun derived/wordcheck/run-<job id>` runs the leakage part again on all segments of `x0` and `rev`, as the `x0` control may ask. It refuses a first run that is not done, that comes from another commit, config or inputs, whose `x0` control did not ask for the rerun, or whose `transcripts.parquet` no longer matches its `run.json`.
- The encoder caches are one GPU job, `build_caches.py` (job file `scripts/slurm/build_caches.sbatch`, job name `lbw-build-caches`), with two modes, whose options name a pinned view or earlier runs, never a setting (section 5):
  - `--view <view>` runs Voxtral's frozen audio encoder on every clip of the pinned view, one clip per call, and writes its outputs in fp16 to `cache/<view>/run-<job id>/` (section 6), with a `manifest.parquet` that the pin `cache_runs` names by its SHA-256;
  - `--verify cache/<view>/run-<job id> ...` computes the log-mel features and the encoder outputs of a sample of each named cache run again from the audio and compares them byte for byte, then runs the language model without its output layer on the longest clips with the overflow watch on. It writes only `derived/cache_check/run-<job id>/run.json`, which a pin of `cache_runs` names as its `verified_by`. A flag in the language model leaves the caches standing; `cache.pinned_cache` passes it on to the jobs that score.
- The zero-shot scores are one GPU job, `score_zeroshot.py` (job file `scripts/slurm/score_zeroshot.sbatch`, job name `lbw-score-zeroshot`), which takes no option. It refuses to run until the rules of the training code are dated (`configs/pilot/train.yaml` exists), while their repeatability mode (`precision.deterministic` there) still says `pending`, which the commit that records the GPU tests and the training timing run changes, until the pinned word-check run shows the checks of `x1` finished, and while the check of any of the four caches flagged the language model. It scores a check sample from the audio and from the cache, which must agree bit for bit, and then scores every segment of `x0`, `x1` and `rev` from the pinned caches. It writes `runs/pilot/pilot-zeroshot-<YYYYmmdd-HHMMSS>/`, the run name of section 4 without round and fold, whose `scores.parquet` the pin `zeroshot_scores` names by its SHA-256. It reads no label.
- Download and unpack stay shell scripts with their current names: `scripts/download_edaic.sh`, `scripts/unpack_edaic_pilot.sh` and its job file `scripts/slurm/unpack_edaic_pilot.sbatch`.
- Create a folder only when its first file lands, and delete its `.gitkeep` then.
- Shell scripts are glue for OMNI. New pipeline logic goes into Python.
- The training of B0, B1 and CPG is one GPU job, `train.py` (job file `scripts/slurm/train.sbatch`, job name `lbw-train`, submitted as `lbw-train-<arm>`), one array of 5 fold-runs per arm and round: `--config configs/pilot/<arm>.yaml --round <round> --fold <fold>`, the fold set by the array task. It refuses to run outside a clean offline GPU job, with an arm file outside `configs/pilot/` of the checkout, off the release tags (section 8), while any cache check flagged the language model, and with other numeric settings than the caches'; a run of CPG's round 1, fold 1 (the timing run excepted) also refuses while the run.json of another run of it cannot be read, since it could not tell where the once-only readings were made; apart from the timing run, it also refuses while `precision.deterministic` in `configs/pilot/train.yaml` says `pending` or the embargo holds, and B1 and CPG need the pinned zero-shot scores, so that a run that could not score stops before it trains. It trains from the pinned caches (pilot protocol, section "Training"), saves LoRA and the adapter copy, releases the training model and scores its own test fold on `x0` and `rev` with the zero-shot job's scoring code (`scoring.score_views`) from the saved file. It writes `runs/pilot/pilot-<arm>-r<round>-f<fold>-<YYYYmmdd-HHMMSS>/` with `adapter/`, `metrics.jsonl` (one line per step, no ID), `run.log`, `scores.parquet` and `run.json`. With `--timing` it is the timing run of check F5 (CPG, round 1, fold 1): the smoke bag at loss scale 1.0, then the first steps in warn-only and in strict mode, and the projection of a three-view run; it writes only `derived/train_timing/run-<job id>/run.json`, keeps no weights and scores nothing. `--allow-dirty` is for debugging only (section 8), and `--verbose` logs every step.
- The analysis step, `evaluate`, is still to come. Where section 8 describes it, it gives the plan, and it arrives with its code.

## 2. Environment

- Python 3.12, set in `.python-version`. uv installs it on every machine; the OMNI Python modules are not used.
- [uv](https://docs.astral.sh/uv/) manages the environment. `pyproject.toml` lists the direct dependencies. `uv.lock` pins every package for the Mac (arm64) and for Linux x86_64 (OMNI and CI). Commit the two files together.
- PyTorch comes from the PyTorch package index, through one of two extras. Use exactly one:
  - `cpu` on the Mac and in CI;
  - `cu126` on OMNI. It is the last CUDA build of PyTorch with V100 (sm_70) kernels. The PyPI build of the same version has none.
- The `pipeline` extra adds the tools for clip building, the word check and the probe (Kokoro, faster-whisper, opensmile, Parselmouth). Only OMNI needs it. opensmile's licence allows research use, not commercial use.
- jiwer, which computes the word error rate, and num2words, which writes out the numbers in the text of the re-voiced clip as misaki does, are in the main set, so that their tests run on the Mac and in CI.
- `pyproject.toml` pins exact versions of the packages that can change the numbers: torch, torchaudio, transformers, peft, mistral-common and num2words, and the pipeline tools kokoro, misaki, faster-whisper, opensmile and en-core-web-sm. Ruff is pinned to match its git hook. The rest have lower bounds, and `uv.lock` pins them.
- Add a dependency with `uv add --no-sync <package>`, then run `make env`. Upgrade a package on purpose with `uv lock --upgrade-package <package>`, in its own `build(deps)` commit. No upgrades between the settings freeze and the pilot report.
- In VS Code, use the Ruff, Python and EditorConfig extensions, and select `.venv` as the interpreter.

### On the Mac

```bash
make env     # .venv with the CPU build of PyTorch
make hooks   # git hooks, once per clone
make check   # lint, type check and tests (CI also runs make hooks-run)
```

### On OMNI

Add this block to `~/.bashrc` once:

```bash
export LBW_WORKSPACE="$(ws_find cpg-thesis)"
export UV_PROJECT_ENVIRONMENT="$LBW_WORKSPACE/env/venv"
export UV_CACHE_DIR="$LBW_WORKSPACE/env/uv-cache"
export LBW_TORCH=cu126 LBW_EXTRAS=pipeline
export UV_NO_SYNC=1
export PATH="$HOME/.local/bin:$PATH"
```

Then, on a login node, inside the clone in your home folder, run `bash scripts/setup_omni_env.sh`. It installs uv if needed, builds the environment in the workspace from `uv.lock` and checks that PyTorch has V100 kernels. It also builds espeak-ng 1.52.0 into the workspace once, with `scripts/build_espeak_ng.sh`, and checks that Kokoro's text front end uses it. The espeak-ng library that comes with that front end needs glibc 2.29, and OMNI has 2.28. Run the setup script again after every pull that changes `uv.lock`, and after a move to a new workspace.

`UV_NO_SYNC=1` stops a bare `uv run` from changing the environment. Only the setup script and `make env` change it.

Code reaches OMNI only through git: commit and push on the Mac, then `git pull` on OMNI. Never edit code on OMNI.

## 3. Everyday commands

| Command | What it does |
| --- | --- |
| `make env` | Create or update the environment from `uv.lock` |
| `make lock` | Update `uv.lock` after a change to `pyproject.toml` |
| `make fmt` | Format the code and apply safe lint fixes |
| `make lint` | Check format and lint rules without changing files |
| `make typecheck` | Type-check the package with mypy |
| `make test` | Run the fast tests (synthetic data, CPU) |
| `make test-all` | Run every CPU test, including slow ones |
| `make test-gpu` | Run the GPU tests (on OMNI, inside a GPU job) |
| `make check` | Run lint, type check and tests (CI also runs `make hooks-run`) |
| `make hooks` | Install the git hooks |
| `make hooks-run` | Run every git hook on every file |
| `make clean` | Delete tool caches and build output, never data |

## 4. Code style

- Ruff formats and lints all Python code, with the settings in `pyproject.toml`: line length 100, double quotes, sorted imports, absolute imports only.
- Ruff also bans `print()` outside `scripts/`, the global NumPy random state and `os.path`.
- Every function in `src/` has type hints. mypy checks the package in CI. Tests are not type-checked.
- Docstrings follow the Google style: a one-line summary, then `Args:`, `Returns:` and `Raises:`. Every public module, class and function has one.
- Docstrings give tensor shapes, for example `z0: Scores of shape (B,).`
- Code that implements a formula names its source, for example "pilot protocol, fixed settings".
- Comments explain why, not what.
- Raise specific exceptions with a clear message. Never write a bare `except:`. Use `assert` only in tests.

### Naming

| Thing | Convention | Example |
| --- | --- | --- |
| Modules, functions, variables | snake_case | `build_segments`, `z0_bar` |
| Classes | PascalCase | `TrainConfig` |
| Constants | UPPER_SNAKE_CASE | `PILOT_SESSION_IDS` |
| Maths symbols | the method's symbol, in ASCII | `z0_hat` for ẑ0, `gain` for G_b |
| Tensor dimensions | single capitals, inside a function | `B, T, D = h.shape` |
| Clip versions | `x0` original, `x1` re-voiced, `x2` muffled, `rev` reversed | `views/x1/` |
| Studies | `pilot`, later `main` | `configs/pilot/` |
| Arms | `zeroshot`, `b0`, `b1`, `cpg` | `configs/pilot/cpg.yaml` |
| Participants and segments | `pid` (int); segment `{pid}_{index:03d}` | `303_007` |
| Runs | `{study}-{arm}-r{round}-f{fold}-{YYYYmmdd-HHMMSS}` | `pilot-cpg-r1-f3-20261201-142233` |
| Files and folders | snake_case | `unpack_edaic_pilot.sh` |
| Environment variables | `LBW_` prefix | `LBW_WORKSPACE` |
| Branches | `<type>/<kebab-case>` | `feat/cpg-gain` |

### Logging

- Each module gets its logger with `logger = logging.getLogger(__name__)`. Library code never prints.
- The training steps set up logging once: level INFO (DEBUG with `--verbose`), the format `%(asctime)s %(levelname)s %(name)s: %(message)s`, to stderr (the Slurm log) and to `run.log` in the run folder. The data steps built so far print aggregate numbers only.
- Training metrics go to `metrics.jsonl` in the run folder, one JSON object per logged step. No external trackers.
- Logs never contain transcript text or audio content. Participant IDs may appear in workspace logs, which never go into git.
- The steps that read transcript text (those of `x1`) never print or store an error's message, because it could hold a word: on any error they print one line with the error's type, the segment and part, and where `errors.log` is, which holds traceback frames and type names only (`runinfo.record_failure`). Where no `errors.log` holds them, as when a job fails before its folder exists or in the merge, which writes nothing when it fails, the frames and type names go to stderr, never the message. The failure is reported outside the `except` clause, so that nothing raised while reporting is chained to the error, and Python's own report of an error in a finaliser is replaced by its type (`runinfo.quiet_unraisable`). Only the steps' own errors (`runinfo.StepError`), whose messages name rules, counts and IDs, are shown in full. What misaki and Kokoro print is caught and counted.

## 5. Configuration

- One YAML file per experiment, in `configs/<study>/`, for example `configs/pilot/cpg.yaml`. A run is fully defined by its config file and its coordinates (round and fold) on the command line. No command-line option changes a setting.
- Settings shared by all arms live in one file, which each arm file extends with `extends: train.yaml`. One level only. A shared change, such as the 4-epoch rule, is then one edit.
- `config.py` turns the YAML into frozen dataclasses. The loader rejects unknown keys and wrong types, so a typo stops the run before it starts.
- Keys are snake_case and carry their unit: `gap_ms: 300`, `cutoff_hz: 400`, `segment_words: 40`.
- Write floats with a dot and a signed exponent: `1.0e-4`, not `1e-4`. PyYAML reads `1e-4` as a string.
- A value taken from the method cites its source in a comment, for example `# pilot protocol, fixed settings`.
- Model revisions are commit hashes, never `main`.
- Paths in configs are relative to the workspace root.
- The run record stores the resolved config, the arm file's own keys and those of `train.yaml` merged, and its SHA-256 (canonical JSON, sorted keys). Training run folders are named as in section 4, with the arm, round, fold and time, not by the hash.

The pilot configs:

```text
configs/pilot/data.yaml       labels, partitions (seeds 101 and 102, 5 folds, file
                              hashes), segments (40 words, 300 ms), bags, clip versions,
                              word-check thresholds
configs/pilot/models.yaml     pinned revisions of Voxtral, Kokoro and the two ASR models
configs/pilot/train.yaml      shared training settings: model, prompt, seeds per round,
                              epochs, LoRA, optimiser, precision, sampler, loss, gain,
                              rules, scoring and the timing run
configs/pilot/b0.yaml         extends train.yaml: cross-entropy on x0
configs/pilot/b1.yaml         extends train.yaml: cross-entropy on x0, x1 and x2
configs/pilot/cpg.yaml        extends train.yaml: B1 plus the gain term
configs/pilot/zeroshot.yaml   extends train.yaml: scoring only
configs/pilot/evaluate.yaml   the analysis (docs/pilot_analysis.md): arms, test views,
                              differences, bootstrap, F3's bound, the decision rule and
                              its framing line, the first runs' commit and, after
                              training, the pins of the kept runs
```

## 6. Paths and the workspace

- No absolute paths in code or configs. `settings.get_workspace()` reads the workspace root from `LBW_WORKSPACE` and names the standard folders.
- On OMNI the root is `$(ws_find cpg-thesis)`. The Mac holds no data, except the clips of a listening check, copied there only to be heard. The tests point `LBW_WORKSPACE` at a temporary folder.

```text
$LBW_WORKSPACE/
  edaic/raw/              packed originals and labels, read-only
  edaic/manifest/         checksums, inventories and status files
  edaic/sessions/         the 189 unpacked pilot sessions, read-only
  daicwoz/raw/            the official DAIC-WOZ files (189 session archives), read-only
  daicwoz/manifest/       checksums, server listing and status of the DAIC-WOZ download
  daicwoz/run-<time>/     DAIC-WOZ transcripts taken from the archives (read-only), their index
  derived/                labels, partitions, word times, segment manifests, the records of
                          the word check, of the cache checks (cache_check/) and of the
                          training timing run (train_timing/run-<job id>/run.json)
  views/<view>/           clip versions (FLAC): x0, x1, x2, rev
  cache/<view>/           frozen-encoder outputs: run-<job id>/<session>/<segment>.npy in
                          fp16, listed in the run's manifest.parquet
  runs/<study>/<run_id>/  a trained run: adapter/ (LoRA and the adapter copy, in fp32),
                          metrics.jsonl, run.log, scores.parquet, run.json; the zero-shot
                          scores in runs/pilot/pilot-zeroshot-<time>/
  hf/                     pinned model snapshots (HF_HOME)
  logs/slurm/             Slurm logs
  env/                    environment, uv cache and espeak-ng, rebuilt by scripts/setup_omni_env.sh
```

- A workspace expires after 30 days and allows 3 extensions. Extend it with `ws_extend` before it expires. It has no backup. Everything except `edaic/raw/` and `daicwoz/raw/` can be rebuilt from git and the raw data, but `x1` not bit for bit: Kokoro's renders depend on the CPU and the number of threads.
- Before the last extension runs out, allocate a new workspace, copy `edaic/`, `daicwoz/`, `derived/` and `views/`, update `LBW_WORKSPACE` and run `scripts/setup_omni_env.sh` again. Copy `cache/` and `runs/` too once they hold pinned runs: a cache built again is a new run, which needs its check and its pin again.

## 7. Data protection

Only code, configs, documentation and aggregate numbers go into git.

- Never commit audio, transcripts, clip versions, features, caches, checkpoints, labels, partitions, segment manifests, per-person or per-segment scores, the USC download link, approval e-mails or secrets.
- `.gitignore` and two hooks guard this. `forbid-data-files` blocks audio, tensor, checkpoint, archive and notebook files. `forbid-tables-outside-reports` blocks CSV, TSV and JSONL files outside `reports/`.
- `reports/` holds aggregate numbers only: per-fold AUROC, means, intervals and figures. Report writers refuse columns that identify a person or a segment (`pid`, `segment_id`).
- Sessions 600–718 are never opened or used before the external test. The download script fetches them and records their sizes and checksums; on OMNI they stay packed and locked (no read permission). Apart from the download and unpack scripts, all code that opens a session's files (recording and transcript) goes through `sessions.py`. Its `session_files` gives a session's paths only for the 189 pilot IDs (300–492 without 342, 394, 398 and 460) and raises an error for any other ID; `clips.read_session` reads a recording through it. The label files also hold rows of other sessions: `labels.py` reads them and keeps only the pilot IDs (`sessions.PILOT_IDS`) right after reading.
- Labels come from the PHQ-8 total (10 or more), never from the binary column.
- Data never leaves OMNI, except the clips of a listening check, copied to the Mac only to be heard. It never goes to external services, hosted notebooks or online trackers.
- Files are private: scripts and jobs set `umask 077`.
- The download links are passed to `scripts/download_edaic.sh` and `scripts/download_daicwoz.sh` as arguments when they run. They never appear in a file in git.
- Code never deletes anything on OMNI, apart from the partial temporary file of a write that failed, which the write made itself and no run lists (`cache.write_file`). Each run writes a new folder. Clean-ups are manual: list what would go first, then delete.
- Tests and CI use synthetic data only.

## 8. Reproducibility

- Every random source is seeded from the config. In the pilot, round 1 uses partition and training seed 101, and round 2 uses 102.
- `runinfo.seed_everything(seed)` seeds Python, NumPy and PyTorch. NumPy code takes a `numpy.random.Generator`.
- Training's repeatability follows `precision.deterministic` in `configs/pilot/train.yaml` (pilot protocol, section "Training", bullet "Number formats and repeatability"). With `warn_only`, PyTorch's deterministic algorithms are on with `warn_only=True`, as in the jobs of the encoder caches and the zero-shot scores (`runinfo.deterministic_torch`, which `voxtral.set_numerics` calls), so an operation without a repeatable version only warns; with `strict`, training stops at any such operation. The timing run decides: `strict` if strict mode runs and its median three-view step takes at most 10% longer than in warn-only mode, otherwise `warn_only`. Until the commit that records the timing run sets it, the key says `pending`, and every training run but the timing run refuses it, as does the zero-shot job. Scoring always switches back to the zero-shot job's numeric settings first.
- Randomness per item, such as the Kokoro rendering of a segment, uses a seed derived from the run seed and the item ID with `hashlib`. Never use Python's `hash()`: it changes between processes.
- GPU runs are close to repeatable, not bit-identical. Record the facts and promise no more. The comparisons of the encoder caches and the zero-shot scores are the one exception (pilot protocol, section "Encoder caches and zero-shot scores"): the encoder's output for a clip, and the score z of a clip scored on its own, repeat byte for byte when the jobs use the same recorded numeric settings on the same GPU and CPU type. So these comparisons demand equal bytes, never a tolerance, and a difference is a fault to find and fix.
- Every step writes `run.json` into its output folder, at the start and again at the end:

| Field | Content |
| --- | --- |
| `run_id`, `step`, `arm`, `round`, `fold` | what ran |
| `status`, `started_utc`, `finished_utc` | `running`, `done` or `failed`, and when |
| `git` | commit, dirty flag, and the output of `git describe --tags --always --dirty` with the tags at the commit, from `runinfo.git_state` |
| `config` | path, resolved content and SHA-256 |
| `inputs` | workspace-relative input paths with their SHA-256 or run IDs |
| `seeds` | every seed used |
| `environment` | Python, `uv.lock` SHA-256, torch, transformers, peft and numpy versions, CUDA, cuDNN, GPU and driver |
| `slurm` | job ID, array task ID and node |

The data steps built so far record the fields that apply to them: `config` holds the file and its SHA-256, `settings` the values used, and `environment` the packages they use. `runinfo.write_json` writes a `run.json` in one step, to a temporary file that is then renamed onto it, so a step that dies while writing never leaves half of one; the merge of `x1` counts a run folder with a `run.json` as merged.

- `train` and `evaluate` refuse to start when the checkout has uncommitted changes. `train` scores its own test fold, in the same job, right after it saves the weights (pilot protocol, section "Training", bullet "Scoring"), so there is no `score` step. Their `--allow-dirty` option exists for debugging only: `run.json` records it, and pilot runs never use it. `submit.sh` refuses a dirty checkout too, and each job checks that the checkout has not moved since it was submitted. Do not pull on OMNI while pilot jobs wait or run.
- Partitions are saved with their SHA-256 before any training. `configs/pilot/data.yaml` records the hashes, and training stops when a partition file does not match.
- Settings freeze, after the training logs of the first CPG run of round 1, fold 1: the logged gain-alive rule (m to 0.5 or 0.1) and the 4-epoch rule are the only allowed changes, decided from training logs only. Each is made once, in one `exp(pilot)` commit, and applies to all arms. It is dated in the settings change log of [`pilot_protocol.md`](pilot_protocol.md) and in `CHANGELOG.md`, and sent to the supervisor in one line. Then release 0.1.0 as described in section 13: a `chore(release): 0.1.0` commit tagged `v0.1.0`. Until a `v0.1.x` tag exists, the train job refuses every run but the timing run, B0's round 1 and that first CPG run; once one exists, every pilot run, reruns of these included, must come from a checkout at the newest `v0.1.x` tag, so that every run after a release uses what it holds. No setting changes after test scores are seen, apart from these two rules and the pre-set fallbacks (8 segments for check F5; the flagged layers in fp32 after an overflow), which were fixed before any score. A fallback used after the freeze is dated in the same way, with its evidence, in one `exp(pilot)` commit released as the next `0.1.x`.

## 9. Slurm jobs

- A step that needs a GPU, or that runs for more than a few minutes on a CPU, has one job file, `scripts/slurm/<step>.sbatch` (section 1); shorter steps run on a login node. Job names are `lbw-<step>` in hyphens, plus the arm where it helps (`lbw-train-cpg`).
- Submit with `scripts/slurm/submit.sh [--option=value ...] <step> [arguments]`. It refuses a checkout with uncommitted changes, sends the log to `$LBW_WORKSPACE/logs/slurm/<job name>-<job id>.log`, records the commit for the job to check and submits every job with `--no-requeue`. Slurm on OMNI would otherwise requeue a job after a node failure under the same job ID, and the requeued job would meet its own half-written output. A job lost this way is rerun by hand.
- Every job file has this shape, shown here for the train job, `scripts/slurm/train.sbatch`, with its comment cut short:

```bash
#!/usr/bin/env bash
# Slurm job: train one arm of the pilot in one round, one fold-run per array task (GPU).
# The 4 CPUs are those of the cache jobs (encoder_cache.cpu_threads), so that the numeric
# settings match the caches' record. After the settings freeze, submit from a checkout at
# the newest v0.1.x tag (git checkout v0.1.x on OMNI); the job refuses any other.
#
# Submit from an OMNI login node:
#   scripts/slurm/submit.sh --job-name=lbw-train-cpg train configs/pilot/cpg.yaml 1
#
# Array task t trains fold t + 1 of the given round (5 fold-runs). CPG's round 1, fold 1
# runs alone first (--array=0); after the freeze, CPG's round 1 takes --array=1-4.
#
#SBATCH --job-name=lbw-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:30:00
#SBATCH --array=0-4
set -euo pipefail
source scripts/slurm/common.sh

config="${1:?usage: train.sbatch <config> <round> [--timing]}"
round="${2:?usage: train.sbatch <config> <round> [--timing]}"
shift 2
task="${SLURM_ARRAY_TASK_ID:?submit this job as an array}"
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv
python scripts/train.py --config "$config" --round "$round" --fold $((task + 1)) "$@"
```

- Order: shebang, a comment with the purpose, how to submit and any array mapping, then the `#SBATCH` lines. Slurm stops reading `#SBATCH` lines at the first command.
- Set partition, time, memory and CPUs explicitly. GPU steps use `--partition=gpu --gres=gpu:1`. CPU steps use `short` when they fit its time limit, otherwise `medium`. Ask for about 1.5 times the measured run time; the limit is 6 h on `short` and 1 day on `gpu` and `medium`.
- Independent runs go side by side as array tasks: on `gpu` with one GPU each, at most 10 at once (`%10`); on `short`, at most 20 at once (`%20`). The limit on `gpu` holds across arrays too: the train job runs one array of 5 tasks per arm and round, so at most two such arrays run at a time, and later ones wait with `--dependency`.
- `common.sh` sets `umask 077`, turns off core dumps (`ulimit -c 0`, so that a crash never writes process memory into the clone), purges the modules (the PyTorch wheels bring their own CUDA), activates the environment, sets `HF_HOME`, `HF_HUB_OFFLINE=1` and the determinism variables, and prints the job, node and commit.
- Jobs never download. Fetch models on a login node into `$LBW_WORKSPACE/hf` first, at the revisions in `configs/pilot/models.yaml`.
- A job never deletes or overwrites earlier output. A failed run keeps its folder with `status: failed`; the rerun writes a new folder.

## 10. Tests

- pytest, in `tests/`, with one file per module or step script: `tests/test_<module>.py`, `tests/test_<script>.py`. Shared fixtures live in `tests/conftest.py`, the stand-ins of the re-voiced clip's tests (a synthetic workspace and Kokoro's pipeline) in `tests/x1_fakes.py`, those of the word check's tests (a synthetic workspace with clips of all four views, fake ASR systems and a fake opensmile) in `tests/wordcheck_fakes.py`, and a tiny random Voxtral with a stand-in for its processor in `tests/voxtral_fakes.py`. The tests of the encoder caches and of the zero-shot scores reuse the word check's workspace. So do those of the training job (`tests/test_train.py`), with its labels and partitions kept and 8 to 12 segments per participant: the cache job and the zero-shot job build and pin its caches and zero-shot scores on the tiny Voxtral, and the job trains and scores on the CPU; they are marked `slow`. `tests/wordcheck_fakes.py` also holds the check of what may leave OMNI, `assert_only_aggregates_leave`, which the tests of every job that reads participant clips share (the word check, the encoder caches, the zero-shot scores and training): no segment or participant ID on the streams, apart from the one `FAILED:` line, and in `run.json` none outside its `error` and nothing per item. It sees the pilot IDs 300 to 399 only, so the job tests use pilot IDs, and a control shows that it fails on a planted ID. The training job's tests also look for each session ID of their workspace as a whole number, however it is worded, on the streams and in `run.log`, `metrics.jsonl` and `run.json`.
- Synthetic data only. Fixtures build audio (tones and noise at 16 kHz), word lists and label tables in code. Fake participant IDs start at 9001, outside the real ID ranges. Code behind the session guard (`sessions.session_files`, `clips.read_session`) accepts only pilot IDs, so its tests use pilot IDs such as 300 and 301, with synthetic audio in a temporary folder.
- Tests never touch the network or the real workspace. `conftest.py` sets `HF_HUB_OFFLINE=1` and points `LBW_WORKSPACE` at a temporary folder.
- Markers: `slow` for tests over about 10 seconds, `gpu` for tests that need CUDA. `make test` skips both. `make test-all` adds the slow tests. `make test-gpu` runs the GPU tests inside a GPU job on OMNI. GPU tests skip themselves when no GPU is present.
- Test names say what must hold, for example `test_revoiced_clip_gets_no_gradient_from_the_gain`.
- `tests/test_cpg_gain.py` ports these seven reference checks to PyTorch, in float64, with the reference's own numbers:
  1. A softmax over {Yes, No} equals sigmoid(logit(Yes) − logit(No)), and dCE/dz = sigmoid(z) − y.
  2. On 2,000 random cases, the gradient of the gain term on z̄0 matches −λG · sigmoid((m − G)/s) · (2y − 1) / σ0, and z̄1 gets exactly zero gradient.
  3. Without the stop-gradient, the gradient on z̄1 would push it towards the wrong label. The implementation must not have this path.
  4. With z0 = k · z1 + c and exact statistics, G is zero on bags where the words are right, and mirrored bags of the two classes get the same push where the words are wrong.
  5. The clamp example (y = 1, words −1.0, real audio −0.7): G = −0.70 with the clamp and +0.30 without it; push strength 1.000 and 0.500.
  6. The soft hinge is never more than s · ln 2 above the hard hinge, with the largest gap at G = m.
  7. As a bag grows confident (y = 1, words-only reference 0), the cross-entropy push |dCE/dz| = 1 − sigmoid(z̄0) fades gradually, while the gain push sigmoid((m − G)/s) stays close to 1 until G reaches the margin m and then falls off quickly.
- The cache-equals-audio test (pilot protocol, section "Encoder caches and zero-shot scores") demands equal bytes, never a tolerance: z from a stored encoder output equals z from the audio, for clips of one and two chunks. It runs on a tiny random Voxtral on the CPU (`tests/test_voxtral.py`, `tests/test_cache.py`, through a `.npy` file) and on the real model on OMNI (the GPU tests of `tests/test_voxtral.py`, synthetic clips only). On real clips the cache check compares log-mel features and encoder outputs, never z, and the zero-shot job compares z on its check sample before it scores anything else.
- It also checks the pushes on bags where the words are right, with exact and with running statistics, and the invariance cases. `tests/test_loss.py` checks that the gain is computed in at least fp32 from fp16 inputs, that λG ramps up over the first 20% of steps, that the running statistics use momentum 0.9 from first and second moments, once per step, and that σ never falls below a quarter of its zero-shot value or 0.1.
- The training code has its own tests on the tiny Voxtral on the CPU: the configs (`tests/test_config.py`, `tests/test_train_config.py`), the bag sampler (`tests/test_bags.py`), training, saving and the scoring model (`tests/test_training.py`), and the scoring loop (`tests/test_scoring.py`). Their GPU tests, and the training job's hand-over (`tests/test_train.py`), run on the real model in `scripts/slurm/smoke_test.sbatch`, with synthetic encoder outputs only.

## 11. Commits

Commit messages follow [Conventional Commits 1.0.0](https://www.conventionalcommits.org/en/v1.0.0/). The `commit-msg` hook checks them.

```text
<type>(<scope>): <subject>

<body: what changed and why, wrapped at 72 characters>

Refs: #12
```

- The header has at most 72 characters. The scope is optional.
- The subject is in the imperative and starts in lowercase: "add", "fix", "remove". It has no full stop at the end. A release subject may be the bare version, for example `chore(release): 0.1.0`.
- The body is optional. It explains why, not how.
- Footers: `Refs: #<issue>` and `BREAKING CHANGE: <what>`. No other trailers: the hook rejects every `<word>-by:` line. The commit author says who wrote the commit.
- One logical change per commit. `make check` passes at every commit.

| Type | Use for |
| --- | --- |
| `feat` | new behaviour in the package, scripts or jobs |
| `fix` | a bug fix |
| `test` | tests only |
| `refactor` | code changes that keep the behaviour |
| `perf` | faster or smaller, same results |
| `docs` | documentation only |
| `exp` | experiment configs, frozen settings and recorded aggregate results |
| `build` | dependencies, `pyproject.toml`, `uv.lock` |
| `ci` | the CI workflow |
| `chore` | repository upkeep and releases |
| `style` | formatting only |
| `revert` | undoing an earlier commit |

Scopes: `data`, `views`, `cache`, `model`, `loss`, `train`, `eval`, `config`, `cli`, `slurm`, `deps`, `pilot`, `readme`, `release` and `repo`. A new scope goes into `scripts/check_commit_msg.py` and this list in the same commit.

```text
feat(loss): add the CPG gain term with running statistics
test(loss): port the reference checks of the gain gradient
exp(pilot): set margin m to 0.1 after the gain-alive check
build(deps): pin transformers 5.22.2
chore(release): 0.1.0
```

`git log --oneline --grep='^exp'` lists every change to experiment settings.

### Definition of done

A change is done when one commit holds all of these:

1. The code, formatted and free of lint errors.
2. Unit tests on synthetic data that cover the new logic.
3. The matching doc section, with commands that have actually been run.
4. A line in `CHANGELOG.md` under *Unreleased*.

`make check` and the git hooks pass. When a study starts or ends, update the Status and Results sections of the README in the same commit.

## 12. Branches and pull requests

- `main` is the only long-lived branch. It stays green: CI passes on every push.
- Small, tested changes go straight to `main`.
- Use a short branch and a pull request when a change:
  - touches the loss, the training loop or a config after the settings freeze;
  - updates dependencies (`pyproject.toml`, `uv.lock`); or
  - takes more than a day, or needs CI before it lands.
- Branch names follow `<type>/<kebab-case>`, for example `feat/cpg-gain`.
- Squash-merge with the pull request title as the commit header, then delete the branch. The history stays linear.
- Never force-push `main`.

## 13. Versions, changelog and releases

- Versions follow [Semantic Versioning](https://semver.org/). They stay at `0.x` until the thesis is submitted. `pyproject.toml` holds the version, and the package reads it at run time.
- Between releases the version is the next release plus `.dev0`, for example `0.1.0.dev0`.
- `CHANGELOG.md` follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Add a line under `[Unreleased]` with each change worth knowing. Changes to method settings go under **Changed**, with the date and the reason.

| Tag | Marks |
| --- | --- |
| `v0.1.0` | Pilot settings frozen, after the training logs of the first CPG run of round 1, fold 1 (section 8). Every later run uses the newest `0.1.x` tag. |
| `v0.1.x` | Fixes that change no method setting, for example in the analysis code, and a pre-set fallback used after the freeze (section 8) |
| `v0.2.0` | Pilot report: aggregate results and the go / no-go decision |
| `v0.3.0` and later | Main study: registration, frozen main runs, external test |
| `v1.0.0` | The code and aggregate results of the submitted thesis |

To release:

1. Move the `[Unreleased]` entries under `## [0.1.0] - YYYY-MM-DD`.
2. Set `version = "0.1.0"` in `pyproject.toml`, then run `make lock` and `make check`.
3. Commit `chore(release): 0.1.0` and tag it: `git tag -a v0.1.0 -m "Pilot settings frozen"`.
4. Push the commit, then the tag (`git push origin v0.1.0`). Create a GitHub release from the tag with the changelog section as its notes.
5. Set the version to `0.2.0.dev0` in the next commit.

Tags are never moved or deleted.

## 14. Writing documentation

- Plain English. Short sentences. Active voice. One idea per sentence.
- British spelling: licence, optimise, behaviour.
- Sentence-case headings and one H1 per page.
- Open each page with one or two sentences on what it covers.
- Use tables for facts, numbered lists for procedures and fenced code blocks with a language tag for commands.
- Write placeholders in angle brackets inside code formatting, such as `<workspace-name>`. Bare angle brackets vanish on GitHub.
- Never write user names, absolute workspace paths, e-mail addresses or links that grant data access.
- Write workspace paths relative to `$LBW_WORKSPACE`.
- Write dates in ISO form (2026-11-12). Put a space between a number and its unit (16 kHz, 300 ms). Use an en dash for ranges (300–492) and a minus sign for differences (CPG − B0).
- Draw diagrams in Mermaid, so that GitHub renders them and they stay easy to edit.
- Link between pages with relative links. Do not paste bare URLs.
- Do not hard-wrap paragraphs. One paragraph per line keeps diffs readable.
- Use the terms of the [glossary](README.md#glossary). Give each thing one name.
- Every command in the docs has been run.
