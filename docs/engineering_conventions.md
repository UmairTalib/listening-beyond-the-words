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
configs/<study>/             experiment configs, one YAML file per experiment
docs/                        pilot protocol, these conventions and the glossary
reports/<study>/             aggregate results only: tables and figures
scripts/                     shell scripts for OMNI: download, unpack, environment
scripts/slurm/               Slurm job files, common.sh and submit.sh
scripts/check_commit_msg.py  the commit-msg hook
src/listening_beyond_words/  the package
tests/                       pytest suite, synthetic data only
```

Inside the package:

```text
settings.py   workspace paths, read from LBW_WORKSPACE
config.py     typed config dataclasses and the YAML loader
runinfo.py    seeds and the run record (run.json)
cli.py        the lbw command, one subcommand per pipeline step
data/         E-DAIC access, labels, partitions, word times, segments
views/        re-voiced, muffled and reversed clips; the word check
model/        Voxtral loading, prompt and score, LoRA, encoder cache
training/     bag sampler, losses (cross-entropy and CPG gain), training loop
evaluation/   person scores, AUROC, bootstrap, report tables
pipeline/     one module per pipeline step
```

- Library modules hold pure functions and classes. They take arrays, tensors and dataclasses, not file paths, so tests can call them with synthetic data.
- `pipeline/` modules do the reading and writing. Each step reads one config, calls the library, writes into its own workspace folder and leaves a `run.json` there.
- The pipeline steps are `inventory`, `labels`, `partitions`, `align`, `segments`, `views`, `wordcheck`, `cache`, `zeroshot`, `train`, `score`, `evaluate`, `probe` and `report`. Each step has one name, used everywhere: the subcommand (`lbw segments`), the module (`pipeline/segments.py`), the job file (`scripts/slurm/segments.sbatch`), the job name (`lbw-segments`) and the output folder.
- Download and unpack stay shell scripts with their current names: `scripts/download_edaic.sh` and `scripts/unpack_edaic_pilot.sh`.
- `cli.py` and the `lbw` entry in `[project.scripts]` land in the same commit.
- Create a folder only when its first file lands, and delete its `.gitkeep` then.
- Shell scripts are glue for OMNI. New pipeline logic goes into Python.

## 2. Environment

- Python 3.12, set in `.python-version`. uv installs it on every machine; the OMNI Python modules are not used.
- [uv](https://docs.astral.sh/uv/) manages the environment. `pyproject.toml` lists the direct dependencies. `uv.lock` pins every package for the Mac (arm64) and for Linux x86_64 (OMNI and CI). Commit the two files together.
- PyTorch comes from the PyTorch package index, through one of two extras. Use exactly one:
  - `cpu` on the Mac and in CI;
  - `cu126` on OMNI. It is the last CUDA build of PyTorch with V100 (sm_70) kernels. The PyPI build of the same version has none.
- The `pipeline` extra adds the tools for clip building, the word check and the probe (Kokoro, faster-whisper, jiwer, Parselmouth). Only OMNI needs it.
- `pyproject.toml` pins exact versions of the packages that can change the numbers: torch, torchaudio, transformers, peft and mistral-common. The rest have lower bounds, and `uv.lock` pins them.
- Add a dependency with `uv add --no-sync <package>`, then run `make env`. Upgrade a package on purpose with `uv lock --upgrade-package <package>`, in its own `build(deps)` commit. No upgrades between the settings freeze and the pilot report.
- In VS Code, use the Ruff, Python and EditorConfig extensions, and select `.venv` as the interpreter.

### On the Mac

```bash
make env     # .venv with the CPU build of PyTorch
make hooks   # git hooks, once per clone
make check   # everything CI runs
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

Then, on a login node, inside the clone in your home folder, run `bash scripts/setup_omni_env.sh`. It installs uv if needed, builds the environment in the workspace from `uv.lock` and checks that PyTorch has V100 kernels. Run it again after every pull that changes `uv.lock`, and after a move to a new workspace.

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
| `make check` | Run everything CI runs |
| `make hooks` | Install the git hooks |
| `make hooks-run` | Run every git hook on every file |
| `make clean` | Delete tool caches and build output, never data |

## 4. Code style

- Ruff formats and lints all Python code, with the settings in `pyproject.toml`: line length 100, double quotes, sorted imports, absolute imports only.
- Ruff also bans `print()` outside `scripts/`, the global NumPy random state and `os.path`.
- Every function in `src/` has type hints. mypy checks the package in CI. Tests are not type-checked.
- Docstrings follow the Google style: a one-line summary, then `Args:`, `Returns:` and `Raises:`. Every public module, class and function has one.
- Docstrings give tensor shapes, for example `z0: Scores of shape (B,).`
- Code that implements a formula names its source, for example "methodology rev 26, Appendix A.1".
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
- `lbw` sets up logging once: level INFO (DEBUG with `--verbose`), the format `%(asctime)s %(levelname)s %(name)s: %(message)s`, to stderr (the Slurm log) and to `run.log` in the run folder.
- Training metrics go to `metrics.jsonl` in the run folder, one JSON object per logged step. No external trackers.
- Logs never contain transcript text or audio content. Participant IDs may appear in workspace logs, which never go into git.

## 5. Configuration

- One YAML file per experiment, in `configs/<study>/`, for example `configs/pilot/cpg.yaml`. A run is fully defined by its config file and its coordinates (round and fold) on the command line. No command-line option changes a setting.
- Settings shared by all arms live in one file, which each arm file extends with `extends: train.yaml`. One level only. A shared change, such as the 4-epoch rule, is then one edit.
- `config.py` turns the YAML into frozen dataclasses. The loader rejects unknown keys and wrong types, so a typo stops the run before it starts.
- Keys are snake_case and carry their unit: `gap_ms: 300`, `cutoff_hz: 400`, `segment_words: 40`.
- Write floats with a dot and a signed exponent: `1.0e-4`, not `1e-4`. PyYAML reads `1e-4` as a string.
- A value taken from the method cites its source in a comment, for example `# rev 26, A.1`.
- Model revisions are commit hashes, never `main`.
- Paths in configs are relative to the workspace root.
- The run record stores the resolved config and its SHA-256 (canonical JSON, sorted keys). Folder names use the first 12 characters of the hash.

The pilot configs:

```text
configs/pilot/data.yaml       labels, partitions (seeds 101 and 102, 5 folds, 15% inner
                              validation, file hashes), segments (40 words, 300 ms),
                              clip versions, word-check thresholds
configs/pilot/models.yaml     pinned revisions of Voxtral, Kokoro and the two ASR models
configs/pilot/train.yaml      shared training settings: model, LoRA, optimiser, epochs,
                              bags, prompt, seeds per round
configs/pilot/b0.yaml         extends train.yaml: cross-entropy on x0
configs/pilot/b1.yaml         extends train.yaml: cross-entropy on x0, x1 and x2
configs/pilot/cpg.yaml        extends train.yaml: B1 plus the gain term
configs/pilot/zeroshot.yaml   scoring only
```

## 6. Paths and the workspace

- No absolute paths in code or configs. `settings.get_workspace()` reads the workspace root from `LBW_WORKSPACE` and names the standard folders.
- On OMNI the root is `$(ws_find cpg-thesis)`. The Mac never holds data, and the tests point `LBW_WORKSPACE` at a temporary folder.

```text
$LBW_WORKSPACE/
  edaic/raw/              packed originals and labels, read-only
  edaic/manifest/         checksums, inventories and status files
  edaic/sessions/         the 189 unpacked pilot sessions, read-only
  derived/                labels, partitions, word times, segment manifests
  views/<view>/           clip versions (FLAC): x0, x1, x2, rev
  cache/<view>/           frozen-encoder outputs
  runs/<study>/<run_id>/  checkpoints, metrics.jsonl, run.log, scores, run.json
  hf/                     pinned model snapshots (HF_HOME)
  logs/slurm/             Slurm logs
  env/                    environment and uv cache, rebuilt by scripts/setup_omni_env.sh
```

- A workspace expires after 30 days and allows 3 extensions. Extend it with `ws_extend` before it expires. It has no backup. Everything except `edaic/raw/` can be rebuilt from git and the raw data.
- Before the last extension runs out, allocate a new workspace, copy `edaic/` and `derived/`, update `LBW_WORKSPACE` and run `scripts/setup_omni_env.sh` again.

## 7. Data protection

Only code, configs, documentation and aggregate numbers go into git.

- Never commit audio, transcripts, clip versions, features, caches, checkpoints, labels, partitions, segment manifests, per-person or per-segment scores, the USC download link, approval e-mails or secrets.
- `.gitignore` and two hooks guard this. `forbid-data-files` blocks audio, tensor, checkpoint, archive and notebook files. `forbid-tables-outside-reports` blocks CSV, TSV and JSONL files outside `reports/`.
- `reports/` holds aggregate numbers only: per-fold AUROC, means, intervals and figures. Report writers refuse columns that identify a person or a segment (`pid`, `segment_id`).
- Sessions 600-718 are never listed, globbed or opened. All code that touches E-DAIC files goes through one module, `data/edaic.py`. It accepts only the 189 pilot IDs (300-492 without 342, 394, 398 and 460) and raises an error for any other ID.
- Labels come from the PHQ-8 total (10 or more), never from the binary column.
- Data never leaves OMNI: no copies to the Mac, no external services, no hosted notebooks, no online trackers.
- Files are private: scripts and jobs set `umask 077`.
- The download link is passed to `scripts/download_edaic.sh` as an argument when it runs. It never appears in a file in git.
- Code never deletes anything on OMNI. Each run writes a new folder. Clean-ups are manual: list what would go first, then delete.
- Tests and CI use synthetic data only.

## 8. Reproducibility

- Every random source is seeded from the config. In the pilot, round 1 uses partition and training seed 101, and round 2 uses 102.
- `runinfo.seed_everything(seed)` seeds Python, NumPy and PyTorch and turns on deterministic algorithms with `warn_only=True`. NumPy code takes a `numpy.random.Generator`.
- Randomness per item, such as the Kokoro rendering of a segment, uses a seed derived from the run seed and the item ID with `hashlib`. Never use Python's `hash()`: it changes between processes.
- GPU runs are close to repeatable, not bit-identical. Record the facts and promise no more.
- Every step writes `run.json` into its output folder, at the start and again at the end:

| Field | Content |
| --- | --- |
| `run_id`, `step`, `arm`, `round`, `fold` | what ran |
| `status`, `started_utc`, `finished_utc` | `running`, `done` or `failed`, and when |
| `git` | commit, `git describe` output and a dirty flag |
| `config` | path, resolved content and SHA-256 |
| `inputs` | workspace-relative input paths with their SHA-256 or run IDs |
| `seeds` | every seed used |
| `environment` | Python, `uv.lock` SHA-256, torch, transformers, peft and numpy versions, CUDA, cuDNN, GPU and driver |
| `slurm` | job ID, array task ID and node |

- `train`, `score` and `evaluate` refuse to start when the checkout has uncommitted changes. Their `--allow-dirty` option exists for debugging only: `run.json` records it, and pilot runs never use it. `submit.sh` refuses a dirty checkout too, and each job checks that the checkout has not moved since it was submitted. Do not pull on OMNI while pilot jobs wait or run.
- Partitions are saved with their SHA-256 before any training. `configs/pilot/data.yaml` records the hashes, and training stops when a partition file does not match.
- Settings freeze (week 3): the logged gain-alive rule (m to 0.5 or 0.1) and the 4-epoch rule are the only allowed changes, decided from training logs only. Each is made once, in one `exp(pilot)` commit, and applies to all arms. It is dated in the settings change log of [`pilot_protocol.md`](pilot_protocol.md) and in `CHANGELOG.md`, and sent to the supervisor in one line. Then tag the commit `v0.1.0`; every later pilot run uses it. No setting changes after test scores are seen.

## 9. Slurm jobs

- One job file per pipeline step: `scripts/slurm/<step>.sbatch`. Job names are `lbw-<step>`, plus the arm where it helps (`lbw-train-cpg`).
- Submit with `scripts/slurm/submit.sh [--option=value ...] <step> [arguments]`. It refuses a checkout with uncommitted changes, sends the log to `$LBW_WORKSPACE/logs/slurm/<job name>-<job id>.log` and records the commit for the job to check.
- Every job file has this shape:

```bash
#!/usr/bin/env bash
# Slurm job: train one arm of the pilot, one fold-run per array task (GPU).
#
# Submit from an OMNI login node:
#   scripts/slurm/submit.sh --job-name=lbw-train-cpg train configs/pilot/cpg.yaml
#
# Array task t trains round t/5 + 1, fold t%5 + 1 (10 fold-runs), at most 4 at once.
#
#SBATCH --job-name=lbw-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --array=0-9%4
set -euo pipefail
source scripts/slurm/common.sh

config="${1:?usage: train.sbatch <config>}"
task="${SLURM_ARRAY_TASK_ID:?submit this job as an array}"
lbw train --config "$config" --round $((task / 5 + 1)) --fold $((task % 5 + 1))
```

- Order: shebang, a comment with the purpose, how to submit and any array mapping, then the `#SBATCH` lines. Slurm stops reading `#SBATCH` lines at the first command.
- Set partition, time, memory and CPUs explicitly. GPU steps use `--partition=gpu --gres=gpu:1`. CPU steps use `short` when they fit its time limit, otherwise `medium`. Ask for about 1.5 times the measured run time; the limit is 1 day.
- Independent runs go side by side as array tasks with one GPU each, at most 4 at once (`%4`).
- `common.sh` sets `umask 077`, purges the modules (the PyTorch wheels bring their own CUDA), activates the environment, sets `HF_HOME`, `HF_HUB_OFFLINE=1` and the determinism variables, and prints the job, node and commit.
- Jobs never download. Fetch models on a login node into `$LBW_WORKSPACE/hf` first, at the revisions in `configs/pilot/models.yaml`.
- A job never deletes or overwrites earlier output. A failed run keeps its folder with `status: failed`; the rerun writes a new folder.

## 10. Tests

- pytest, in `tests/`, with one file per module: `tests/test_<module>.py`. Shared fixtures live in `tests/conftest.py`.
- Synthetic data only. Fixtures build audio (tones and noise at 16 kHz), word lists and label tables in code. Fake participant IDs start at 9001, outside the real ID ranges.
- Tests never touch the network or the real workspace. `conftest.py` sets `HF_HUB_OFFLINE=1` and points `LBW_WORKSPACE` at a temporary folder.
- Markers: `slow` for tests over about 10 seconds, `gpu` for tests that need CUDA. `make test` skips both. `make test-all` adds the slow tests. `make test-gpu` runs the GPU tests inside a GPU job on OMNI. GPU tests skip themselves when no GPU is present.
- Test names say what must hold, for example `test_revoiced_clip_gets_no_gradient_from_the_gain`.
- Before the first CPG run, `tests/test_cpg_gain.py` ports these seven reference checks to PyTorch, in float64:
  1. A softmax over {Yes, No} equals sigmoid(logit(Yes) − logit(No)), and dCE/dz = sigmoid(z) − y.
  2. On 2,000 random cases, the gradient of the gain term on z̄0 matches −λG · sigmoid((m − G)/s) · (2y − 1) / σ0, and z̄1 gets exactly zero gradient.
  3. Without the stop-gradient, the gradient on z̄1 would push it towards the wrong label. The implementation must not have this path.
  4. With z0 = k · z1 + c and exact statistics, G is zero on bags where the words are right.
  5. The clamp example (y = 1, words −1.0, real audio −0.7): G = −0.70 with the clamp and +0.30 without it; push strength 1.000 and 0.500.
  6. The soft hinge is never more than s · ln 2 above the hard hinge, with the largest gap at G = m.
  7. The gain push and the cross-entropy push as a bag grows confident.
- The pilot also needs these tests: the gain is computed in fp32 from fp16 inputs; λG ramps up over the first 20% of steps; the running statistics use momentum 0.9 from first and second moments; σ never falls below a quarter of its zero-shot value or 0.1; "Yes" and "No" are single tokens; and a score from the cache equals the score from the audio (GPU).

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
| `v0.1.0` | Pilot settings frozen (week 3). Every later run uses this tag or a later `0.1.x`. |
| `v0.1.x` | Fixes that change no method setting, for example in the analysis code |
| `v0.2.0` | Pilot report: aggregate results and the go / no-go decision (week 4) |
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
