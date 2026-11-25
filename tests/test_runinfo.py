import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from listening_beyond_words.runinfo import (
    check_sha256,
    environment,
    git_state,
    item_seed,
    seed_everything,
    sha256_file,
    utc_stamp,
    write_json,
)

PINNED_SEED = 7691467536369641371  # item_seed(106, "9001_012", 2), computed 2026-11-22


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    path = tmp_path / "data.bin"
    path.write_bytes(b"synthetic bytes" * 1000)
    assert sha256_file(path) == hashlib.sha256(path.read_bytes()).hexdigest()


def test_check_sha256_stops_on_a_changed_file(tmp_path: Path) -> None:
    path = tmp_path / "data.bin"
    path.write_bytes(b"one")
    pinned = sha256_file(path)
    path.write_bytes(b"two")
    with pytest.raises(ValueError, match="does not match"):
        check_sha256(path, pinned)


def test_utc_stamp_suits_folder_names() -> None:
    stamp = utc_stamp()
    assert len(stamp) == 16
    assert stamp.endswith("Z")


def test_item_seed_is_pinned() -> None:
    # SHA-256 of '[106,"9001_012",2]', first 8 bytes, shifted right by one bit.
    digest = hashlib.sha256(b'[106,"9001_012",2]').digest()
    assert item_seed(106, "9001_012", 2) == int.from_bytes(digest[:8], "big") >> 1
    assert item_seed(106, "9001_012", 2) == PINNED_SEED


def test_item_seed_is_the_same_in_another_process() -> None:
    code = (
        "from listening_beyond_words.runinfo import item_seed; print(item_seed(106, '9001_012', 2))"
    )
    for hash_seed in ("1", "2"):
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "PYTHONHASHSEED": hash_seed},
        )
        assert int(result.stdout) == item_seed(106, "9001_012", 2)


def test_item_seed_differs_by_seed_key_order_and_key_type() -> None:
    seeds = {
        item_seed(106, "9001_012", 2),
        item_seed(107, "9001_012", 2),
        item_seed(106, "9001_012", 3),
        item_seed(106, 2, "9001_012"),
        item_seed(106, "9001_012", "2"),
        item_seed(106, "9001_012"),
    }
    assert len(seeds) == 6
    assert all(0 <= seed < 2**63 for seed in seeds)


def test_item_seed_takes_numpy_integers_like_python_integers() -> None:
    assert item_seed(np.int64(106), "9001_012", np.int32(2)) == item_seed(106, "9001_012", 2)


@pytest.mark.parametrize("key", [2.0, True, None, b"9001"])
def test_item_seed_refuses_other_keys(key: object) -> None:
    with pytest.raises(TypeError, match="whole numbers or text"):
        item_seed(106, key)  # type: ignore[arg-type]


def test_write_json_writes_numpy_numbers_as_plain_numbers(tmp_path: Path) -> None:
    data = {
        "count": np.int64(5883),
        "share": np.float32(0.25),
        "limited": np.bool_(False),
        "parts": np.array([1, 2]),
        "nested": {"total": np.arange(3).sum()},
    }
    write_json(tmp_path / "run.json", data)  # type: ignore[arg-type]
    assert json.loads((tmp_path / "run.json").read_text(encoding="utf-8")) == {
        "count": 5883,
        "share": 0.25,
        "limited": False,
        "parts": [1, 2],
        "nested": {"total": 3},
    }


def test_write_json_still_refuses_other_objects(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="cannot write"):
        write_json(tmp_path / "run.json", {"bad": object()})


def test_environment_records_versions_and_the_lock_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "uv.lock").write_text("lock", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    found = environment(["numpy", "no-such-package-9001"])
    assert found["packages"] == {"numpy": np.__version__, "no-such-package-9001": None}
    assert found["uv_lock_sha256"] == hashlib.sha256(b"lock").hexdigest()
    assert str(found["python"]).startswith("3.12")
    (tmp_path / "uv.lock").unlink()
    assert environment([])["uv_lock_sha256"] is None


def test_error_record_lists_frames_and_types_of_the_whole_chain_but_no_message() -> None:
    from listening_beyond_words.runinfo import StepError, error_record

    def inner() -> None:
        raise KeyError("SYNTHETICWORD")

    def outer() -> None:
        try:
            inner()
        except KeyError:
            raise StepError("segment 300_001, part 2: KeyError") from None

    try:
        outer()
    except StepError as error:
        record = error_record(error)
    assert "SYNTHETICWORD" not in record
    assert "segment 300_001" not in record  # not even a message of the step's own
    lines = record.splitlines()
    assert lines[0] == "listening_beyond_words.runinfo.StepError"
    assert "builtins.KeyError" in lines
    assert any(line.endswith(" in inner") for line in lines)
    assert any(line.endswith(" in outer") for line in lines)
    assert all(line.startswith("  ") or "." in line for line in lines)


def test_slurms_stop_ends_a_step_through_its_error_path() -> None:
    import signal

    from listening_beyond_words.runinfo import STOPPED, stop_on_sigterm

    previous = signal.getsignal(signal.SIGTERM)
    try:
        stop_on_sigterm()
        assert signal.getsignal(signal.SIGTERM) is not previous  # else the signal ends pytest
        with pytest.raises(SystemExit) as stopped:
            os.kill(os.getpid(), signal.SIGTERM)
        assert stopped.value.code == STOPPED
    finally:
        signal.signal(signal.SIGTERM, previous)


def test_write_json_leaves_the_earlier_file_or_none_but_never_half_of_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "run.json"
    write_json(path, {"status": "running"})
    real = Path.write_text

    def dying(self: Path, data: str, *args: object, **kwargs: object) -> int:
        real(self, data[: len(data) // 3], *args, **kwargs)  # type: ignore[arg-type]
        raise KeyboardInterrupt

    monkeypatch.setattr(Path, "write_text", dying)
    with pytest.raises(KeyboardInterrupt):
        write_json(path, {"status": "done", "counts": list(range(100))})
    monkeypatch.undo()
    assert json.loads(path.read_text(encoding="utf-8")) == {"status": "running"}
    write_json(path, {"status": "done"})
    assert json.loads(path.read_text(encoding="utf-8")) == {"status": "done"}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["run.json"]


def test_a_failure_names_its_errors_log_or_hands_its_frames_to_stderr(tmp_path: Path) -> None:
    from listening_beyond_words.runinfo import Progress, StepError, error_record, record_failure

    def raised(error: BaseException) -> BaseException:
        try:
            raise error
        except BaseException as caught:
            return caught

    secret = raised(KeyError("SYNTHETICWORD"))
    folder = tmp_path / "run"
    folder.mkdir()
    info: dict[str, object] = {"status": "running"}
    logged = record_failure(secret, Progress(run_dir=folder, info=info, segment="300_001"))
    assert (
        logged.line
        == f"FAILED: KeyError in segment 300_001; traceback frames in {folder / 'errors.log'}"
    )
    assert logged.frames is None  # errors.log holds them
    assert (folder / "errors.log").read_text(encoding="utf-8") == error_record(secret)
    stored = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    assert stored["status"] == "failed"
    assert stored["error"] == {"type": "KeyError", "segment": "300_001", "part": None}
    # No folder, or one that cannot be written: the frames go to stderr, never the message.
    for progress in (Progress(), Progress(run_dir=tmp_path / "missing", info={})):
        unlogged = record_failure(secret, progress)
        assert unlogged.line == "FAILED: KeyError"
        assert unlogged.frames == error_record(secret)
        assert "SYNTHETICWORD" not in unlogged.line + unlogged.frames
    own = record_failure(raised(StepError("the check run did not pass")), Progress())
    assert own.line == "FAILED: StepError: the check run did not pass"
    assert own.frames is None  # the step's own message says what happened
    foreign = raised(KeyError("x"))
    foreign.segment = "SYNTHETICWORD"  # type: ignore[attr-defined]
    foreign.part = "SYNTHETICWORD"  # type: ignore[attr-defined]
    assert record_failure(foreign, Progress(segment="300_002")).line == (
        "FAILED: KeyError in segment 300_002"
    )


def test_slurms_stop_is_recorded_by_its_fixed_text_only(tmp_path: Path) -> None:
    import numpy as np

    from listening_beyond_words.runinfo import STOPPED, Progress, record_failure

    stop = SystemExit("cannot read SYNTHETICWORD")
    stop.code = STOPPED  # code says Slurm's stop, but the error's own text holds a word
    info: dict[str, object] = {}
    line = record_failure(stop, Progress(run_dir=tmp_path, info=info)).line
    assert line.startswith(f"FAILED: SystemExit: {STOPPED}; traceback frames in")
    stored = json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))
    assert stored["error"]["message"] == STOPPED
    odd = SystemExit(np.array(["SYNTHETICWORD", "x"]))  # a code whose comparison is an array
    assert record_failure(odd, Progress()).line == "FAILED: SystemExit"


def test_a_failure_is_recorded_in_run_json_even_when_errors_log_cannot_be_written(
    tmp_path: Path,
) -> None:
    from listening_beyond_words.runinfo import Progress, record_failure

    (tmp_path / "errors.log").mkdir()  # a folder in its place: the write fails
    info: dict[str, object] = {"status": "running"}
    failure = record_failure(KeyError("SYNTHETICWORD"), Progress(run_dir=tmp_path, info=info))
    assert failure.line == "FAILED: KeyError"
    assert failure.frames is not None  # so the frames go to stderr instead
    assert "SYNTHETICWORD" not in failure.frames
    stored = json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))
    assert stored["status"] == "failed"


def test_error_record_never_asks_an_exception_for_its_truth_value() -> None:
    from listening_beyond_words.runinfo import error_record

    class OddError(Exception):
        def __bool__(self) -> bool:
            raise ValueError("SYNTHETICWORD")

    inner = OddError("SYNTHETICWORD")
    outer = KeyError("SYNTHETICWORD")
    outer.__context__ = inner
    record = error_record(outer)
    assert record.splitlines()[0] == "builtins.KeyError"
    assert record.splitlines()[1].endswith("OddError")
    assert "SYNTHETICWORD" not in record


def test_quiet_unraisable_names_the_type_only(capsys: pytest.CaptureFixture[str]) -> None:
    from types import SimpleNamespace

    from listening_beyond_words.runinfo import quiet_unraisable

    quiet_unraisable(SimpleNamespace(exc_value=ValueError("SYNTHETICWORD"), object="SYNTHETICWORD"))
    out, err = capsys.readouterr()
    assert (out, err) == ("", "Exception ignored: ValueError\n")


def test_deterministic_torch_asks_for_repeatable_kernels_without_autotuning() -> None:
    import torch

    from listening_beyond_words.runinfo import deterministic_torch

    saved = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
    )
    try:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        deterministic_torch()
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.is_deterministic_algorithms_warn_only_enabled()
        assert torch.backends.cudnn.deterministic
        assert not torch.backends.cudnn.benchmark
    finally:
        torch.use_deterministic_algorithms(saved[0], warn_only=saved[1])
        torch.backends.cudnn.deterministic = saved[2]
        torch.backends.cudnn.benchmark = saved[3]


# Seeds --------------------------------------------------------------------------------------


def draws() -> tuple[float, float, float]:
    """One draw from Python's, NumPy's global and PyTorch's random number generators."""
    import torch

    return random.random(), float(np.random.random()), float(torch.rand(1))  # noqa: NPY002


def seeded_draws(*seeds: int) -> list[tuple[float, float, float]]:
    """The draws after each seed; every generator is put back as it was afterwards."""
    import torch

    python, numpy = random.getstate(), np.random.get_state()  # noqa: NPY002
    with torch.random.fork_rng(devices=[]):
        try:
            found = []
            for seed in seeds:
                seed_everything(seed)
                found.append(draws())
            return found
        finally:
            random.setstate(python)
            np.random.set_state(numpy)  # noqa: NPY002


def test_seed_everything_repeats_the_draws_of_python_numpy_and_pytorch() -> None:
    seed = item_seed(101, "torch", 1)
    first, again, other = seeded_draws(seed, seed, item_seed(101, "torch", 2))
    assert first == again
    assert all(one != two for one, two in zip(first, other, strict=True))


def test_seed_everything_gives_numpy_all_64_bits_of_the_seed() -> None:
    low, high = seeded_draws(7, 2**32 + 7)
    assert low[1] != high[1]  # NumPy's global state takes the seed as two 32-bit words
    assert seeded_draws(2**64 - 1)[0] != seeded_draws(0)[0]


@pytest.mark.parametrize(
    ("seed", "error"),
    [(True, TypeError), (1.5, TypeError), ("7", TypeError), (-1, ValueError), (2**64, ValueError)],
)
def test_seed_everything_refuses_what_is_no_seed(seed: object, error: type[Exception]) -> None:
    with pytest.raises(error):
        seed_everything(seed)  # type: ignore[arg-type]


# The checkout --------------------------------------------------------------------------------


def git(folder: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=folder, check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def clone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A throwaway git repository with one commit, as the working folder, with no outside config."""
    settings = {
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_AUTHOR_NAME": "Test",
        "GIT_AUTHOR_EMAIL": "test@example.invalid",
        "GIT_COMMITTER_NAME": "Test",
        "GIT_COMMITTER_EMAIL": "test@example.invalid",
    }
    for name, value in settings.items():
        monkeypatch.setenv(name, value)
    folder = tmp_path / "clone"
    folder.mkdir()
    git(folder, "init", "--quiet")
    (folder / "file.txt").write_text("one", encoding="utf-8")
    git(folder, "add", "file.txt")
    git(folder, "commit", "--quiet", "-m", "first")
    monkeypatch.chdir(folder)
    return folder


def commit(folder: Path, text: str) -> str:
    (folder / "file.txt").write_text(text, encoding="utf-8")
    git(folder, "commit", "--quiet", "-am", text)
    return git(folder, "rev-parse", "HEAD")


def test_git_state_describes_a_checkout_without_tags_by_its_short_commit(clone: Path) -> None:
    state = git_state()
    assert state == {
        "commit": git(clone, "rev-parse", "HEAD"),
        "dirty": False,
        "describe": git(clone, "rev-parse", "--short", "HEAD"),
        "tags": [],
        "release_tags": [],
    }


def test_git_state_names_the_tags_at_the_commit_and_every_pilot_release(clone: Path) -> None:
    git(clone, "tag", "-a", "v0.1.0", "-m", "Pilot settings frozen")
    git(clone, "tag", "smoke")
    git(clone, "tag", "v0.2.0")
    state = git_state()
    assert state["tags"] == ["smoke", "v0.1.0", "v0.2.0"]
    assert state["release_tags"] == ["v0.1.0"]  # the pilot's releases only


def test_git_state_describes_commits_after_a_tag_and_a_dirty_checkout(clone: Path) -> None:
    first = git(clone, "rev-parse", "HEAD")
    commit(clone, "two")
    git(clone, "tag", "-a", "v0.1.0", "-m", "Pilot settings frozen")
    assert git_state()["describe"] == "v0.1.0"
    later = commit(clone, "three")
    state = git_state()
    assert state["describe"] == f"v0.1.0-1-g{git(clone, 'rev-parse', '--short', 'HEAD')}"
    assert (state["commit"], state["tags"], state["release_tags"]) == (later, [], ["v0.1.0"])
    (clone / "file.txt").write_text("changed", encoding="utf-8")
    state = git_state()
    assert state["dirty"] is True
    assert str(state["describe"]).endswith("-dirty")
    git(clone, "checkout", "--quiet", "--", "file.txt")
    git(clone, "checkout", "--quiet", first)  # a commit older than the tag ...
    state = git_state()
    assert state["release_tags"] == ["v0.1.0"]  # ... still sees it
    assert (state["tags"], state["dirty"]) == ([], False)
    assert state["describe"] == git(clone, "rev-parse", "--short", "HEAD")
