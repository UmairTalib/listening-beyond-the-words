"""The word-check sample draw (scripts/draw_wordcheck_sample.py) on a synthetic workspace."""

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
import yaml

from listening_beyond_words.runinfo import sha256_file
from listening_beyond_words.wordcheck import draw_sample

REPO = Path(__file__).parents[1]


def load_script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"script_{name}", REPO / "scripts" / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def draw(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[ModuleType, Path, pd.DataFrame]:
    root = tmp_path / "workspace"
    run = root / "derived" / "segments" / "run-1"
    run.mkdir(parents=True)
    table = pd.DataFrame(
        [
            {"segment": f"{session}_{index:03d}", "session": session, "text": "never printed"}
            for session, count in {9001: 3, 9002: 4, 9003: 12}.items()
            for index in range(count)
        ]
    )
    table.to_parquet(run / "segments.parquet", index=False)
    real = yaml.safe_load((REPO / "configs" / "pilot" / "data.yaml").read_text(encoding="utf-8"))
    built = {
        "inputs": {"word_times": {"sha256": "w" * 64}, "participant_lines": {"sha256": "l" * 64}},
        "settings": {"daicwoz_cut": real["daicwoz_cut"], "x0_cut": real["x0_cut"]},
    }
    (run / "run.json").write_text(json.dumps(built), encoding="utf-8")
    config = {
        **real,
        "segments": {
            "run": "derived/segments/run-1",
            "sha256": sha256_file(run / "segments.parquet"),
        },
        "word_times": {"run": "derived/word_times/run-0", "sha256": "w" * 64},
        "participant_lines": {"run": "derived/participant_lines/run-0", "sha256": "l" * 64},
    }
    config_path = tmp_path / "data.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    module = load_script("draw_wordcheck_sample")
    monkeypatch.setattr(module, "CONFIG", config_path)
    monkeypatch.setattr(module, "git_state", lambda: {"commit": "0" * 40, "dirty": False})
    monkeypatch.setenv("LBW_WORKSPACE", str(root))
    return module, root, table


def test_the_draw_writes_and_reports_the_pinned_sample(
    draw: tuple[ModuleType, Path, pd.DataFrame], capsys: pytest.CaptureFixture[str]
) -> None:
    module, root, table = draw
    module.main()
    (run,) = (root / "derived" / "wordcheck_sample").iterdir()
    sample = pd.read_parquet(run / "sample.parquet")
    expected = draw_sample(table, np.random.default_rng(105), 5)
    pd.testing.assert_frame_equal(sample, expected)
    info = json.loads((run / "run.json").read_text(encoding="utf-8"))
    assert info["status"] == "done"
    assert info["seeds"] == {"sample": 105}
    assert info["counts"] == {
        "segments_in_table": 19,
        "participants": 3,
        "segments": 3 + 4 + 5,
        "participants_with_fewer_than_the_count": 2,
        "participants_with_exactly_the_count": 0,
        "segments_per_participant": {"3": 1, "4": 1, "5": 1},
    }
    assert info["output"]["sha256"] == sha256_file(run / "sample.parquet")
    out = capsys.readouterr().out
    assert "12 segments from 3 participants, drawn with seed 105 from 19 segments" in out
    assert "fewer than 5 segments (all taken): 2; with exactly 5: 0" in out
    assert f"sha256: {info['output']['sha256']}" in out
    assert "never printed" not in out
    assert "9003_" not in out


def test_the_draw_refuses_a_checkout_with_uncommitted_changes(
    draw: tuple[ModuleType, Path, pd.DataFrame], monkeypatch: pytest.MonkeyPatch
) -> None:
    module, root, _ = draw
    monkeypatch.setattr(module, "git_state", lambda: {"commit": "0" * 40, "dirty": True})
    with pytest.raises(SystemExit, match="uncommitted changes"):
        module.main()
    assert not (root / "derived" / "wordcheck_sample").exists()
