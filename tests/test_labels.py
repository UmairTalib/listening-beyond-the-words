from pathlib import Path

import pandas as pd
import pytest

from listening_beyond_words.labels import ITEMS, load_labels

KEEP = (9001, 9002, 9003, 9004)
OTHER = 9600  # stands for a session outside the requested ones


def write_files(folder: Path, totals: dict[int, int], binary: dict[int, int]) -> Path:
    """Synthetic label files: items that sum to each total, one extra session everywhere."""
    sessions = [*totals, OTHER]
    every = {**totals, OTHER: 20}

    def split_rows(ids: list[int]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Participant_ID": ids,
                "Gender": ["female " if i % 2 else "male" for i in ids],
                "PHQ_Binary": [binary.get(i, 1) for i in ids],
                "PHQ_Score": [every[i] for i in ids],
                "PCL-C (PTSD)": 0,
                "PTSD Severity": 0,
            }
        )

    split_rows([9001, 9002, OTHER]).to_csv(folder / "train_split.csv", index=False)
    split_rows([9003, 9004]).to_csv(folder / "dev_split.csv", index=False)
    items = {item: [0] * len(sessions) for item in ITEMS}
    for row, session in enumerate(sessions):
        remaining = every[session]
        for item in ITEMS:
            items[item][row] = min(3, remaining)
            remaining -= items[item][row]
    detailed = pd.DataFrame({"Participant": sessions, **items})
    detailed["gender"] = ["female" if i % 2 else "male" for i in sessions]
    detailed["age"] = 40
    detailed.to_csv(folder / "detailed_lables.csv", index=False)
    return folder


def test_load_labels_recomputes_the_label_from_the_total(tmp_path: Path) -> None:
    folder = write_files(tmp_path, {9001: 9, 9002: 10, 9003: 15, 9004: 0}, {9002: 0, 9004: 0})
    labels, report = load_labels(folder, KEEP)
    assert labels["label"].to_dict() == {9001: 0, 9002: 1, 9003: 1, 9004: 0}
    assert report["positive"] == 2
    assert report["binary_column_disagrees"] == 2  # 9001 (binary 1, total 9) and 9002


def test_load_labels_keeps_only_the_requested_sessions(tmp_path: Path) -> None:
    folder = write_files(tmp_path, {9001: 9, 9002: 10, 9003: 15, 9004: 0}, {})
    labels, report = load_labels(folder, KEEP)
    assert list(labels.index) == list(KEEP)
    assert report["sessions"] == 4


def test_load_labels_strips_gender_codes_and_counts_disagreements(tmp_path: Path) -> None:
    folder = write_files(tmp_path, {9001: 9, 9002: 10, 9003: 15, 9004: 0}, {})
    labels, report = load_labels(folder, KEEP)
    assert set(labels["gender"]) == {"female", "male"}
    assert report["gender_files_disagree"] == 0


def test_load_labels_refuses_a_total_that_differs_from_the_items(tmp_path: Path) -> None:
    folder = write_files(tmp_path, {9001: 9, 9002: 10, 9003: 15, 9004: 0}, {})
    split = pd.read_csv(folder / "dev_split.csv")
    split.loc[split["Participant_ID"] == 9003, "PHQ_Score"] = 16
    split.to_csv(folder / "dev_split.csv", index=False)
    with pytest.raises(ValueError, match="differs from the sum of the items"):
        load_labels(folder, KEEP)


def test_load_labels_refuses_a_session_without_a_split_row(tmp_path: Path) -> None:
    folder = write_files(tmp_path, {9001: 9, 9002: 10, 9003: 15, 9004: 0}, {})
    with pytest.raises(ValueError, match="no row in the split files"):
        load_labels(folder, (*KEEP, 9005))
