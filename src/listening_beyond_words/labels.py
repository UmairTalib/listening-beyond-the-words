"""PHQ-8 labels of the pilot sessions.

The label is a PHQ-8 total of 10 or more, recomputed from the total rather than read
from the binary column, which disagrees for 15 pilot sessions. Every label file also
holds rows of other sessions, among them the test sessions 600-718. The loader keeps
only the requested sessions right after reading and never returns other rows.
"""

from collections.abc import Collection
from pathlib import Path
from typing import Final

import pandas as pd

from listening_beyond_words.sessions import PILOT_IDS

PHQ8_CUTOFF: Final = 10
SPLIT_FILES: Final = {"train": "train_split.csv", "dev": "dev_split.csv"}
DETAILED_FILE: Final = "detailed_lables.csv"  # the file name as distributed
PHQ8_FILE: Final = "Detailed_PHQ8_Labels.csv"
ITEMS: Final = (
    "PHQ8_1_NoInterest",
    "PHQ8_2_Depressed",
    "PHQ8_3_Sleep",
    "PHQ8_4_Tired",
    "PHQ8_5_Appetite",
    "PHQ8_6_Failure",
    "PHQ8_7_Concentration",
    "PHQ8_8_Psychomotor",
)
GENDERS: Final = frozenset({"female", "male"})


def read_rows(path: Path, id_column: str, keep_ids: Collection[int]) -> pd.DataFrame:
    """Read a label file and keep only the rows of the given sessions.

    Args:
        path: The label file.
        id_column: The column with the session ID.
        keep_ids: The sessions to keep; every other row is dropped right away.

    Returns:
        The kept rows, indexed by session ID.

    Raises:
        ValueError: If a kept session appears more than once.
    """
    frame = pd.read_csv(path)
    ids = pd.to_numeric(frame[id_column]).astype(int)
    keep = ids.isin(set(keep_ids))
    kept = frame.loc[keep].copy()
    kept.index = pd.Index(ids[keep].to_numpy(), name="session")
    if kept.index.duplicated().any():
        raise ValueError(f"{path.name}: a session appears more than once")
    return kept


def load_labels(
    labels_dir: Path, keep_ids: Collection[int] = PILOT_IDS
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Return the labels of the requested sessions and counts that compare the files.

    The PHQ-8 total comes from the official split files and must equal the sum of
    the eight item scores in the detailed file. Gender comes from the split files,
    with stray spaces removed.

    Args:
        labels_dir: Folder with the E-DAIC label files.
        keep_ids: The sessions to load; by default the 189 pilot sessions.

    Returns:
        One row per session, sorted by ID, with the columns ``phq8_total``,
        ``label`` (1 if the total is at least 10), ``official_binary``, ``gender``,
        ``gender_detailed``, ``age`` and ``split``; and counts over all sessions.

    Raises:
        ValueError: If a session is missing from the split files or the detailed
            file, is in more than one split file, lacks an item score, has a total
            that differs from the sum of its items, or has an unknown gender code.
    """
    wanted = set(keep_ids)
    split = pd.concat(
        [
            read_rows(labels_dir / name, "Participant_ID", wanted).assign(split=part)
            for part, name in SPLIT_FILES.items()
        ]
    )
    if split.index.duplicated().any():
        raise ValueError("a session is in more than one split file")
    detailed = read_rows(labels_dir / DETAILED_FILE, "Participant", wanted)
    for name, frame in (("the split files", split), (DETAILED_FILE, detailed)):
        missing = wanted - set(frame.index)
        if missing:
            raise ValueError(f"{len(missing)} sessions have no row in {name}")
    items = detailed.loc[:, list(ITEMS)]
    if items.isna().any().any():
        raise ValueError(f"{DETAILED_FILE}: some PHQ-8 item scores are missing")
    total = split["PHQ_Score"].astype(int).sort_index()
    differs = int((items.sum(axis=1).reindex(total.index) != total).sum())
    if differs:
        raise ValueError(f"{differs} sessions: PHQ_Score differs from the sum of the items")
    gender = split["Gender"].astype(str).str.strip().str.lower().reindex(total.index)
    unknown = set(gender) - GENDERS
    if unknown:
        raise ValueError(f"unknown gender codes in the split files: {sorted(unknown)}")
    labels = pd.DataFrame(
        {
            "phq8_total": total,
            "label": (total >= PHQ8_CUTOFF).astype(int),
            "official_binary": split["PHQ_Binary"].astype(int).reindex(total.index),
            "gender": gender,
            "gender_detailed": detailed["gender"]
            .astype(str)
            .str.strip()
            .str.lower()
            .reindex(total.index),
            "age": detailed["age"].reindex(total.index),
            "split": split["split"].reindex(total.index),
        }
    )
    report = {
        "sessions": len(labels),
        "positive": int(labels["label"].sum()),
        "female": int((labels["gender"] == "female").sum()),
        "male": int((labels["gender"] == "male").sum()),
        "binary_column_disagrees": int((labels["label"] != labels["official_binary"]).sum()),
        "gender_files_disagree": int((labels["gender"] != labels["gender_detailed"]).sum()),
    }
    if (labels_dir / PHQ8_FILE).exists():
        other = read_rows(labels_dir / PHQ8_FILE, "Participant_ID", wanted)["PHQ_8Total"]
        other = other.reindex(total.index)
        report["phq8_file_total_differs"] = int((other != total).sum())
        report["phq8_file_label_differs"] = int(
            ((other >= PHQ8_CUTOFF).astype(int) != labels["label"]).sum()
        )
    return labels, report
