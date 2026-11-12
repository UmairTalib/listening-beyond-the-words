"""Partitions: participants split into folds, never segments.

Each round is one 5-fold split of the participants with its own seed (pilot: 101
and 102), stratified by label and gender, so every fold holds about the same
share of PHQ-8-positive participants and of women.
"""

from collections.abc import Sequence
from typing import Final

import pandas as pd
from sklearn.model_selection import StratifiedKFold

SEEDS: Final = (101, 102)
N_FOLDS: Final = 5


def make_partitions(
    labels: pd.DataFrame, seeds: Sequence[int] = SEEDS, n_folds: int = N_FOLDS
) -> pd.DataFrame:
    """Assign every participant to one fold in every round.

    Args:
        labels: One row per participant, indexed by session ID, with the columns
            ``label`` and ``gender``.
        seeds: One seed per round; round r uses ``seeds[r - 1]``.
        n_folds: Folds per round.

    Returns:
        One row per participant and round with the columns ``session``, ``round``
        and ``fold`` (1-based), sorted by round and session.
    """
    strata = labels["label"].astype(str) + "_" + labels["gender"].astype(str)
    sessions = labels.index.to_numpy()
    rows = []
    for round_number, seed in enumerate(seeds, start=1):
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for fold_number, (_, test) in enumerate(splitter.split(sessions, strata), start=1):
            rows += [
                {"session": int(session), "round": round_number, "fold": fold_number}
                for session in sessions[test]
            ]
    return pd.DataFrame(rows).sort_values(["round", "session"]).reset_index(drop=True)
