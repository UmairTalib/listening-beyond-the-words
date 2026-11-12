import numpy as np
import pandas as pd

from listening_beyond_words.partitions import make_partitions


def synthetic_labels(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(9006)
    sessions = pd.Index(range(9001, 9001 + n), name="session")
    return pd.DataFrame(
        {
            "label": rng.permutation([1] * (n // 3) + [0] * (n - n // 3)),
            "gender": rng.permutation(["female"] * (n // 2) + ["male"] * (n - n // 2)),
        },
        index=sessions,
    )


def test_every_participant_is_in_exactly_one_fold_per_round() -> None:
    parts = make_partitions(synthetic_labels())
    assert parts.groupby("round")["session"].apply(lambda s: s.is_unique and len(s) == 60).all()
    assert sorted(parts["fold"].unique()) == [1, 2, 3, 4, 5]


def test_folds_are_balanced_by_label() -> None:
    labels = synthetic_labels()
    parts = make_partitions(labels).join(labels, on="session")
    positives = parts.groupby(["round", "fold"])["label"].sum()
    assert positives.max() - positives.min() <= 2  # leftovers of the label x gender strata


def test_partitions_repeat_with_the_same_seed_and_differ_between_rounds() -> None:
    labels = synthetic_labels()
    first, again = make_partitions(labels), make_partitions(labels)
    assert first.equals(again)
    round_1 = first[first["round"] == 1].set_index("session")["fold"]
    round_2 = first[first["round"] == 2].set_index("session")["fold"]
    assert not round_1.equals(round_2)
