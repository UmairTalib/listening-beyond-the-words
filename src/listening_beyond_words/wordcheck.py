"""The word check: check F4 and the checks of the re-voiced clip x1. So far, its sample.

The word check transcribes a sample of the segments: five per participant, drawn at
random, or all of a participant's segments if there are fewer (pilot protocol, section
"Clip versions"; ``word_check.sample`` in ``configs/pilot/data.yaml``). The sample is
drawn and pinned before any clip is transcribed.
"""

import numpy as np
import pandas as pd


def draw_sample(
    segments: pd.DataFrame, rng: np.random.Generator, per_participant: int
) -> pd.DataFrame:
    """Draw the segments that the word check transcribes.

    Participants are taken in ascending ID, and each participant's segments in ascending
    ID, so the draw depends only on the seed, never on the order of the rows.

    Args:
        segments: One row per segment with the columns ``segment`` and ``session``.
        rng: The generator, seeded from the config.
        per_participant: Segments drawn per participant, without replacement; all of a
            participant's segments if there are this many or fewer.

    Returns:
        The drawn rows with the columns ``segment`` and ``session``, sorted by segment.

    Raises:
        ValueError: If the count is not positive or a segment appears twice.
    """
    if per_participant < 1:
        raise ValueError(f"draw at least one segment per participant, not {per_participant}")
    if segments["segment"].duplicated().any():
        raise ValueError("a segment appears twice")
    table = segments[["segment", "session"]].sort_values(["session", "segment"], ignore_index=True)
    drawn: list[pd.DataFrame] = []
    for _, group in table.groupby("session", sort=True):
        if len(group) <= per_participant:
            drawn.append(group)
        else:
            chosen = rng.choice(len(group), size=per_participant, replace=False)
            drawn.append(group.iloc[np.sort(chosen)])
    return pd.concat(drawn).sort_values("segment", ignore_index=True)
