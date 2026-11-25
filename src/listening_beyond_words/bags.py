"""The bag sampler: which segments train together, and in which step of which epoch.

A bag is 4 segments of one participant, and a step takes 2 bags of PHQ-8-positive and 2 of
PHQ-8-negative participants (pilot protocol, fixed settings "Segments and bags", and
section "Training", bullet "Bags and epochs"). An epoch is one pass of the sampler:

1. Each training participant with n segments offers 4·⌊min(n, 16)/4⌋ of them (8 in place
   of 16 under the fallback of check F5): those used least often so far, ties drawn at
   random, so that leftovers wait. They are shuffled and cut into bags of 4. A participant
   with fewer than 4 segments gives no bag.
2. Within each class, the bags are ranked by their participant's trained share: the bags
   trained over the bags offered in earlier epochs, 0 in epoch 1; lowest first, ties at
   random.
3. The epoch ends when a class has fewer than 2 bags left: with k steps, the first 2k bags
   of each class train. They are shuffled within their class, and step i takes positive
   bags 2i and 2i + 1 and negative bags 2i and 2i + 1, in the order positive, negative,
   positive, negative. Two bags of one participant may share a step.
4. A bag counts as trained, and its segments as used, once it is placed in a step.

A participant whose bags were left over thus has a lower share and is chosen earlier in
the next epoch, while the shuffle keeps the order of the steps from following earlier use.
The sampler takes nothing from the training itself, so a skipped step changes no bag. Each
fold-run has one random stream (``stream_seed``), from its round's seed and its fold, the
same for every arm, so B0, B1 and CPG of a fold-run see the same bags in the same order
(bullet "Who trains, and seeds"). Every epoch offers the same bags per participant, so
every epoch has the same number of steps, known before training starts.

The settings are ``bags`` in ``configs/pilot/data.yaml`` (``BagSettings``) and ``sampler``
in ``configs/pilot/train.yaml`` (``config.SamplerSettings``). The sampler is built from the
training participants of a fold-run only (``training_people``).
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Final

import numpy as np
import pandas as pd

from listening_beyond_words.config import SamplerSettings
from listening_beyond_words.runinfo import StepError, item_seed

STREAM: Final = "bags"  # the sampler's stream: item_seed(<round seed>, STREAM, <fold>)
POSITIVE: Final = 1  # the label of a PHQ-8-positive participant (total of 10 or more)
NEGATIVE: Final = 0
# The options of train.yaml's sampler that this code implements (read-only).
IMPLEMENTED: Final[Mapping[str, str]] = MappingProxyType(
    {
        "epoch_end": "a_class_runs_out",
        "draw": "least_trained_first",
        "bag_choice": "lowest_trained_share_first",
        "step_order": "shuffled",
    }
)
_BAG_KEYS: Final = ("segments_per_bag", "max_segments_per_epoch", "repeat_segments")


@dataclass(frozen=True, slots=True)
class BagSettings:
    """How segments fill bags (``bags`` in ``configs/pilot/data.yaml``).

    Attributes:
        segments_per_bag: Segments of one participant in a bag: 4.
        max_segments_per_epoch: The most segments that a participant offers in an epoch:
            16, or 8 under the fallback of check F5.
        repeat_segments: Whether a segment may repeat to fill a bag: never (whole bags
            only; leftovers wait).
    """

    segments_per_bag: int
    max_segments_per_epoch: int
    repeat_segments: bool

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "BagSettings":
        """Read the settings and check every key and value.

        Args:
            config: The data config, with the ``bags`` block.

        Returns:
            The settings.

        Raises:
            StepError: If the block is missing, lacks a key or has an unknown one, a count
                is not a whole number of at least 1, a participant could offer fewer
                segments than a bag holds, or segments may repeat, which the sampler does
                not do.
        """
        block = config.get("bags")
        if not isinstance(block, Mapping):
            raise StepError("bags must be a mapping in the data config")
        missing = sorted(set(_BAG_KEYS) - set(block))
        unknown = sorted(str(key) for key in set(block) - set(_BAG_KEYS))
        if missing or unknown:
            raise StepError(f"bags: missing keys {missing}, unknown keys {unknown}")
        per_bag = _count(block["segments_per_bag"], "bags.segments_per_bag", 1)
        most = _count(block["max_segments_per_epoch"], "bags.max_segments_per_epoch", per_bag)
        if block["repeat_segments"] is not False:
            raise StepError(
                "bags.repeat_segments must be false: whole bags only, so leftover segments"
                f" wait for a later epoch, not {block['repeat_segments']!r}"
            )
        return cls(segments_per_bag=per_bag, max_segments_per_epoch=most, repeat_segments=False)


def _count(value: Any, name: str, minimum: int) -> int:
    """A whole number of at least ``minimum`` from the config; true and false are refused."""
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise StepError(f"{name} must be a whole number of at least {minimum}, not {value!r}")
    return value


@dataclass(frozen=True, slots=True)
class Bag:
    """One bag: segments of one participant that go through the model together.

    Attributes:
        session: The participant's session ID.
        label: 1 if the participant is PHQ-8-positive, 0 if not.
        segments: The segment IDs, in the bag's order.
    """

    session: int
    label: int
    segments: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EpochPlan:
    """The bags of one epoch: those placed in its steps, in order, and those left over.

    Attributes:
        index: The epoch, counted from 1.
        steps: Each step's bags, in the order positive, negative, positive, negative.
        unused: The bags offered in the epoch but placed in no step, positive ones first;
            their segments count as unused and wait for a later epoch.
    """

    index: int
    steps: tuple[tuple[Bag, ...], ...]
    unused: tuple[Bag, ...]

    def bags(self) -> tuple[Bag, ...]:
        """Return the trained bags, step by step."""
        return tuple(bag for step in self.steps for bag in step)

    def counts(self) -> dict[str, int]:
        """Return the epoch's steps and its offered and unused bags of each class.

        Returns:
            Whole numbers only, for logs and ``run.json``: ``steps``, ``positive_offered``,
            ``negative_offered``, ``positive_unused`` and ``negative_unused``.
        """
        trained, unused = self.bags(), self.unused
        counts = {"steps": len(self.steps)}
        for name, label in (("positive", POSITIVE), ("negative", NEGATIVE)):
            left = sum(bag.label == label for bag in unused)
            counts[f"{name}_offered"] = sum(bag.label == label for bag in trained) + left
            counts[f"{name}_unused"] = left
        return counts


def stream_seed(round_seed: int, fold: int) -> int:
    """Return the seed of a fold-run's bag stream: ``item_seed(round_seed, STREAM, fold)``.

    One stream per fold-run, the same for every arm (pilot protocol, section "Training",
    bullet "Who trains, and seeds").

    Args:
        round_seed: The round's seed (``seeds`` in ``train.yaml``: 101 or 102).
        fold: The fold-run's test fold, from 1.

    Returns:
        The seed of ``numpy.random.default_rng``.
    """
    return item_seed(round_seed, STREAM, fold)


def training_people(partitions: pd.DataFrame, round_number: int, fold: int) -> tuple[int, ...]:
    """Return the training participants of a fold-run: those of the round's other folds.

    No part of a training fold is held out (pilot protocol, section "Training", bullet "Who
    trains, and seeds").

    Args:
        partitions: The pinned partitions (``clips.load_pinned_partitions``): columns
            ``session``, ``round`` and ``fold``.
        round_number: The round.
        fold: The fold-run's test fold.

    Returns:
        The training participants' session IDs, in ascending order.

    Raises:
        ValueError: If the partitions hold no such round, or no such fold in it.
    """
    # True would match round 1, since True == 1.
    in_round = partitions[partitions["round"] == round_number]
    if isinstance(round_number, bool) or in_round.empty:
        raise ValueError(f"the partitions hold no round {round_number!r}")
    folds = sorted({int(number) for number in in_round["fold"]})
    if isinstance(fold, bool) or fold not in folds:
        raise ValueError(f"round {round_number} holds folds {folds}, not {fold!r}")
    return tuple(
        sorted(int(session) for session in in_round.loc[in_round["fold"] != fold, "session"])
    )


@dataclass(slots=True)
class _Person:
    """One training participant as the sampler tracks it."""

    session: int
    label: int
    segments: tuple[str, ...]  # in ascending ID
    bags: int  # bags offered in every epoch
    uses: np.ndarray = field(init=False)  # how often each segment was placed in a step
    trained: int = 0  # bags placed in a step so far
    offered: int = 0  # bags offered so far

    def __post_init__(self) -> None:
        self.uses = np.zeros(len(self.segments), dtype=np.int64)

    def share(self) -> float:
        """The trained share: bags trained over bags offered in earlier epochs, 0 before."""
        return self.trained / self.offered if self.offered else 0.0


class BagSampler:
    """The bags of a fold-run, epoch by epoch (module docstring, points 1 to 4).

    The plan of each epoch follows from the inputs and the seed alone and is made once, in
    the order of the epochs, from one ``numpy.random.default_rng(seed)``. Its draws come in
    this order in each epoch: for each participant with a bag, in ascending session ID, one
    uniform number per segment breaks the ties of their use counts, then a permutation
    shuffles the offered segments; then, for the positive class and then for the negative
    class, one uniform number per bag breaks the ties of their shares, then a permutation
    shuffles the chosen bags. Asking for a later epoch first makes the earlier ones, so a
    4th epoch keeps the bags of the first three.

    Args:
        segments: The segments of the training participants, with the columns ``segment``
            (its ID) and ``session``, such as rows of a cache manifest; other columns are
            ignored, and the row order does not matter.
        labels: The label of each training participant (session ID to 1 or 0), from the
            pinned labels; a participant without a segment gives no bag.
        bags: How segments fill bags (``BagSettings``).
        sampler: The sampler's settings (``sampler`` in ``train.yaml``).
        seed: The seed of the fold-run's stream (``stream_seed``).

    Raises:
        StepError: If a segment is listed twice or belongs to a session without a training
            label, a label is not 0 or 1, or the participants give fewer bags of a class
            than one step takes.
        ValueError: If the settings name an option that the sampler does not implement, or
            the seed is not a whole number of at least 0.
    """

    def __init__(
        self,
        segments: pd.DataFrame,
        labels: Mapping[int, int] | pd.Series,
        bags: BagSettings,
        sampler: SamplerSettings,
        seed: int,
    ) -> None:
        for name, value in IMPLEMENTED.items():
            if getattr(sampler, name) != value:
                raise ValueError(
                    f"the sampler implements {name} {value!r}, not {getattr(sampler, name)!r}"
                )
        if bags.repeat_segments:
            raise ValueError("the sampler never repeats a segment (bags.repeat_segments)")
        if isinstance(seed, bool) or seed < 0:
            raise ValueError(f"the seed must be a whole number of at least 0, not {seed!r}")
        self._bags = bags
        self._per_step = {POSITIVE: sampler.positive_bags, NEGATIVE: sampler.negative_bags}
        self._people = _people(segments, labels, bags)
        self._by_session = {person.session: person for person in self._people}
        self._rng = np.random.default_rng(int(seed))
        self._plans: list[EpochPlan] = []
        offered = {
            label: sum(person.bags for person in self._people if person.label == label)
            for label in (POSITIVE, NEGATIVE)
        }
        self._offered = offered
        self.steps_per_epoch = min(offered[label] // self._per_step[label] for label in offered)
        if self.steps_per_epoch == 0:
            raise StepError(
                f"the training participants give {offered[POSITIVE]} positive and"
                f" {offered[NEGATIVE]} negative bags per epoch, fewer than one step's"
                f" {self._per_step[POSITIVE]} + {self._per_step[NEGATIVE]}"
            )

    def counts(self) -> dict[str, int]:
        """Return the fold-run's participants, bags and steps per epoch, as counts.

        Returns:
            Whole numbers only, for ``run.json``: the participants of each class, those
            without a bag (fewer segments than a bag holds), the bags that each class
            offers per epoch and the steps per epoch.
        """
        counts: dict[str, int] = {}
        for name, label in (("positive", POSITIVE), ("negative", NEGATIVE)):
            people = [person for person in self._people if person.label == label]
            counts[f"{name}_people"] = len(people)
            counts[f"{name}_people_without_a_bag"] = sum(person.bags == 0 for person in people)
            counts[f"{name}_bags_per_epoch"] = self._offered[label]
        counts["steps_per_epoch"] = self.steps_per_epoch
        return counts

    def total_steps(self, epochs: int) -> int:
        """Return T, the steps of a run of so many epochs.

        Args:
            epochs: The run's epochs (``epochs`` in ``train.yaml``).

        Returns:
            The epochs times the steps per epoch.

        Raises:
            ValueError: If ``epochs`` is not a whole number of at least 1.
        """
        if isinstance(epochs, bool) or epochs < 1:
            raise ValueError(f"a run has at least 1 epoch, not {epochs!r}")
        return epochs * self.steps_per_epoch

    def epoch(self, index: int) -> EpochPlan:
        """Return the plan of an epoch.

        Args:
            index: The epoch, counted from 1.

        Returns:
            Its steps and its unused bags.

        Raises:
            ValueError: If ``index`` is not a whole number of at least 1.
        """
        if isinstance(index, bool) or index < 1:
            raise ValueError(f"epochs are counted from 1, not {index!r}")
        while len(self._plans) < index:
            self._plans.append(self._plan_next())
        return self._plans[index - 1]

    def start_bags(self, steps: int) -> list[Bag]:
        """Return the bags of the first steps of epoch 1, which start the running statistics.

        The zero-shot scores of their segments start μ and σ (pilot protocol, section
        "Encoder caches and zero-shot scores", bullet "Start of the running statistics").

        Args:
            steps: The steps (``gain.start_steps`` in ``train.yaml``: 8, so 16 positive and
                16 negative bags).

        Returns:
            Their bags, step by step.

        Raises:
            ValueError: If ``steps`` is not a whole number of at least 1.
            StepError: If epoch 1 has fewer steps.
        """
        if isinstance(steps, bool) or steps < 1:
            raise ValueError(f"the start takes at least 1 step, not {steps!r}")
        plan = self.epoch(1)
        if steps > len(plan.steps):
            raise StepError(
                f"epoch 1 has {len(plan.steps)} steps, fewer than the {steps} whose bags start"
                " the running statistics"
            )
        return [bag for step in plan.steps[:steps] for bag in step]

    def _plan_next(self) -> EpochPlan:
        """Plan the next epoch and count its bags as trained and its segments as used."""
        rng, per_bag = self._rng, self._bags.segments_per_bag
        offered: dict[int, list[tuple[Bag, np.ndarray]]] = {POSITIVE: [], NEGATIVE: []}
        for person in self._people:
            if person.bags == 0:
                continue
            # The least used segments first, ties at random; then shuffled into bags.
            order = np.lexsort((rng.random(len(person.segments)), person.uses))
            picked = rng.permutation(order[: person.bags * per_bag])
            for start in range(0, len(picked), per_bag):
                indices = picked[start : start + per_bag]
                segments = tuple(person.segments[index] for index in indices)
                offered[person.label].append((Bag(person.session, person.label, segments), indices))
        chosen: dict[int, list[tuple[Bag, np.ndarray]]] = {}
        unused: list[Bag] = []
        for label in (POSITIVE, NEGATIVE):
            candidates = offered[label]
            shares = [self._by_session[bag.session].share() for bag, _ in candidates]
            ranked = np.lexsort((rng.random(len(candidates)), np.asarray(shares)))
            take = self.steps_per_epoch * self._per_step[label]
            chosen[label] = [candidates[index] for index in rng.permutation(ranked[:take])]
            unused += [candidates[index][0] for index in sorted(ranked[take:])]
        steps = []
        for number in range(self.steps_per_epoch):
            parts = {
                label: [
                    bag
                    for bag, _ in chosen[label][
                        number * self._per_step[label] : (number + 1) * self._per_step[label]
                    ]
                ]
                for label in (POSITIVE, NEGATIVE)
            }
            steps.append(tuple(_alternate(parts[POSITIVE], parts[NEGATIVE])))
        # Placed in a step: the bags count as trained and their segments as used.
        for bag, indices in chosen[POSITIVE] + chosen[NEGATIVE]:
            person = self._by_session[bag.session]
            person.trained += 1
            person.uses[indices] += 1
        for person in self._people:
            person.offered += person.bags
        return EpochPlan(index=len(self._plans) + 1, steps=tuple(steps), unused=tuple(unused))


def _alternate(first: Sequence[Bag], second: Sequence[Bag]) -> Iterable[Bag]:
    """The bags of two classes in turn, starting with the first; the rest of the longer last."""
    for index in range(max(len(first), len(second))):
        if index < len(first):
            yield first[index]
        if index < len(second):
            yield second[index]


def _people(
    segments: pd.DataFrame, labels: Mapping[int, int] | pd.Series, bags: BagSettings
) -> tuple[_Person, ...]:
    """The training participants, in ascending session ID, with their segments and bags."""
    if not {"segment", "session"} <= set(segments.columns):
        raise StepError("the segments need the columns segment and session")
    label_of: dict[int, int] = {}
    items: Iterable[tuple[Any, Any]] = labels.items()
    for session, label in items:
        valid = isinstance(label, int | np.integer) and not isinstance(label, bool | np.bool_)
        if not valid or label not in (POSITIVE, NEGATIVE):
            raise StepError(f"session {session}: the label must be 1 or 0, not {label!r}")
        label_of[int(session)] = int(label)
    by_session: dict[int, list[str]] = {session: [] for session in label_of}
    seen: set[str] = set()
    for segment, session in zip(segments["segment"], segments["session"], strict=True):
        name, number = str(segment), int(session)
        if name in seen:
            raise StepError(f"segment {name} is listed twice", segment=name)
        if number not in by_session:
            raise StepError(
                f"segment {name} belongs to session {number}, which has no training label: the"
                " sampler takes the training participants' segments only",
                segment=name,
            )
        seen.add(name)
        by_session[number].append(name)
    per_bag, most = bags.segments_per_bag, bags.max_segments_per_epoch
    return tuple(
        _Person(
            session=session,
            label=label_of[session],
            segments=tuple(sorted(by_session[session])),
            bags=min(len(by_session[session]), most) // per_bag,
        )
        for session in sorted(by_session)
    )
