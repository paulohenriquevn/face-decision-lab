"""Leak-free splits. Frames of one person or one video never appear on both sides.

Unsloth's FastDecisionModel.split_holdout shuffles rows, and every frame is a row, so frames of the
same video would end up in training and in the holdout. Use these splits instead.
"""

import random

from .errors import SplitError
from .manifest import FaceSample


def group_split(
    samples: list[FaceSample], test_fraction: float, seed: int
) -> tuple[list[FaceSample], list[FaceSample]]:
    """Splits by subject_id; roughly test_fraction of the subjects go to the test side."""
    if not 0 < test_fraction < 1:
        raise SplitError(f"test_fraction must be between 0 and 1, got {test_fraction}")
    subjects = sorted({sample.subject_id for sample in samples})
    if len(subjects) < 2:
        raise SplitError(f"need at least 2 subjects to split, got {len(subjects)}")
    random.Random(seed).shuffle(subjects)
    test_count = min(len(subjects) - 1, max(1, round(len(subjects) * test_fraction)))
    test_subjects = set(subjects[:test_count])
    train = [sample for sample in samples if sample.subject_id not in test_subjects]
    test = [sample for sample in samples if sample.subject_id in test_subjects]
    _assert_no_shared_video(train, test)
    return train, test


def cross_dataset_split(
    samples: list[FaceSample], train_sources: set[str], test_sources: set[str]
) -> tuple[list[FaceSample], list[FaceSample]]:
    """Trains on some datasets and tests on others: the generalization protocol."""
    overlap = train_sources & test_sources
    if overlap:
        raise SplitError(f"datasets {sorted(overlap)} are on both sides")
    train = [sample for sample in samples if sample.source_dataset in train_sources]
    test = [sample for sample in samples if sample.source_dataset in test_sources]
    for side, chosen, sources in (("train", train, train_sources), ("test", test, test_sources)):
        if not chosen:
            raise SplitError(f"{side} side is empty for datasets {sorted(sources)}")
    return train, test


def _assert_no_shared_video(train: list[FaceSample], test: list[FaceSample]) -> None:
    shared = {sample.video_id for sample in train} & {sample.video_id for sample in test}
    if shared:
        raise SplitError(
            f"videos {sorted(shared)[:5]} hold frames of subjects on both sides; "
            "check that each video_id belongs to a single subject_id"
        )
