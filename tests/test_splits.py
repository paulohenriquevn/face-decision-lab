import pytest

from face_decision_lab.errors import SplitError
from face_decision_lab.splits import capped, cross_dataset_split, group_split, standard_split


def _frames(make_sample, subjects, frames_per_subject=3):
    return [
        make_sample(subject=subject, video=f"{subject}_video", path=f"{subject}_{frame}.png")
        for subject in subjects
        for frame in range(frames_per_subject)
    ]


def test_group_split_never_puts_one_subject_on_both_sides(make_sample):
    samples = _frames(make_sample, [f"s{i}" for i in range(20)])

    train, test = group_split(samples, test_fraction=0.25, seed=7)

    assert {s.subject_id for s in train}.isdisjoint({s.subject_id for s in test})


def test_group_split_keeps_every_sample(make_sample):
    samples = _frames(make_sample, [f"s{i}" for i in range(20)])

    train, test = group_split(samples, test_fraction=0.25, seed=7)

    assert sorted(train + test, key=lambda s: s.image_path) == sorted(samples, key=lambda s: s.image_path)


def test_group_split_is_reproducible_with_the_same_seed(make_sample):
    samples = _frames(make_sample, [f"s{i}" for i in range(20)])

    assert group_split(samples, 0.25, seed=7) == group_split(samples, 0.25, seed=7)


def test_group_split_rejects_a_video_shared_by_two_subjects(make_sample):
    samples = [make_sample(subject="s1", video="shared"), make_sample(subject="s2", video="shared")]

    with pytest.raises(SplitError, match="hold frames of subjects on both sides"):
        group_split(samples, test_fraction=0.5, seed=0)


def test_group_split_rejects_a_single_subject(make_sample):
    samples = _frames(make_sample, ["s1"])

    with pytest.raises(SplitError, match="at least 2 subjects"):
        group_split(samples, test_fraction=0.5, seed=0)


def test_cross_dataset_split_tests_only_on_unseen_datasets(make_sample):
    samples = [make_sample(source="ffpp"), make_sample(source="celebdf"), make_sample(source="df40")]

    train, test = cross_dataset_split(samples, {"ffpp"}, {"celebdf", "df40"})

    assert {s.source_dataset for s in train} == {"ffpp"}
    assert {s.source_dataset for s in test} == {"celebdf", "df40"}


def test_cross_dataset_split_rejects_a_dataset_on_both_sides(make_sample):
    samples = [make_sample(source="ffpp")]

    with pytest.raises(SplitError, match=r"\['ffpp'\] are on both sides"):
        cross_dataset_split(samples, {"ffpp"}, {"ffpp"})


def test_cross_dataset_split_rejects_an_empty_side(make_sample):
    samples = [make_sample(source="ffpp")]

    with pytest.raises(SplitError, match="test side is empty"):
        cross_dataset_split(samples, {"ffpp"}, {"celebdf"})


def test_standard_split_keeps_each_subject_on_one_side(make_sample):
    samples = _frames(make_sample, [f"s{i}" for i in range(40)])

    train, calibration, test = standard_split(samples, 0.2, 0.1, seed=3)

    sides = [{s.subject_id for s in side} for side in (train, calibration, test)]
    assert sides[0].isdisjoint(sides[1]) and sides[0].isdisjoint(sides[2]) and sides[1].isdisjoint(sides[2])


def test_standard_split_is_the_same_on_every_call(make_sample):
    samples = _frames(make_sample, [f"s{i}" for i in range(40)])

    assert standard_split(samples, 0.2, 0.1, seed=3) == standard_split(samples, 0.2, 0.1, seed=3)


def test_capped_returns_the_requested_count_in_original_order(make_sample):
    samples = _frames(make_sample, [f"s{i}" for i in range(10)])

    subset = capped(samples, 7, seed=1)

    assert len(subset) == 7
    assert subset == [s for s in samples if s in subset]


def test_capped_is_reproducible_with_the_same_seed(make_sample):
    samples = _frames(make_sample, [f"s{i}" for i in range(10)])

    assert capped(samples, 7, seed=1) == capped(samples, 7, seed=1)


def test_capped_keeps_everything_without_a_cap(make_sample):
    samples = _frames(make_sample, ["s1", "s2"])

    assert capped(samples, None, seed=1) == samples
