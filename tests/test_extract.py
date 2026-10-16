import pytest

from face_decision_lab.extract import frame_indices


def test_frame_indices_are_evenly_spaced_and_centered():
    assert frame_indices(total_frames=100, count=4) == [12, 37, 62, 87]


def test_frame_indices_never_exceed_the_video_length():
    assert frame_indices(total_frames=3, count=8) == [0, 1, 2]


def test_frame_indices_reject_an_empty_video():
    with pytest.raises(ValueError, match="0 frames"):
        frame_indices(total_frames=0, count=8)
