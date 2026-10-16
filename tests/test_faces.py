import numpy as np
import pytest

from face_decision_lab.faces import (
    Box,
    CropReport,
    Outcome,
    encode_jpeg,
    frame_indices,
    letterbox,
    pick_face,
    square_crop,
)


def test_pick_face_reports_no_face_for_an_empty_detection():
    assert pick_face([]) == (None, Outcome.NO_FACE)


def test_pick_face_keeps_a_dominant_face_over_a_small_background_face():
    main, background = Box(0, 0, 100, 100), Box(200, 0, 30, 30)

    assert pick_face([background, main]) == (main, Outcome.SAVED)


def test_pick_face_rejects_two_faces_of_similar_size():
    assert pick_face([Box(0, 0, 100, 100), Box(200, 0, 90, 90)]) == (None, Outcome.AMBIGUOUS)


def test_square_crop_has_the_requested_size():
    image = np.full((300, 200, 3), 255, dtype=np.uint8)

    crop = square_crop(image, Box(50, 50, 80, 120), size=64)

    assert crop.shape == (64, 64, 3)


def test_square_crop_pads_with_black_past_the_image_border():
    image = np.full((100, 100, 3), 255, dtype=np.uint8)

    # A face in the top-left corner: the square reaches outside the image on the top and left.
    crop = square_crop(image, Box(0, 0, 40, 40), margin=0.5, size=80)

    assert crop[0, 0].tolist() == [0, 0, 0]
    assert crop[79, 79].tolist() == [255, 255, 255]


def test_crops_of_different_source_sizes_come_out_identical_in_shape():
    small, large = np.zeros((260, 360, 3), np.uint8), np.zeros((512, 512, 3), np.uint8)

    shapes = {square_crop(img, Box(50, 50, 100, 100)).shape for img in (small, large)}

    assert shapes == {(256, 256, 3)}


def test_encode_jpeg_produces_a_jpeg_file():
    assert encode_jpeg(np.zeros((32, 32, 3), np.uint8))[:2] == b"\xff\xd8"


def test_crop_report_computes_the_saved_share_per_generator():
    report = CropReport()
    for outcome in (Outcome.SAVED, Outcome.SAVED, Outcome.SAVED, Outcome.NO_FACE):
        report.add("insight", outcome)

    assert report.saved_share("insight") == 0.75


def test_frame_indices_are_evenly_spaced_and_centered():
    assert frame_indices(total_frames=100, count=4) == [12, 37, 62, 87]


def test_frame_indices_never_exceed_the_video_length():
    assert frame_indices(total_frames=3, count=8) == [0, 1, 2]


def test_frame_indices_reject_an_empty_video():
    with pytest.raises(ValueError, match="0 frames"):
        frame_indices(total_frames=0, count=8)


def test_letterbox_gives_every_image_the_same_shape():
    shapes = {letterbox(np.zeros(size + (3,), np.uint8), 640)[0].shape for size in [(260, 360), (1200, 800)]}

    assert shapes == {(640, 640, 3)}


def test_letterbox_shrinks_by_the_longest_side():
    _, scale = letterbox(np.zeros((1200, 800, 3), np.uint8), 640)

    assert scale == 640 / 1200


def test_letterbox_never_enlarges_a_small_image():
    canvas, scale = letterbox(np.full((100, 50, 3), 7, np.uint8), 640)

    assert scale == 1.0
    assert canvas[99, 49].tolist() == [7, 7, 7]
    assert canvas[100, 50].tolist() == [0, 0, 0]
