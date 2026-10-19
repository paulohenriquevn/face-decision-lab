import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from face_decision_lab.faces import (
    Box,
    CropReport,
    Outcome,
    RetinaFaceDetector,
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


class _ConcurrencyProbe:
    """Stands in for the retina-face module and records how many inferences overlap."""

    def __init__(self):
        self.active = 0
        self.peak = 0
        self._lock = threading.Lock()

    def detect_faces(self, image, **kwargs):
        with self._lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        time.sleep(0.01)
        with self._lock:
            self.active -= 1
        return {"face_1": {"facial_area": [10, 10, 50, 50]}}


def _probed_detector():
    detector = RetinaFaceDetector.__new__(RetinaFaceDetector)
    detector._retinaface, detector._model = _ConcurrencyProbe(), None
    detector._threshold, detector._size = 0.9, 640
    detector._inference = threading.Lock()
    return detector


def test_detector_never_runs_two_inferences_at_once():
    detector = _probed_detector()

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(detector, [np.zeros((100, 100, 3), np.uint8)] * 16))

    assert detector._retinaface.peak == 1


def test_detector_maps_boxes_back_to_the_original_scale():
    detector = _probed_detector()

    [box] = detector(np.zeros((1280, 1280, 3), np.uint8))

    # The 1280 px image is shrunk by half to fit 640 px, so the 10..50 box doubles.
    assert (box.x, box.y, box.w, box.h) == (20, 20, 80, 80)


def test_tighten_crop_keeps_the_center_at_a_smaller_margin():
    from PIL import Image

    from face_decision_lab.faces import tighten_crop

    image = Image.new("RGB", (256, 256))
    image.putpixel((128, 128), (255, 0, 0))

    tightened = tighten_crop(image, face_scale=1.3)

    # 256 px hold the face box at 1.5x; at 1.3x the same box needs round(256 * 1.3 / 1.5) = 222 px.
    assert tightened.size == (222, 222)
    assert tightened.getpixel((128 - 17, 128 - 17)) == (255, 0, 0)


def test_tighten_crop_at_the_build_margin_is_a_no_op():
    from PIL import Image

    from face_decision_lab.faces import CROP_FACE_SCALE, tighten_crop

    image = Image.new("RGB", (256, 256), (1, 2, 3))

    assert tighten_crop(image, face_scale=CROP_FACE_SCALE).size == (256, 256)


def test_tighten_crop_cannot_add_margin_the_crop_does_not_have():
    from PIL import Image

    from face_decision_lab.faces import tighten_crop

    with pytest.raises(ValueError, match="face_scale 1.8 is wider than the crop's 1.5"):
        tighten_crop(Image.new("RGB", (256, 256)), face_scale=1.8)


# Landmarks in image order: two eyes, nose, two mouth corners (GenD's 5-point layout).
def _face_landmarks(center_x=200.0, center_y=180.0, size=100.0):
    reference = [(0.34, 0.46), (0.66, 0.46), (0.5, 0.64), (0.37, 0.82), (0.63, 0.82)]
    return tuple((center_x + (x - 0.5) * size, center_y + (y - 0.5) * size) for x, y in reference)


def test_aligned_crop_puts_the_nose_where_gend_puts_it():
    from face_decision_lab.faces import aligned_crop

    landmarks = _face_landmarks()
    image = np.zeros((400, 400, 3), dtype=np.uint8)
    nose_x, nose_y = (round(v) for v in landmarks[2])
    image[nose_y - 2 : nose_y + 3, nose_x - 2 : nose_x + 3] = 255

    crop = aligned_crop(image, landmarks, size=256)

    # GenD's reference nose (0.5, 0.64) with a 1.3x margin lands at ((0.5 + 0.15) / 1.3,
    # (0.64 + 0.15) / 1.3) of the output: (128, 155.6) in a 256-pixel crop.
    ys, xs = np.nonzero(crop[:, :, 0] > 128)
    assert crop.shape == (256, 256, 3)
    assert abs(xs.mean() - 128) < 2 and abs(ys.mean() - 155.6) < 2


def test_aligned_crop_does_not_depend_on_the_landmark_order():
    from face_decision_lab.faces import aligned_crop

    landmarks = _face_landmarks()
    swapped = (landmarks[1], landmarks[0], landmarks[2], landmarks[4], landmarks[3])
    image = np.random.default_rng(0).integers(0, 255, (400, 400, 3), dtype=np.uint8)

    assert np.array_equal(aligned_crop(image, landmarks), aligned_crop(image, swapped))


def test_aligned_crop_rejects_a_face_without_five_landmarks():
    from face_decision_lab.faces import aligned_crop

    with pytest.raises(ValueError, match="needs 5 landmarks, got 0"):
        aligned_crop(np.zeros((100, 100, 3), dtype=np.uint8), ())
