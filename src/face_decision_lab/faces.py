"""Face detection and the standard crop every image goes through, whatever its class.

Every crop has the same size, margin and JPEG encoding. Real and fake photos often differ in
resolution and compression (in DeepFakeFace every fake is 512x512 and the real photos are not), and
a model would learn that shortcut instead of the manipulation.
"""

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

CROP_SIZE = 256
# Context around the detected box: swap and inpainting artifacts sit on the face boundary.
CROP_MARGIN = 0.25
JPEG_QUALITY = 95
# The largest face must be this many times larger than the next one, or the image is ambiguous.
DOMINANCE = 2.0


class Outcome(str, Enum):
    SAVED = "saved"
    NO_FACE = "no_face"
    AMBIGUOUS = "ambiguous"
    UNREADABLE = "unreadable"


@dataclass(frozen=True)
class Box:
    x: float
    y: float
    w: float
    h: float

    @property
    def area(self) -> float:
        return self.w * self.h


@dataclass
class CropReport:
    """Counts per (generator, outcome). A detector that misses one class more than another biases
    the dataset, so check these counts before training."""

    counts: dict[str, dict[str, int]] = field(default_factory=dict)

    def add(self, generator: str, outcome: Outcome) -> None:
        by_outcome = self.counts.setdefault(generator, {})
        by_outcome[outcome.value] = by_outcome.get(outcome.value, 0) + 1

    def saved_share(self, generator: str) -> float:
        by_outcome = self.counts.get(generator, {})
        total = sum(by_outcome.values())
        return by_outcome.get(Outcome.SAVED.value, 0) / total if total else 0.0


def pick_face(boxes: list[Box], dominance: float = DOMINANCE) -> tuple[Box | None, Outcome]:
    """The face to crop, or why there is none. Background faces are fine when much smaller."""
    if not boxes:
        return None, Outcome.NO_FACE
    ordered = sorted(boxes, key=lambda box: box.area, reverse=True)
    if len(ordered) > 1 and ordered[0].area < dominance * ordered[1].area:
        return None, Outcome.AMBIGUOUS
    return ordered[0], Outcome.SAVED


def square_crop(
    image: np.ndarray, box: Box, margin: float = CROP_MARGIN, size: int = CROP_SIZE
) -> np.ndarray:
    """A square around the box, grown by margin on each side, padded with black past the image
    border and resized to size x size."""
    import cv2

    side = max(box.w, box.h) * (1 + 2 * margin)
    center_x, center_y = box.x + box.w / 2, box.y + box.h / 2
    left, top = int(round(center_x - side / 2)), int(round(center_y - side / 2))
    side = int(round(side))
    height, width = image.shape[:2]
    canvas = np.zeros((side, side) + image.shape[2:], dtype=image.dtype)
    src_left, src_top = max(left, 0), max(top, 0)
    src_right, src_bottom = min(left + side, width), min(top + side, height)
    if src_right > src_left and src_bottom > src_top:
        canvas[src_top - top : src_bottom - top, src_left - left : src_right - left] = image[
            src_top:src_bottom, src_left:src_right
        ]
    interpolation = cv2.INTER_AREA if side > size else cv2.INTER_CUBIC
    return cv2.resize(canvas, (size, size), interpolation=interpolation)


def encode_jpeg(crop: np.ndarray) -> bytes:
    import cv2

    ok, buffer = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise ValueError(f"cannot encode a {crop.shape} crop as JPEG")
    return buffer.tobytes()


def frame_indices(total_frames: int, count: int) -> list[int]:
    """Evenly spaced frame indices, centered in each slice of the video."""
    if total_frames <= 0:
        raise ValueError(f"video has {total_frames} frames")
    if count <= 0:
        raise ValueError(f"frame count must be positive, got {count}")
    count = min(count, total_frames)
    step = total_frames / count
    return [int(step * i + step / 2) for i in range(count)]


# Detection input side. The retina-face package upscales every image so its short side is 1024 px,
# which on a GPU means a large, different input shape per image. Portraits fit in 640 px.
DETECTION_SIZE = 640


def letterbox(image: np.ndarray, size: int = DETECTION_SIZE) -> tuple[np.ndarray, float]:
    """Shrinks the image to fit a size x size canvas (never enlarges it), pads the rest with black
    and returns the canvas with the scale applied. Every detection input then has the same shape."""
    import cv2

    height, width = image.shape[:2]
    scale = min(1.0, size / max(height, width))
    if scale < 1.0:
        image = cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size) + image.shape[2:], dtype=image.dtype)
    canvas[: image.shape[0], : image.shape[1]] = image
    return canvas, scale


class RetinaFaceDetector:
    """RetinaFace (the `retina-face` package, MIT). It runs on TensorFlow, PyTorch or ONNX Runtime,
    whichever is installed first in that order; only TensorFlow and onnxruntime-gpu use a GPU.
    Boxes come back in the coordinates of the original image."""

    def __init__(self, threshold: float = 0.9, size: int = DETECTION_SIZE):
        from retinaface import RetinaFace

        self._retinaface = RetinaFace
        self._model = RetinaFace.build_model()
        self._threshold = threshold
        self._size = size

    def __call__(self, image: np.ndarray) -> list[Box]:
        canvas, scale = letterbox(image, self._size)
        found = self._retinaface.detect_faces(
            canvas, threshold=self._threshold, model=self._model, allow_upscaling=False
        )
        if not isinstance(found, dict):
            return []
        boxes = []
        for face in found.values():
            x1, y1, x2, y2 = (value / scale for value in face["facial_area"])
            boxes.append(Box(x=x1, y=y1, w=x2 - x1, h=y2 - y1))
        return boxes
