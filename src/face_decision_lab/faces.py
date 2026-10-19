"""Face detection and the standard crop every image goes through, whatever its class.

Every crop has the same size, margin and JPEG encoding. Real and fake photos often differ in
resolution and compression (in DeepFakeFace every fake is 512x512 and the real photos are not), and
a model would learn that shortcut instead of the manipulation.
"""

import threading
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

CROP_SIZE = 256
# Context around the detected box: swap and inpainting artifacts sit on the face boundary.
CROP_MARGIN = 0.25
JPEG_QUALITY = 95
# Side of a crop as a multiple of the face box side.
CROP_FACE_SCALE = 1 + 2 * CROP_MARGIN
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
    # The detector's 5 landmarks (two eyes, nose, two mouth corners), in image coordinates.
    landmarks: tuple[tuple[float, float], ...] = ()

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


# GenD's reference positions of the 5 landmarks in a face crop, as fractions of its side
# (yermandy/GenD, detector.py, align_face; MIT, Copyright (c) 2025 Andy).
ALIGNED_REFERENCE = ((0.34, 0.46), (0.66, 0.46), (0.5, 0.64), (0.37, 0.82), (0.63, 0.82))
ALIGNED_FACE_SCALE = 1.3


def aligned_crop(
    image: np.ndarray,
    landmarks: tuple[tuple[float, float], ...],
    scale: float = ALIGNED_FACE_SCALE,
    size: int = CROP_SIZE,
) -> np.ndarray:
    """The face warped so its landmarks land on GenD's reference points, with the same context
    margin GenD trained on (scale 1.3), as a size x size image.

    A port of GenD's align_face with a fixed output size. Eyes and mouth corners are ordered left to
    right by x, so the result does not depend on which eye a detector calls "left".
    """
    import cv2

    if len(landmarks) != 5:
        raise ValueError(f"aligned_crop needs 5 landmarks, got {len(landmarks)}")
    eyes = sorted(landmarks[:2])
    mouth = sorted(landmarks[3:])
    source = np.array([*eyes, landmarks[2], *mouth], dtype=np.float32)
    margin = (scale - 1) / 2
    target = (np.array(ALIGNED_REFERENCE, dtype=np.float32) + margin) / scale * size
    matrix = cv2.estimateAffinePartial2D(source, target, method=cv2.LMEDS)[0]
    return cv2.warpAffine(image, matrix, (size, size), flags=cv2.INTER_LINEAR)


def tighten_crop(image, face_scale: float, crop_face_scale: float = CROP_FACE_SCALE):
    """Center-crops a built crop (PIL image) down to a smaller margin around the face.

    Pretrained detectors expect their own margin: GenD trained on face boxes grown to 1.3x, while
    the lab's crops hold the box at CROP_FACE_SCALE. The face sits at the center of every crop, so
    a center crop of side * face_scale / crop_face_scale gives the tighter framing.
    """
    if face_scale > crop_face_scale:
        raise ValueError(f"face_scale {face_scale} is wider than the crop's {crop_face_scale}")
    side = round(image.width * face_scale / crop_face_scale)
    left = (image.width - side) // 2
    top = (image.height - side) // 2
    return image.crop((left, top, left + side, top + side))


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
    Boxes come back in the coordinates of the original image.

    Safe to call from several threads: inference runs one call at a time. Concurrent runs of one
    ONNX Runtime CUDA session crashed a Colab L4 build with "CUDA failure 700: an illegal memory
    access", so callers may decode and crop in parallel but never infer in parallel."""

    def __init__(self, threshold: float = 0.9, size: int = DETECTION_SIZE):
        from retinaface import RetinaFace

        self._retinaface = RetinaFace
        self._model = RetinaFace.build_model()
        self._threshold = threshold
        self._size = size
        self._inference = threading.Lock()

    def __call__(self, image: np.ndarray) -> list[Box]:
        canvas, scale = letterbox(image, self._size)
        with self._inference:
            found = self._retinaface.detect_faces(
                canvas, threshold=self._threshold, model=self._model, allow_upscaling=False
            )
        if not isinstance(found, dict):
            return []
        boxes = []
        for face in found.values():
            x1, y1, x2, y2 = (value / scale for value in face["facial_area"])
            points = face.get("landmarks", {})
            landmarks = tuple(
                (float(points[name][0]) / scale, float(points[name][1]) / scale)
                for name in ("right_eye", "left_eye", "nose", "mouth_right", "mouth_left")
                if name in points
            )
            boxes.append(Box(x=x1, y=y1, w=x2 - x1, h=y2 - y1, landmarks=landmarks))
        return boxes
