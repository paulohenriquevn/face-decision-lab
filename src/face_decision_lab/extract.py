"""Samples frames from a video and saves one face crop per frame.

Needs the `extract` extra (DeepFace and OpenCV). Frames with no face or with more than one face are
skipped and counted, never guessed: a wrong crop would teach the model the wrong label.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ExtractionReport:
    saved: list[Path] = field(default_factory=list)
    no_face: int = 0
    several_faces: int = 0
    unreadable: int = 0


def frame_indices(total_frames: int, count: int) -> list[int]:
    """Evenly spaced frame indices, centered in each slice of the video."""
    if total_frames <= 0:
        raise ValueError(f"video has {total_frames} frames")
    if count <= 0:
        raise ValueError(f"frame count must be positive, got {count}")
    count = min(count, total_frames)
    step = total_frames / count
    return [int(step * i + step / 2) for i in range(count)]


def extract_face_crops(
    video_path: str | Path,
    output_dir: str | Path,
    frames_per_video: int = 8,
    detector_backend: str = "retinaface",
) -> ExtractionReport:
    import cv2
    from deepface import DeepFace

    video_path, output_dir = Path(video_path), Path(output_dir)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise OSError(f"cannot open video {video_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    report = ExtractionReport()
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        for index in frame_indices(total, frames_per_video):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            read, frame = capture.read()
            if not read:
                report.unreadable += 1
                continue
            faces = DeepFace.extract_faces(
                frame,
                detector_backend=detector_backend,
                enforce_detection=False,
                align=True,
                color_face="bgr",
                normalize_face=False,
            )
            # With enforce_detection=False, DeepFace returns the whole frame with confidence 0
            # when it finds no face.
            faces = [face for face in faces if face["confidence"] > 0]
            if not faces:
                report.no_face += 1
                continue
            if len(faces) > 1:
                report.several_faces += 1
                continue
            path = output_dir / f"{video_path.stem}_{index:06d}.png"
            if not cv2.imwrite(str(path), faces[0]["face"]):
                raise OSError(f"cannot write {path}")
            report.saved.append(path)
    finally:
        capture.release()
    return report
