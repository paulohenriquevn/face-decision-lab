"""Builds the training dataset: source files in, standard face crops and a manifest out.

The build is resumable. Every source file that was handled, whatever its outcome, is appended to
`processed.jsonl` with the rows it produced. That log is the only source of truth: the manifest is
rebuilt from it, so a Colab session that dies halfway restarts where it stopped and loses nothing.
"""

import json
import zipfile
from collections.abc import Callable, Iterable, Iterator
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .faces import Box, CropReport, Outcome, encode_jpeg, frame_indices, pick_face, square_crop
from .manifest import FaceSample, write_manifest
from .sources.base import SourceItem

Detector = Callable[[np.ndarray], list[Box]]
FRAMES_PER_VIDEO = 16


def crop_frame(image: np.ndarray | None, detector: Detector) -> tuple[bytes | None, Outcome]:
    if image is None or image.size == 0:
        return None, Outcome.UNREADABLE
    box, outcome = pick_face(detector(image))
    if box is None:
        return None, outcome
    return encode_jpeg(square_crop(image, box)), Outcome.SAVED


def decode_image(data: bytes) -> np.ndarray | None:
    import cv2

    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def video_frames(path: Path, count: int) -> Iterator[tuple[int, np.ndarray | None]]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise OSError(f"cannot open video {path}")
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        for index in frame_indices(total, count):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            read, frame = capture.read()
            yield index, frame if read else None
    finally:
        capture.release()


class DatasetBuilder:
    def __init__(self, output_dir: str | Path, detector: Detector, frames_per_video: int = FRAMES_PER_VIDEO):
        self.output_dir = Path(output_dir)
        self.detector = detector
        self.frames_per_video = frames_per_video
        self.report = CropReport()
        self._log_path = self.output_dir / "processed.jsonl"
        self._manifest_path = self.output_dir / "manifest.csv"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._done: set[str] = set()
        self._samples: list[FaceSample] = []
        if self._log_path.exists():
            for line in self._log_path.read_text().splitlines():
                entry = json.loads(line)
                self._done.add(entry["key"])
                for outcome in entry["outcomes"]:
                    self.report.add(entry["generator"], Outcome(outcome))
                self._samples.extend(FaceSample(**row) for row in entry["samples"])

    @property
    def samples(self) -> list[FaceSample]:
        return list(self._samples)

    def add_image(self, item: SourceItem, data: bytes) -> None:
        key = f"{item.source_dataset}:{item.path}"
        if key in self._done:
            return
        crop, outcome = crop_frame(decode_image(data), self.detector)
        self._record(item, key, [(0, crop, outcome)])

    def add_video(self, item: SourceItem, path: Path) -> None:
        key = f"{item.source_dataset}:{item.path}"
        if key in self._done:
            return
        results = [
            (index, *crop_frame(frame, self.detector))
            for index, frame in video_frames(path, self.frames_per_video)
        ]
        self._record(item, key, results)

    def _record(self, item: SourceItem, key: str, results: list[tuple[int, bytes | None, Outcome]]) -> None:
        new_samples = []
        for index, crop, outcome in results:
            self.report.add(item.generator, outcome)
            if crop is None:
                continue
            relative = Path("images") / item.source_dataset / f"{item.video_id}_{index:05d}.jpg"
            target = self.output_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(crop)
            sample_fields = {k: v for k, v in asdict(item).items() if k != "path"}
            new_samples.append(FaceSample(image_path=relative.as_posix(), **sample_fields))
        self._samples.extend(new_samples)
        with self._log_path.open("a") as log:
            entry = {
                "key": key,
                "generator": item.generator,
                "outcomes": [o.value for _, _, o in results],
                "samples": [asdict(sample) for sample in new_samples],
            }
            log.write(json.dumps(entry) + "\n")
        self._done.add(key)

    def flush(self) -> None:
        """Writes the manifest; call it periodically and at the end."""
        write_manifest(self._samples, self._manifest_path)
        (self.output_dir / "report.json").write_text(json.dumps(self.report.counts, indent=2, sort_keys=True))


def zip_members(path: Path, parse: Callable[[str], SourceItem | None]) -> Iterator[tuple[SourceItem, bytes]]:
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            item = parse(name)
            if item is not None:
                yield item, archive.read(name)


def folder_files(root: Path, parse: Callable[[str], SourceItem | None]) -> Iterable[tuple[SourceItem, Path]]:
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative.startswith("."):
            continue
        item = parse(relative)
        if item is not None:
            yield item, path
