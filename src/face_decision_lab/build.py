"""Builds the training dataset: source files in, standard face crops and a manifest out.

The build is resumable. Every source file that was handled, whatever its outcome, is appended to
`processed.jsonl` with the rows it produced. That log is the only source of truth: the manifest is
rebuilt from it, so a Colab session that dies halfway restarts where it stopped and loses nothing.
"""

import io
import json
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
import zlib
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from itertools import islice
from pathlib import Path

import numpy as np

from .faces import Box, CropReport, Outcome, encode_jpeg, frame_indices, pick_face, square_crop
from .manifest import FaceSample, write_manifest
from .sources.base import SourceItem

Detector = Callable[[np.ndarray], list[Box]]
# Turns an image and the chosen face into the stored crop: square_crop, or aligned_crop on the
# face's landmarks for detectors trained on aligned faces.
Cropper = Callable[[np.ndarray, Box], np.ndarray]
FRAMES_PER_VIDEO = 16


def crop_frame(
    image: np.ndarray | None, detector: Detector, cropper: Cropper = square_crop
) -> tuple[bytes | None, Outcome]:
    if image is None or image.size == 0:
        return None, Outcome.UNREADABLE
    box, outcome = pick_face(detector(image))
    if box is None:
        return None, outcome
    return encode_jpeg(cropper(image, box)), Outcome.SAVED


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
    def __init__(
        self,
        output_dir: str | Path,
        detector: Detector,
        frames_per_video: int = FRAMES_PER_VIDEO,
        cropper: Cropper = square_crop,
    ):
        self.output_dir = Path(output_dir)
        self.detector = detector
        self.cropper = cropper
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
        crop, outcome = crop_frame(decode_image(data), self.detector, self.cropper)
        self._record(item, key, [(0, crop, outcome)])

    def add_images(
        self, items: Iterable[tuple[SourceItem, bytes]], workers: int = 8, batch: int = 256
    ) -> int:
        """Crops many images on a thread pool; returns how many were processed.

        Detection runs in ONNX Runtime and OpenCV, which release the GIL, so threads keep the GPU
        busy while others decode and post-process. Results are recorded in input order.
        """
        pending = (
            (item, data) for item, data in items if f"{item.source_dataset}:{item.path}" not in self._done
        )
        processed = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            while chunk := list(islice(pending, batch)):
                crops = pool.map(
                    lambda pair: crop_frame(decode_image(pair[1]), self.detector, self.cropper), chunk
                )
                for (item, _), (crop, outcome) in zip(chunk, crops, strict=True):
                    self._record(item, f"{item.source_dataset}:{item.path}", [(0, crop, outcome)])
                processed += len(chunk)
        return processed

    def add_video(self, item: SourceItem, path: Path) -> None:
        key = f"{item.source_dataset}:{item.path}"
        if key in self._done:
            return
        results = [
            (index, *crop_frame(frame, self.detector, self.cropper))
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


class RangeReader(io.RawIOBase):
    """A read-only, seekable file over fetch(start, end), which returns bytes start to end inclusive,
    so zipfile can read single members of a remote archive. Reads go in blocks of at least block
    bytes, each fetched once."""

    def __init__(self, fetch: Callable[[int, int], bytes], size: int, block: int = 1 << 20):
        self._fetch, self.size, self.block, self._position = fetch, size, block, 0

    def fetch(self, start: int, end: int) -> bytes:
        """Bytes start to end inclusive, read directly, outside the file position."""
        data = self._fetch(start, end)
        if len(data) != end - start + 1:
            raise OSError(f"asked for bytes {start}-{end} of {self.size}, got {len(data)}")
        return data

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self._position, io.SEEK_END: self.size}[whence]
        self._position = base + offset
        return self._position

    def readinto(self, buffer) -> int:
        count = min(max(len(buffer), self.block), self.size - self._position)
        if count <= 0:
            return 0
        start, end = self._position, self._position + count - 1
        data = self.fetch(start, end)
        taken = min(len(buffer), count)
        buffer[:taken] = data[:taken]
        self._position += taken
        return taken


class RangeRefusedError(OSError):
    """The server answered something other than the requested byte range."""


def http_range_reader(
    url: str, attempts: int = 6, urlopen=urllib.request.urlopen, sleep: Callable[[float], None] = time.sleep
) -> RangeReader:
    """A RangeReader over an HTTP URL that answers byte ranges. Google Drive keeps answering ranges
    of a file after it refuses whole downloads of it for the day ("Quota exceeded"), but after many
    thousands of requests from one address it answers a page for a while; such a refusal, like a
    network error, is retried with a growing wait."""

    def get(start: int, end: int):
        request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
        for attempt in range(attempts):
            try:
                with urlopen(request, timeout=120) as response:
                    if response.status != 206:
                        kind = response.headers.get("Content-Type", "unknown type")
                        raise RangeRefusedError(
                            f"{url} did not answer a byte range (status {response.status}, {kind})"
                        )
                    return response.headers["Content-Range"], response.read()
            except (TimeoutError, ConnectionError, urllib.error.URLError, RangeRefusedError):
                if attempt == attempts - 1:
                    raise
                sleep(min(5 * 2**attempt, 300))
        raise AssertionError("unreachable")

    content_range, _ = get(0, 0)
    return RangeReader(lambda start, end: get(start, end)[1], int(content_range.rsplit("/", 1)[1]))


def remote_zip_members(
    reader: RangeReader,
    parse: Callable[[str], SourceItem | None],
    keep: Callable[[SourceItem], bool] = lambda item: True,
    workers: int = 8,
) -> Iterator[tuple[SourceItem, bytes]]:
    """Reads the kept members of a zip archive behind reader, in archive order, fetching only the
    central directory and those members, each with one ranged request on a thread pool, so a sample
    of a large remote archive is built without downloading it."""
    with zipfile.ZipFile(io.BufferedReader(reader, buffer_size=reader.block)) as archive:
        chosen = [
            (item, info)
            for info in archive.infolist()
            if (item := parse(info.filename)) is not None and keep(item)
        ]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for (item, _), data in zip(
            chosen, pool.map(lambda pair: _member(reader, pair[1]), chosen), strict=True
        ):
            yield item, data


# A local file header is 30 bytes, then the name and an extra field whose length can differ from
# the central directory's copy; this much slack covers it in the same request.
_LOCAL_HEADER_SLACK = 1024


def _member(reader: RangeReader, info: zipfile.ZipInfo) -> bytes:
    start = info.header_offset
    end = min(
        reader.size, start + 30 + len(info.orig_filename.encode()) + _LOCAL_HEADER_SLACK + info.compress_size
    )
    chunk = reader.fetch(start, end - 1)
    if chunk[:4] != b"PK\x03\x04":
        raise zipfile.BadZipFile(f"no local file header for {info.filename!r} at offset {start}")
    name_length, extra_length = int.from_bytes(chunk[26:28], "little"), int.from_bytes(chunk[28:30], "little")
    data_start = 30 + name_length + extra_length
    raw = chunk[data_start : data_start + info.compress_size]
    if len(raw) < info.compress_size:
        raw += reader.fetch(start + data_start + len(raw), start + data_start + info.compress_size - 1)
    if info.compress_type == zipfile.ZIP_STORED:
        return raw
    if info.compress_type == zipfile.ZIP_DEFLATED:
        return zlib.decompress(raw, -15)
    raise zipfile.BadZipFile(
        f"{info.filename!r} uses compression {info.compress_type}, not stored or deflate"
    )


def tar_members(
    stream, parse: Callable[[str], SourceItem | None], keep: Callable[[str], bool] = lambda name: True
) -> Iterator[tuple[SourceItem, bytes]]:
    """Reads a tar archive front to back from a non-seekable stream, such as an HTTP download, so
    a sample of a large archive is built without storing the archive."""
    with tarfile.open(fileobj=stream, mode="r|") as archive:
        for member in archive:
            if not member.isfile() or not keep(member.name):
                continue
            item = parse(member.name)
            if item is not None:
                yield item, archive.extractfile(member).read()


def folder_files(root: Path, parse: Callable[[str], SourceItem | None]) -> Iterable[tuple[SourceItem, Path]]:
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative.startswith("."):
            continue
        item = parse(relative)
        if item is not None:
            yield item, path
