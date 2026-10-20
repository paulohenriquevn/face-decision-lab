import json
import zipfile

import cv2
import numpy as np
import pytest

from face_decision_lab.build import DatasetBuilder, zip_members
from face_decision_lab.faces import Box
from face_decision_lab.manifest import load_manifest
from face_decision_lab.sources import axondata, deepfakeface


def one_face(image):
    return [Box(10, 10, 40, 40)]


def no_face(image):
    return []


def jpeg(color=128, size=(96, 128)):
    ok, data = cv2.imencode(".jpg", np.full(size + (3,), color, np.uint8))
    return data.tobytes()


@pytest.fixture
def wiki_item():
    return deepfakeface.parse_member("wiki/61/19242061_1982-08-16_2009.jpg")


def test_builder_writes_a_crop_and_a_manifest_row(tmp_path, wiki_item):
    builder = DatasetBuilder(tmp_path, one_face)

    builder.add_image(wiki_item, jpeg())
    builder.flush()

    [sample] = load_manifest(tmp_path / "manifest.csv")
    assert (sample.attack_type, sample.subject_id, sample.generator) == (
        "bona_fide",
        "imdbwiki_19242061",
        "wiki",
    )
    assert cv2.imread(str(tmp_path / sample.image_path)).shape == (256, 256, 3)


def test_builder_counts_images_without_a_face_and_writes_no_row(tmp_path, wiki_item):
    builder = DatasetBuilder(tmp_path, no_face)

    builder.add_image(wiki_item, jpeg())
    builder.flush()

    assert builder.samples == []
    assert json.loads((tmp_path / "report.json").read_text()) == {"wiki": {"no_face": 1}}


def test_builder_counts_undecodable_bytes_as_unreadable(tmp_path, wiki_item):
    builder = DatasetBuilder(tmp_path, one_face)

    builder.add_image(wiki_item, b"not an image")

    assert builder.report.counts == {"wiki": {"unreadable": 1}}


def test_builder_resumes_without_reprocessing_or_losing_rows(tmp_path, wiki_item):
    first = DatasetBuilder(tmp_path, one_face)
    first.add_image(wiki_item, jpeg())
    # No flush: the session dies before the manifest is written.

    second = DatasetBuilder(tmp_path, no_face)
    second.add_image(wiki_item, jpeg())

    assert len(second.samples) == 1
    assert second.report.counts == {"wiki": {"saved": 1}}


def test_builder_samples_frames_from_a_video(tmp_path):
    video = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (128, 96))
    for _ in range(30):
        writer.write(np.full((96, 128, 3), 90, np.uint8))
    writer.release()
    item = axondata.parse_path("Latex_mask/Mask_6/1.mp4")
    builder = DatasetBuilder(tmp_path / "out", one_face, frames_per_video=4)

    builder.add_video(item, video)

    assert len(builder.samples) == 4
    assert {s.video_id for s in builder.samples} == {item.video_id}


def test_zip_members_yields_parsed_items_with_their_bytes(tmp_path):
    archive = tmp_path / "wiki.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("wiki/61/", "")
        handle.writestr("wiki/61/19242061_1982-08-16_2009.jpg", b"bytes")

    [(item, data)] = list(zip_members(archive, deepfakeface.parse_member))

    assert (item.generator, data) == ("wiki", b"bytes")


def test_add_images_records_every_image_in_input_order(tmp_path):
    names = [f"wiki/61/{i}_1982-08-16_2009.jpg" for i in range(10)]
    items = [(deepfakeface.parse_member(name), jpeg()) for name in names]
    builder = DatasetBuilder(tmp_path, one_face)

    processed = builder.add_images(items, workers=4, batch=3)

    assert processed == 10
    assert [s.subject_id for s in builder.samples] == [f"imdbwiki_{i}" for i in range(10)]


def test_add_images_skips_images_already_processed(tmp_path, wiki_item):
    builder = DatasetBuilder(tmp_path, one_face)
    builder.add_image(wiki_item, jpeg())

    processed = builder.add_images([(wiki_item, jpeg())])

    assert processed == 0
    assert len(builder.samples) == 1


def test_builder_crops_with_the_cropper_it_was_given(tmp_path, wiki_item):
    calls = []

    def tiny_crop(image, box):
        calls.append(box)
        return np.zeros((32, 32, 3), np.uint8)

    builder = DatasetBuilder(tmp_path, one_face, cropper=tiny_crop)

    builder.add_image(wiki_item, jpeg())

    [sample] = builder.samples
    assert calls == [Box(10, 10, 40, 40)]
    assert cv2.imread(str(tmp_path / sample.image_path)).shape == (32, 32, 3)


def _tar_stream(files):
    import io
    import tarfile

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    buffer.seek(0)
    return buffer


def test_tar_members_reads_a_stream_once_and_keeps_only_the_chosen_members():
    from face_decision_lab.build import tar_members
    from face_decision_lab.sources import diffusionface

    stream = _tar_stream({"Real/1.png": b"one", "Real/2.png": b"two", "Real/3.png": b"three"})

    pairs = list(tar_members(stream, diffusionface.parse_member, keep=lambda name: name != "Real/2.png"))

    assert [(item.path, data) for item, data in pairs] == [("Real/1.png", b"one"), ("Real/3.png", b"three")]


def _zip_bytes(members):
    import io
    import zipfile

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def test_remote_zip_members_reads_only_the_kept_members_of_a_ranged_archive():
    from face_decision_lab.build import RangeReader, remote_zip_members
    from face_decision_lab.sources import diffusionface

    data = _zip_bytes({f"Real/{n}.png": bytes([n]) * 100_000 for n in range(20)})
    fetched = []

    def fetch(start, end):
        fetched.append(end - start + 1)
        return data[start : end + 1]

    reader = RangeReader(fetch, len(data), block=16_384)

    pairs = list(
        remote_zip_members(reader, diffusionface.parse_member, keep=lambda item: item.path == "Real/3.png")
    )

    assert [(item.path, payload) for item, payload in pairs] == [("Real/3.png", bytes([3]) * 100_000)]
    assert sum(fetched) < len(data) / 4


def test_range_reader_rejects_a_short_answer():
    from face_decision_lab.build import RangeReader

    reader = RangeReader(lambda start, end: b"x", 100, block=10)

    with pytest.raises(OSError, match="asked for bytes 0-9 of 100, got 1"):
        reader.read(10)


class _Response:
    def __init__(self, status, headers, body):
        self.status, self.headers, self._body = status, headers, body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_http_range_reader_learns_the_size_from_content_range():
    from face_decision_lab.build import http_range_reader

    def urlopen(request, timeout):
        start, end = map(int, request.headers["Range"].removeprefix("bytes=").split("-"))
        return _Response(206, {"Content-Range": f"bytes {start}-{end}/1000"}, b"a" * (end - start + 1))

    reader = http_range_reader("https://example.org/a.zip", urlopen=urlopen)

    assert reader.size == 1000
    assert reader.read(5) == b"aaaaa"


def test_http_range_reader_fails_loudly_on_a_page_instead_of_bytes():
    from face_decision_lab.build import http_range_reader

    def urlopen(request, timeout):
        return _Response(200, {"Content-Type": "text/html"}, b"<title>Google Drive - Quota exceeded</title>")

    with pytest.raises(
        OSError, match="https://example.org/a.zip did not answer a byte range .status 200, text/html"
    ):
        http_range_reader("https://example.org/a.zip", urlopen=urlopen, sleep=lambda seconds: None)


def test_remote_zip_members_reads_stored_and_deflated_members_in_parallel():
    import io
    import zipfile

    from face_decision_lab.build import RangeReader, remote_zip_members
    from face_decision_lab.sources import diffusionface

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for n in range(30):
            method = zipfile.ZIP_DEFLATED if n % 2 else zipfile.ZIP_STORED
            archive.writestr(f"Real/{n}.png", bytes([n]) * 5000, compress_type=method)
    data = buffer.getvalue()
    reader = RangeReader(lambda start, end: data[start : end + 1], len(data), block=4096)

    pairs = list(remote_zip_members(reader, diffusionface.parse_member, workers=8))

    assert [(item.path, payload) for item, payload in pairs] == [
        (f"Real/{n}.png", bytes([n]) * 5000) for n in range(30)
    ]


def test_http_range_reader_waits_out_a_temporary_refusal():
    from face_decision_lab.build import http_range_reader

    answers = [_Response(200, {"Content-Type": "text/html"}, b"<html>"), None]
    waits = []

    def urlopen(request, timeout):
        answer = answers.pop(0)
        if answer is not None:
            return answer
        start, end = map(int, request.headers["Range"].removeprefix("bytes=").split("-"))
        return _Response(206, {"Content-Range": f"bytes {start}-{end}/10"}, b"a" * (end - start + 1))

    reader = http_range_reader("https://example.org/a.zip", urlopen=urlopen, sleep=waits.append)

    assert reader.size == 10
    assert len(waits) == 1
