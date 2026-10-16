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
