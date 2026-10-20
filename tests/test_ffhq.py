import pytest

from face_decision_lab.sources import ffhq
from face_decision_lab.sources.base import SourceLayoutError


def test_an_ffhq_image_is_a_bona_fide_photo_of_its_own_subject():
    item = ffhq.parse_name("01234.png")

    assert item.attack_type == "bona_fide"
    assert item.subject_id == "ffhq_01234"
    assert item.video_id == "ffhq/01234"
    assert item.source_dataset == "ffhq"
    assert item.generator == "ffhq_real"


def test_ffhq_names_outside_the_numbered_layout_are_refused():
    with pytest.raises(SourceLayoutError):
        ffhq.parse_name("images/thumb.jpg")


def test_the_sample_keeps_the_same_names_on_every_run_and_about_the_asked_share():
    names = [f"{i:05d}.png" for i in range(7000)]

    kept = [n for n in names if ffhq.sampled(n, keep=3, out_of=7)]

    assert kept == [n for n in names if ffhq.sampled(n, keep=3, out_of=7)]
    assert 2800 < len(kept) < 3200
