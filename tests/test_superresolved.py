import pytest

from face_decision_lab.pairs import pair_key
from face_decision_lab.sources import superresolved


def test_a_super_resolved_copy_is_a_bona_fide_sample_of_the_same_person(make_sample):
    original = make_sample(subject="s7", video="real/imdb/0001.jpg", source="deepfakeface", generator="real")

    copy = superresolved.super_resolved(original, "realesrgan_x4plus", "sr/0.jpg")

    assert copy.is_bona_fide
    assert copy.subject_id == "s7"
    assert copy.source_dataset == "superresolved"
    assert copy.generator == "sr_realesrgan_x4plus"
    assert copy.image_path == "sr/0.jpg"


@pytest.mark.parametrize(
    ("source", "subject", "video", "generator"),
    [
        ("deepfakeface", "s7", "real/imdb/0001.jpg", "real"),
        ("faceforensics", "ffpp_000", "ffpp/original/000", "ffpp_original"),
    ],
)
def test_a_copy_lands_in_the_pair_group_of_its_original(make_sample, source, subject, video, generator):
    original = make_sample(subject=subject, video=video, source=source, generator=generator)

    copy = superresolved.super_resolved(original, "swin2sr_realworld_x4", "sr/0.jpg")

    assert pair_key(copy) == pair_key(original)


def test_only_real_crops_are_super_resolved(make_sample):
    fake = make_sample(attack_type="face_swap", generator="insight")

    with pytest.raises(ValueError, match="only bona fide crops are super-resolved, got face_swap"):
        superresolved.super_resolved(fake, "realesrgan_x4plus", "sr/0.jpg")


def test_an_unknown_network_is_rejected(make_sample):
    with pytest.raises(ValueError, match="unknown super-resolution network 'bicubic'"):
        superresolved.super_resolved(make_sample(), "bicubic", "sr/0.jpg")


def test_network_for_is_stable_and_uses_every_network(make_sample):
    samples = [make_sample(path=f"{i}.jpg") for i in range(200)]

    chosen = [superresolved.network_for(s) for s in samples]

    assert chosen == [superresolved.network_for(s) for s in samples]
    assert set(chosen) == set(superresolved.NETWORKS)
