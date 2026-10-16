import pytest

from face_decision_lab.pairs import pair_key, paired_batches
from face_decision_lab.sources import axondata, deepfakeface


def _dff(generator, person):
    item = deepfakeface.parse_member(f"{generator}/61/{person}_1982-08-16_2009.jpg")
    return item


def _sample(make_sample, item):
    return make_sample(
        attack_type=item.attack_type,
        subject=item.subject_id,
        video=item.video_id,
        source=item.source_dataset,
        generator=item.generator,
        path=item.path,
    )


def test_real_photo_and_its_fakes_share_a_pair_key(make_sample):
    keys = {pair_key(_sample(make_sample, _dff(g, "123"))) for g in deepfakeface.GENERATORS}

    assert keys == {"61/123_1982-08-16_2009.jpg"}


def test_unpaired_sources_keep_their_own_key(make_sample):
    item = axondata.parse_path("Latex_mask/Mask_6/1.mp4")

    assert pair_key(_sample(make_sample, item)) == item.video_id


def test_paired_batches_keep_each_group_in_one_batch(make_sample):
    samples = [_sample(make_sample, _dff(g, str(p))) for p in range(10) for g in deepfakeface.GENERATORS]

    batches = paired_batches(samples, batch_size=8, seed=0)

    for batch in batches:
        keys = {pair_key(s) for s in batch}
        for key in keys:
            assert sum(pair_key(s) == key for s in samples) == sum(pair_key(s) == key for s in batch)


def test_paired_batches_use_every_sample_exactly_once(make_sample):
    samples = [_sample(make_sample, _dff(g, str(p))) for p in range(10) for g in deepfakeface.GENERATORS]

    batches = paired_batches(samples, batch_size=8, seed=0)

    flat = [s for batch in batches for s in batch]
    assert sorted(flat, key=lambda s: s.image_path) == sorted(samples, key=lambda s: s.image_path)


def test_paired_batches_never_exceed_the_batch_size(make_sample):
    samples = [_sample(make_sample, _dff(g, str(p))) for p in range(10) for g in deepfakeface.GENERATORS]

    assert max(len(b) for b in paired_batches(samples, batch_size=6, seed=1)) <= 6


def test_paired_batches_are_reproducible(make_sample):
    samples = [_sample(make_sample, _dff(g, str(p))) for p in range(10) for g in deepfakeface.GENERATORS]

    assert paired_batches(samples, 8, seed=3) == paired_batches(samples, 8, seed=3)


def test_paired_batches_reject_a_non_positive_size(make_sample):
    with pytest.raises(ValueError, match="batch_size must be positive"):
        paired_batches([], batch_size=0, seed=0)
