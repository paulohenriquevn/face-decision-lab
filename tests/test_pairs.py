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


def test_faceforensics_original_and_its_manipulations_share_a_pair_key(make_sample):
    from face_decision_lab.sources import faceforensics

    paths = [
        "original_sequences/youtube/c23/videos/071.mp4",
        "manipulated_sequences/Deepfakes/c23/videos/071_054.mp4",
        "manipulated_sequences/Face2Face/c23/videos/071_054.mp4",
    ]

    keys = {pair_key(_sample(make_sample, faceforensics.parse_path(p))) for p in paths}

    assert keys == {"ffpp/071"}


def test_faceforensics_videos_of_different_targets_keep_apart(make_sample):
    from face_decision_lab.sources import faceforensics

    first = faceforensics.parse_path("manipulated_sequences/Deepfakes/c23/videos/071_054.mp4")
    second = faceforensics.parse_path("manipulated_sequences/Deepfakes/c23/videos/054_071.mp4")

    assert pair_key(_sample(make_sample, first)) != pair_key(_sample(make_sample, second))


def _ffpp_group(make_sample, target, frames):
    from face_decision_lab.sources import faceforensics

    paths = [f"original_sequences/youtube/c23/videos/{target}.mp4"] + [
        f"manipulated_sequences/{m}/c23/videos/{target}_999.mp4"
        for m in ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures")
    ]
    out = []
    for path in paths:
        item = faceforensics.parse_path(path)
        out += [
            make_sample(
                attack_type=item.attack_type,
                subject=item.subject_id,
                video=item.video_id,
                source=item.source_dataset,
                path=f"{item.video_id}_{frame}.jpg",
                generator=item.generator,
            )
            for frame in range(frames)
        ]
    return out


def test_large_groups_split_so_a_batch_holds_many_videos(make_sample):
    # Regression: an FF++ group (an original and its four manipulations, 16 frames each) is 80 crops,
    # so whole groups gave one target video per batch and the detector learned the scene.
    samples = [s for t in range(20) for s in _ffpp_group(make_sample, f"{t:03d}", 16)]

    def mean_videos_per_batch(batches):
        return sum(len({pair_key(s) for s in b}) for b in batches) / len(batches)

    whole = paired_batches(samples, batch_size=40, seed=0)
    split = paired_batches(samples, batch_size=40, seed=0, max_group=10)

    assert mean_videos_per_batch(whole) == 1
    assert mean_videos_per_batch(split) > 3


def test_split_groups_still_use_every_sample_once(make_sample):
    samples = [s for t in ("001", "002") for s in _ffpp_group(make_sample, t, 16)]

    flat = [s for batch in paired_batches(samples, batch_size=40, seed=0, max_group=10) for s in batch]

    assert sorted(flat, key=lambda s: s.image_path) == sorted(samples, key=lambda s: s.image_path)


def test_max_group_leaves_small_groups_whole(make_sample):
    samples = [_sample(make_sample, _dff(g, str(p))) for p in range(10) for g in deepfakeface.GENERATORS]

    assert paired_batches(samples, 8, seed=3, max_group=10) == paired_batches(samples, 8, seed=3)


def test_df40_frames_pair_with_the_ffpp_sequence_they_were_made_from(make_sample):
    from face_decision_lab.sources import df40, faceforensics

    items = [
        df40.parse_train_member("simswap", "simswap/frames/071_054/001.png"),
        df40.parse_train_member("sd2.1", "sd2.1/071/001_325.png"),
        df40.parse_real_member("FaceForensics++/original_sequences/youtube/c23/frames/071/277.png"),
        faceforensics.parse_path("original_sequences/youtube/c23/videos/071.mp4"),
    ]

    assert {pair_key(_sample(make_sample, item)) for item in items} == {"ffpp/071"}


def test_df40_unknown_images_keep_their_own_pair_key(make_sample):
    from face_decision_lab.sources import df40

    item = df40.parse_unknown_member("MidJourney", "MidJourney/real/11180.jpg")

    assert pair_key(_sample(make_sample, item)) == item.video_id
