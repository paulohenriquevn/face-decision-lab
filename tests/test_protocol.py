import numpy as np
import pytest

from face_decision_lab.baselines import LinearProbe
from face_decision_lab.errors import MetricInputError
from face_decision_lab.protocol import MAX_TEST_SAMPLES, evaluation_report, protocol_split


def _samples(make_sample, count):
    kinds = ["bona_fide", "face_swap", "fully_synthetic"]
    return [
        make_sample(
            attack_type=kinds[i % 3],
            subject=f"s{i // 3}",
            video=f"v{i}",
            path=f"{i}.jpg",
            generator={"bona_fide": "wiki", "face_swap": "insight", "fully_synthetic": "text2img"}[
                kinds[i % 3]
            ],
        )
        for i in range(count)
    ]


def test_protocol_split_caps_the_test_side(make_sample):
    samples = _samples(make_sample, 3 * 10_000)

    _, _, test = protocol_split(samples)

    assert len(test) == MAX_TEST_SAMPLES


def test_protocol_split_is_identical_across_calls(make_sample):
    samples = _samples(make_sample, 600)

    assert protocol_split(samples) == protocol_split(samples)


def test_evaluation_report_breaks_apcer_down_by_generator(make_sample):
    samples = _samples(make_sample, 6)
    scores = [0.1, 0.9, 0.2, 0.15, 0.8, 0.95]

    report = evaluation_report(samples, scores)

    assert set(report["apcer_by_generator"]) == {"insight", "text2img"}


def test_evaluation_report_gives_auc_one_for_perfect_scores(make_sample):
    samples = _samples(make_sample, 6)
    scores = [0.0 if s.is_bona_fide else 1.0 for s in samples]

    assert evaluation_report(samples, scores)["auc"] == 1.0


def test_linear_probe_separates_linearly_separable_features():
    rng = np.random.default_rng(0)
    bona_fide = rng.normal(-2, 1, size=(200, 8))
    attack = rng.normal(2, 1, size=(200, 8))
    features = np.vstack([bona_fide, attack])
    is_attack = np.array([False] * 200 + [True] * 200)

    scores = LinearProbe().fit(features, is_attack).attack_scores(features)

    assert scores[200:].min() > scores[:200].max()


def test_linear_probe_rejects_a_single_class():
    with pytest.raises(MetricInputError, match="both bona fide and attack"):
        LinearProbe().fit(np.zeros((4, 2)), np.ones(4, dtype=bool))


def test_unseen_generator_protocol_removes_it_from_training(make_sample):
    samples = _samples(make_sample, 900)

    train, calibration, _ = protocol_split(samples, "unseen_text2img")

    assert "text2img" not in {s.generator for s in train + calibration}


def test_unseen_generator_protocol_tests_only_bona_fide_and_that_generator(make_sample):
    samples = _samples(make_sample, 900)

    _, _, test = protocol_split(samples, "unseen_text2img")

    assert {s.generator for s in test} == {"wiki", "text2img"}


def test_unseen_generator_protocol_keeps_the_subject_holdout(make_sample):
    samples = _samples(make_sample, 900)

    train, _, test = protocol_split(samples, "unseen_text2img")

    assert {s.subject_id for s in train}.isdisjoint({s.subject_id for s in test})


def test_protocol_split_rejects_an_unknown_protocol(make_sample):
    with pytest.raises(ValueError, match="unknown protocol"):
        protocol_split(_samples(make_sample, 30), "cross_universe")


def _with_celebdf(make_sample, lab_count, celebdf_videos):
    celebdf = [
        make_sample(
            attack_type="face_swap" if v % 2 else "bona_fide",
            subject=f"celebdf_id{v}",
            video=f"celebdf/{v}",
            source="celebdf",
            path=f"celebdf/{v}_{frame}.jpg",
            generator="celebdf_synthesis" if v % 2 else "celeb_real",
        )
        for v in range(celebdf_videos)
        for frame in range(16)
    ]
    return _samples(make_sample, lab_count), celebdf


def test_cross_dataset_protocol_trains_without_the_held_out_dataset(make_sample):
    lab, celebdf = _with_celebdf(make_sample, 900, 10)

    train, calibration, _ = protocol_split(lab + celebdf, "cross_celebdf")

    assert "celebdf" not in {s.source_dataset for s in train + calibration}


def test_cross_dataset_protocol_tests_on_every_crop_of_the_held_out_dataset(make_sample):
    lab, celebdf = _with_celebdf(make_sample, 900, 300)

    _, _, test = protocol_split(lab + celebdf, "cross_celebdf")

    assert test == celebdf


def test_cross_dataset_protocol_trains_on_the_subject_holdout_training_side(make_sample):
    lab, celebdf = _with_celebdf(make_sample, 900, 10)

    train, calibration, _ = protocol_split(lab + celebdf, "cross_celebdf")

    assert (train, calibration) == protocol_split(lab, "subject_holdout")[:2]


def test_cross_dataset_protocol_needs_the_held_out_dataset(make_sample):
    with pytest.raises(ValueError, match="no 'celebdf' samples"):
        protocol_split(_samples(make_sample, 900), "cross_celebdf")


def test_evaluation_report_scores_videos_by_their_mean_frame_score(make_sample):
    samples = [
        make_sample(attack_type="bona_fide", video="real", path="r0.jpg"),
        make_sample(attack_type="bona_fide", video="real", path="r1.jpg"),
        make_sample(attack_type="face_swap", video="fake", path="f0.jpg", generator="insight"),
        make_sample(attack_type="face_swap", video="fake", path="f1.jpg", generator="insight"),
    ]
    # One real frame outscores one fake frame, so frame AUC < 1; the video means are 0.35 and 0.65.
    scores = [0.1, 0.6, 0.5, 0.8]

    report = evaluation_report(samples, scores)

    assert report["auc"] < 1.0
    assert report["video_auc"] == 1.0
    assert report["videos"] == 2


@pytest.mark.parametrize("protocol", ["subject_holdout", "unseen_text2img"])
def test_standard_protocols_ignore_test_only_datasets(make_sample, protocol):
    # Regression: with Celeb-DF crops loaded next to the lab data, subject_holdout drew its split
    # over both, so its test side held Celeb-DF crops and lab subjects from the training side.
    lab, celebdf = _with_celebdf(make_sample, 900, 10)

    assert protocol_split(lab + celebdf, protocol) == protocol_split(lab, protocol)


def _ffpp(make_sample):
    from face_decision_lab.sources import faceforensics

    pairs = [pair for split in ("train", "val", "test") for pair in faceforensics.OFFICIAL_SPLITS[split][:3]]
    paths = []
    for target, source in pairs:
        paths.append(f"original_sequences/youtube/c23/videos/{target}.mp4")
        for method in ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter"):
            paths.append(f"manipulated_sequences/{method}/c23/videos/{target}_{source}.mp4")
    items = [faceforensics.parse_path(p) for p in paths]
    return [
        make_sample(
            attack_type=i.attack_type,
            subject=i.subject_id,
            video=i.video_id,
            source=i.source_dataset,
            path=f"{i.video_id}_{frame}.jpg",
            generator=i.generator,
        )
        for i in items
        for frame in range(2)
    ]


def test_lab_protocols_ignore_faceforensics(make_sample):
    lab, _ = _with_celebdf(make_sample, 900, 1)

    assert protocol_split(lab + _ffpp(make_sample), "subject_holdout") == protocol_split(
        lab, "subject_holdout"
    )


def test_ffpp_protocol_follows_the_official_splits(make_sample):
    from face_decision_lab.sources import faceforensics

    train, calibration, test = protocol_split(_ffpp(make_sample), "ffpp")

    assert {faceforensics.official_split(s) for s in train} == {"train"}
    assert {faceforensics.official_split(s) for s in calibration} == {"val"}
    assert {faceforensics.official_split(s) for s in test} == {"test"}


def test_ffpp_protocol_uses_the_four_original_methods_only(make_sample):
    train, calibration, test = protocol_split(_ffpp(make_sample), "ffpp")

    assert "ffpp_faceshifter" not in {s.generator for s in train + calibration + test}


def test_ffpp_to_celebdf_trains_on_ffpp_and_tests_on_celebdf(make_sample):
    _, celebdf = _with_celebdf(make_sample, 30, 4)
    ffpp = _ffpp(make_sample)

    train, _, test = protocol_split(ffpp + celebdf, "ffpp_to_celebdf")

    assert {s.source_dataset for s in train} == {"faceforensics"}
    assert test == celebdf


def test_lab_ffpp_to_celebdf_trains_on_both_training_sides(make_sample):
    lab, celebdf = _with_celebdf(make_sample, 900, 4)
    ffpp = _ffpp(make_sample)

    train, _, test = protocol_split(lab + ffpp + celebdf, "lab_ffpp_to_celebdf")

    lab_train = protocol_split(lab, "subject_holdout")[0]
    ffpp_train = protocol_split(ffpp, "ffpp")[0]
    assert sorted(train, key=lambda s: s.image_path) == sorted(
        lab_train + ffpp_train, key=lambda s: s.image_path
    )
    assert test == celebdf


def test_ffpp_protocol_needs_faceforensics_samples(make_sample):
    with pytest.raises(ValueError, match="no 'faceforensics' samples"):
        protocol_split(_samples(make_sample, 900), "ffpp")


def test_lab_ffpp_real_adds_only_ffpp_originals_to_the_lab_training_side(make_sample):
    lab, celebdf = _with_celebdf(make_sample, 900, 4)
    ffpp = _ffpp(make_sample)

    train, calibration, test = protocol_split(lab + ffpp + celebdf, "lab_ffpp_real")

    lab_train, lab_calibration, lab_test = protocol_split(lab, "subject_holdout")
    extra_train = [s for s in train if s.source_dataset == "faceforensics"]
    assert extra_train and all(s.generator == "ffpp_original" for s in extra_train)
    assert [s for s in train if s.source_dataset != "faceforensics"] == lab_train
    assert {s.generator for s in calibration if s.source_dataset == "faceforensics"} == {"ffpp_original"}
    assert test == lab_test


def test_lab_ffpp_real_keeps_the_official_ffpp_splits(make_sample):
    from face_decision_lab.sources import faceforensics

    lab, _ = _with_celebdf(make_sample, 900, 1)

    train, calibration, _ = protocol_split(lab + _ffpp(make_sample), "lab_ffpp_real")

    assert {faceforensics.official_split(s) for s in train if s.source_dataset == "faceforensics"} == {
        "train"
    }
    assert {faceforensics.official_split(s) for s in calibration if s.source_dataset == "faceforensics"} == {
        "val"
    }


def test_evaluation_report_gives_the_mean_score_of_each_generator(make_sample):
    samples = [
        make_sample(attack_type="bona_fide", generator="celeb_real", video="a", path="a.jpg"),
        make_sample(attack_type="bona_fide", generator="celeb_real", video="b", path="b.jpg"),
        make_sample(attack_type="face_swap", generator="celebdf_synthesis", video="c", path="c.jpg"),
    ]

    report = evaluation_report(samples, [0.2, 0.4, 0.9])

    assert report["mean_score_by_generator"] == pytest.approx({"celeb_real": 0.3, "celebdf_synthesis": 0.9})


def _diff(make_sample, identities):
    return [
        make_sample(
            attack_type="bona_fide" if i % 2 else "fully_synthetic",
            subject=f"diff_id_{i:04d}",
            video=f"diff/{i}",
            source="diff",
            path=f"diff/{i}.jpg",
            generator="diff_real" if i % 2 else "diff_t2i_sdxl",
        )
        for i in range(identities)
    ]


def test_cross_diff_tests_on_every_diff_crop(make_sample):
    lab, _ = _with_celebdf(make_sample, 900, 1)
    diff = _diff(make_sample, 40)

    train, calibration, test = protocol_split(lab + diff, "cross_diff")

    assert test == diff
    assert "diff" not in {s.source_dataset for s in train + calibration}


def test_lab_protocols_ignore_diff(make_sample):
    lab, _ = _with_celebdf(make_sample, 900, 1)

    assert protocol_split(lab + _diff(make_sample, 40), "subject_holdout") == protocol_split(
        lab, "subject_holdout"
    )
