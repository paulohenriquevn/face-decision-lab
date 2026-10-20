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


def _scored(make_sample):
    # Bona fide at 0.1-0.4; generator "easy" above every bona fide score, "hard" below all of them.
    samples = [
        make_sample(attack_type="bona_fide", generator="real", path=f"r{i}.jpg", video=f"r{i}")
        for i in range(4)
    ]
    for generator in ("easy", "hard"):
        samples += [
            make_sample(
                attack_type="face_swap",
                generator=generator,
                path=f"{generator}{i}.jpg",
                video=f"{generator}{i}",
            )
            for i in range(2)
        ]
    scores = [0.1, 0.2, 0.3, 0.4, 0.9, 0.8, 0.05, 0.0]
    return samples, scores


def test_auc_by_group_ranks_each_group_of_attacks_against_every_bona_fide(make_sample):
    from face_decision_lab.protocol import auc_by_group

    samples, scores = _scored(make_sample)

    assert auc_by_group(samples, scores) == {"easy": 1.0, "hard": 0.0}


def test_auc_by_group_merges_generators_of_one_group(make_sample):
    from face_decision_lab.protocol import auc_by_group

    samples, scores = _scored(make_sample)

    assert auc_by_group(samples, scores, group=lambda generator: "all") == {"all": 0.5}


def test_evaluation_report_has_the_auc_of_each_generator(make_sample):
    samples, scores = _scored(make_sample)

    assert evaluation_report(samples, scores)["auc_by_generator"] == {"easy": 1.0, "hard": 0.0}


def test_cross_diffusionface_tests_on_every_diffusionface_crop(make_sample):
    lab, _ = _with_celebdf(make_sample, 900, 1)
    faces = [
        make_sample(
            attack_type="bona_fide" if i % 2 else "fully_synthetic",
            subject=f"diffusionface_{i}",
            video=f"diffusionface/{i}",
            source="diffusionface",
            path=f"diffusionface/{i}.jpg",
            generator="diffusionface_real" if i % 2 else "diffusionface_adm",
        )
        for i in range(40)
    ]

    train, calibration, test = protocol_split(lab + faces, "cross_diffusionface")

    assert test == faces
    assert "diffusionface" not in {s.source_dataset for s in train + calibration}
    assert protocol_split(lab + faces, "subject_holdout") == protocol_split(lab, "subject_holdout")


def _super_resolved_copies(samples):
    from face_decision_lab.sources import superresolved

    return [
        superresolved.super_resolved(s, superresolved.network_for(s), f"sr/{s.image_path}")
        for s in samples
        if s.is_bona_fide
    ]


def test_lab_ffpp_real_sr_adds_copies_of_the_training_side_reals_only(make_sample):
    from face_decision_lab.pairs import pair_key

    lab, celebdf = _with_celebdf(make_sample, 900, 1)
    ffpp = _ffpp(make_sample)
    copies = _super_resolved_copies(lab + ffpp)  # of every real crop, test side included

    train, calibration, test = protocol_split(lab + ffpp + celebdf + copies, "lab_ffpp_real_sr")

    base_train, base_calibration, base_test = protocol_split(lab + ffpp + celebdf, "lab_ffpp_real")
    added = [s for s in train if s.source_dataset == "superresolved"]
    trained_reals = [s for s in base_train if s.is_bona_fide]
    assert len(added) == len(trained_reals)
    assert {pair_key(s) for s in added} == {pair_key(s) for s in trained_reals}
    assert [s for s in train if s.source_dataset != "superresolved"] == base_train
    assert (calibration, test) == (base_calibration, base_test)


def test_lab_protocols_ignore_super_resolved_copies(make_sample):
    lab = _samples(make_sample, 900)

    with_copies = protocol_split(lab + _super_resolved_copies(lab), "subject_holdout")

    assert with_copies == protocol_split(lab, "subject_holdout")


def test_lab_ffpp_real_sr_needs_the_super_resolved_build():
    from face_decision_lab.protocol import external_datasets

    assert external_datasets("lab_ffpp_real_sr") == {"faceforensics", "superresolved"}


def _df40(make_sample):
    from face_decision_lab.sources import df40, faceforensics

    names = []
    for split in ("train", "val", "test"):
        for target, source in faceforensics.OFFICIAL_SPLITS[split][:3]:
            names += [("train", "simswap", f"simswap/frames/{target}_{source}/00{i}.png") for i in range(2)]
            names += [("train", "sd2.1", f"sd2.1/{target}/00{i}_325.png") for i in range(2)]
            names += [
                ("real", None, f"FaceForensics++/original_sequences/youtube/c23/frames/{target}/00{i}.png")
                for i in range(2)
            ]
            names += [("unknown", "e4e", f"e4e/e4e/ff/{target}/00{i}.jpg") for i in range(2)]
    names += [
        ("unknown", "MidJourney", f"MidJourney/{label}/{i}.jpg")
        for label in ("fake", "real")
        for i in range(3)
    ]
    parse = {
        "train": df40.parse_train_member,
        "real": lambda _, name: df40.parse_real_member(name),
        "unknown": df40.parse_unknown_member,
    }
    items = [parse[kind](archive, name) for kind, archive, name in names]
    return [
        make_sample(
            attack_type=i.attack_type,
            subject=i.subject_id,
            video=i.video_id,
            source=i.source_dataset,
            path=i.path,
            generator=i.generator,
        )
        for i in items
    ]


def test_df40_ff_trains_on_known_methods_of_training_sequences_and_tests_on_unknown_methods(make_sample):
    from face_decision_lab.sources import df40

    train, calibration, test = protocol_split(_df40(make_sample), "df40_ff")

    assert {df40.official_split(s) for s in train} == {"train"}
    assert {df40.official_split(s) for s in calibration} == {"val"}
    assert {s.generator for s in train} == {"df40_simswap", "df40_sd2.1", "df40_ffpp_real"}
    assert {s.generator for s in test} == {
        "df40_e4e",
        "df40_ffpp_real",
        "df40_midjourney",
        "df40_midjourney_real",
    }
    # e4e inverts FF++ frames: only its test sequences, scored against the real frames of those.
    assert {df40.official_split(s) for s in test if s.subject_id.startswith("ffpp_")} == {"test"}


def test_lab_df40_adds_df40_training_frames_to_lab_ffpp_real(make_sample):
    lab, celebdf = _with_celebdf(make_sample, 900, 1)
    ffpp, df40_samples = _ffpp(make_sample), _df40(make_sample)

    train, calibration, test = protocol_split(lab + ffpp + celebdf + df40_samples, "lab_df40")

    base_train, base_calibration, base_test = protocol_split(lab + ffpp + celebdf, "lab_ffpp_real")
    df40_train, df40_calibration, _ = protocol_split(df40_samples, "df40_ff")
    assert train == base_train + df40_train
    assert calibration == base_calibration + df40_calibration
    assert test == base_test


def test_df40_protocols_load_their_external_datasets():
    from face_decision_lab.protocol import external_datasets

    assert external_datasets("df40_ff") == {"df40"}
    assert external_datasets("lab_df40") == {"faceforensics", "df40"}


def test_auc_within_groups_scores_each_method_against_its_own_reals(make_sample):
    from face_decision_lab.protocol import auc_within_groups

    samples = [
        make_sample(attack_type="bona_fide", generator="a_real", video="1"),
        make_sample(attack_type="fully_synthetic", generator="a", video="2"),
        make_sample(attack_type="bona_fide", generator="b_real", video="3"),
        make_sample(attack_type="fully_synthetic", generator="b", video="4"),
    ]
    # b's fake outscores b's real but not a's real: only b's own reals count.
    scores = [0.9, 0.95, 0.1, 0.5]

    result = auc_within_groups(samples, scores, lambda s: s.generator.removesuffix("_real"))

    assert result == {"a": 1.0, "b": 1.0}


def test_auc_within_groups_rejects_a_group_without_both_classes(make_sample):
    from face_decision_lab.protocol import auc_within_groups

    samples = [make_sample(attack_type="fully_synthetic", generator="a", video="1")]

    with pytest.raises(MetricInputError, match="group 'a' needs bona fide and attack samples"):
        auc_within_groups(samples, [0.5], lambda s: s.generator)


def _separable(make_sample, subjects=40, per_subject=5, overlap=0.0):
    rng = np.random.default_rng(0)
    samples, scores = [], []
    for subject in range(subjects):
        real = subject % 2 == 0
        for frame in range(per_subject):
            samples.append(
                make_sample(
                    attack_type="bona_fide" if real else "face_swap",
                    subject=f"s{subject}",
                    video=f"v{subject}_{frame}",
                    path=f"{subject}_{frame}.jpg",
                )
            )
            scores.append(float(rng.normal(0.0 if real else 1.0 - overlap, 0.3)))
    return samples, scores


def test_bootstrap_interval_brackets_the_auc_and_is_reproducible(make_sample):
    from face_decision_lab.metrics import auc
    from face_decision_lab.protocol import bootstrap_interval

    samples, scores = _separable(make_sample, overlap=0.7)

    def statistic(part, values):
        return auc(values, [s.attack_type for s in part])

    low, high = bootstrap_interval(samples, scores, statistic, rounds=200)

    assert low < statistic(samples, scores) < high
    assert (low, high) == bootstrap_interval(samples, scores, statistic, rounds=200)


def test_bootstrap_interval_resamples_whole_subjects(make_sample):
    from face_decision_lab.protocol import bootstrap_interval

    samples, scores = _separable(make_sample)
    seen = []

    def statistic(part, values):
        by_subject = {}
        for sample in part:
            by_subject[sample.subject_id] = by_subject.get(sample.subject_id, 0) + 1
        seen.append(set(by_subject.values()))
        return 0.0

    bootstrap_interval(samples, scores, statistic, rounds=20)

    # Every subject has 5 crops, so a subject drawn k times contributes 5k of them.
    assert all(count % 5 == 0 for counts in seen for count in counts)


def test_bootstrap_interval_rejects_a_level_outside_zero_and_one(make_sample):
    from face_decision_lab.protocol import bootstrap_interval

    samples, scores = _separable(make_sample)

    with pytest.raises(ValueError, match="level must be between 0 and 1, got 1.5"):
        bootstrap_interval(samples, scores, lambda part, values: 0.0, level=1.5)


def test_ffpp_deepfakes_to_diff_trains_on_ffpp_deepfakes_and_tests_on_diff(make_sample):
    from face_decision_lab.sources import faceforensics

    diff = _diff(make_sample, 40)

    train, calibration, test = protocol_split(_ffpp(make_sample) + diff, "ffpp_deepfakes_to_diff")

    assert {s.generator for s in train} == {"ffpp_original", "ffpp_deepfakes"}
    assert {s.generator for s in calibration} == {"ffpp_original", "ffpp_deepfakes"}
    assert {faceforensics.official_split(s) for s in train} == {"train"}
    assert {faceforensics.official_split(s) for s in calibration} == {"val"}
    assert test == diff


def test_ffpp_deepfakes_to_diff_loads_faceforensics_and_diff():
    from face_decision_lab.protocol import external_datasets

    assert external_datasets("ffpp_deepfakes_to_diff") == {"faceforensics", "diff"}


def _scored_side(make_sample, real_scores, attack_scores, prefix):
    samples = [
        make_sample(subject=f"{prefix}r{i}", video=f"{prefix}rv{i}", path=f"{prefix}r{i}.jpg")
        for i in range(len(real_scores))
    ]
    samples += [
        make_sample(
            attack_type="face_swap",
            subject=f"{prefix}a{i}",
            video=f"{prefix}av{i}",
            path=f"{prefix}a{i}.jpg",
            generator="insight" if i % 2 else "text2img",
        )
        for i in range(len(attack_scores))
    ]
    return samples, list(real_scores) + list(attack_scores)


def test_detection_at_bpcer_sets_the_threshold_on_calibration_and_reads_detection_on_test(make_sample):
    from face_decision_lab.protocol import detection_at_bpcer

    calibration, calibration_scores = _scored_side(make_sample, np.arange(1000) / 1000, [0.5], "c")
    test, test_scores = _scored_side(make_sample, [0.1, 0.997], [0.2, 0.996, 0.999, 1.0], "t")

    report = detection_at_bpcer(calibration, calibration_scores, test, test_scores, max_bpcers=(0.005,))

    point = report["0.005"]
    assert 0.994 < point["threshold"] <= 0.995
    assert point["calibration_bpcer"] == 0.005
    # The test's own reals are not used for the threshold: one of two is flagged.
    assert point["test_bpcer"] == 0.5
    assert point["detection"] == 0.75
    assert point["detection_by_generator"] == {"insight": 1.0, "text2img": 0.5}


def test_detection_at_bpcer_threshold_ignores_the_test_reals(make_sample):
    from face_decision_lab.protocol import detection_at_bpcer

    calibration, calibration_scores = _scored_side(make_sample, np.arange(1000) / 1000, [], "c")
    test, test_scores = _scored_side(make_sample, [0.1, 0.2], [0.999], "t")
    other_test, other_scores = _scored_side(make_sample, [0.99, 0.999], [0.999], "t")

    limit = (0.002,)
    first = detection_at_bpcer(calibration, calibration_scores, test, test_scores, max_bpcers=limit)
    second = detection_at_bpcer(calibration, calibration_scores, other_test, other_scores, max_bpcers=limit)

    assert first["0.002"]["threshold"] == second["0.002"]["threshold"]


def test_detection_at_bpcer_needs_calibration_reals_and_test_attacks(make_sample):
    from face_decision_lab.protocol import detection_at_bpcer

    reals, real_scores = _scored_side(make_sample, [0.1, 0.2], [], "r")
    attacks, attack_scores = _scored_side(make_sample, [], [0.9], "a")

    with pytest.raises(MetricInputError):
        detection_at_bpcer(attacks, attack_scores, attacks, attack_scores)
    with pytest.raises(MetricInputError):
        detection_at_bpcer(reals, real_scores, reals, real_scores)


def _real_photos(make_sample, source, count):
    return [
        make_sample(subject=f"{source}_{i}", video=f"{source}/{i}", source=source, path=f"{source}/{i}.png",
                    generator=f"{source}_real")
        for i in range(count)
    ]


def test_lab_ffpp_real_ffhq_adds_every_ffhq_photo_to_training_only(make_sample):
    lab, celebdf = _with_celebdf(make_sample, 900, 1)
    ffpp = _ffpp(make_sample)
    ffhq = _real_photos(make_sample, "ffhq", 50)

    train, calibration, test = protocol_split(lab + ffpp + celebdf + ffhq, "lab_ffpp_real_ffhq")

    base_train, base_calibration, base_test = protocol_split(lab + ffpp + celebdf, "lab_ffpp_real")
    assert [s for s in train if s.source_dataset != "ffhq"] == base_train
    assert [s for s in train if s.source_dataset == "ffhq"] == ffhq
    assert (calibration, test) == (base_calibration, base_test)


def test_lab_ffpp_real_ffhq_refuses_an_ffhq_crop_that_is_not_bona_fide(make_sample):
    lab, celebdf = _with_celebdf(make_sample, 900, 1)
    fake = make_sample(attack_type="fully_synthetic", source="ffhq", path="ffhq/x.png", generator="ffhq_real")

    with pytest.raises(ValueError, match="bona fide"):
        protocol_split(lab + _ffpp(make_sample) + celebdf + [fake], "lab_ffpp_real_ffhq")


def test_lab_protocols_ignore_ffhq_photos(make_sample):
    lab = _samples(make_sample, 900)

    assert protocol_split(lab + _real_photos(make_sample, "ffhq", 30), "subject_holdout") == protocol_split(
        lab, "subject_holdout"
    )


def test_lab_ffpp_real_ffhq_needs_faceforensics_and_ffhq():
    from face_decision_lab.protocol import external_datasets

    assert external_datasets("lab_ffpp_real_ffhq") == {"faceforensics", "ffhq"}
