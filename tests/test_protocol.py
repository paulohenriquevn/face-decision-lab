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
