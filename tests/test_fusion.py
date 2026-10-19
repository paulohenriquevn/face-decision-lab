import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from face_decision_lab.errors import MetricInputError
from face_decision_lab.fusion import ScoreFusion, balancing_weights, score_logits


def test_score_logits_stay_finite_at_zero_and_one():
    logits = score_logits(np.array([[0.0, 1.0], [0.5, 0.5]]))

    assert np.isfinite(logits).all()
    assert logits[1].tolist() == [0.0, 0.0]


def test_score_logits_keep_the_order_of_the_scores():
    logits = score_logits(np.array([[0.1], [0.4], [0.9]]))

    assert (np.diff(logits[:, 0]) > 0).all()


def test_balancing_weights_give_every_domain_and_class_the_same_total(make_sample):
    samples = (
        [make_sample(source="deepfakeface", attack_type="bona_fide")] * 2
        + [make_sample(source="deepfakeface", attack_type="face_swap")] * 6
        + [make_sample(source="faceforensics", attack_type="bona_fide")] * 5
        + [make_sample(source="faceforensics", attack_type="face_swap")] * 20
    )

    weights = balancing_weights(samples)

    totals = {}
    for sample, weight in zip(samples, weights, strict=True):
        key = (sample.source_dataset, sample.is_bona_fide)
        totals[key] = totals.get(key, 0.0) + weight
    assert len(totals) == 4
    assert np.allclose(list(totals.values()), list(totals.values())[0])


def test_balancing_weights_treat_all_lab_sources_as_one_domain(make_sample):
    samples = [
        make_sample(source="deepfakeface", attack_type="bona_fide"),
        make_sample(source="axondata", attack_type="bona_fide"),
        make_sample(source="deepfakeface", attack_type="face_swap"),
    ]

    weights = balancing_weights(samples)

    assert weights[0] == weights[1]
    assert weights[0] + weights[1] == pytest.approx(weights[2])


def _two_domains(seed):
    # Detector 1 separates domain A and is noise on domain B; detector 2 is the reverse.
    rng = np.random.default_rng(seed)
    n = 400
    labels = np.array([0, 1] * n)
    signal = np.where(labels == 1, 3.0, -3.0)
    a = np.column_stack([signal + rng.normal(0, 1, 2 * n), rng.normal(0, 1, 2 * n)])
    b = np.column_stack([rng.normal(0, 1, 2 * n), signal + rng.normal(0, 1, 2 * n)])
    to_probability = lambda x: 1 / (1 + np.exp(-x))  # noqa: E731
    return to_probability(a), to_probability(b), labels.astype(bool)


def test_fusion_keeps_each_detector_strong_on_its_own_domain():
    a, b, labels = _two_domains(0)
    test_a, test_b, test_labels = _two_domains(1)

    fusion = ScoreFusion().fit(np.vstack([a, b]), np.concatenate([labels, labels]))

    assert roc_auc_score(test_labels, fusion.attack_scores(test_a)) > 0.95
    assert roc_auc_score(test_labels, fusion.attack_scores(test_b)) > 0.95


def test_fusion_rejects_a_single_class():
    with pytest.raises(MetricInputError, match="both bona fide and attack"):
        ScoreFusion().fit(np.full((4, 2), 0.5), np.ones(4, dtype=bool))


def test_fusion_rejects_scores_that_do_not_match_the_labels():
    with pytest.raises(MetricInputError, match="got \\(3, 2\\) scores for \\(4,\\) labels"):
        ScoreFusion().fit(np.full((3, 2), 0.5), np.array([True, False, True, False]))


def test_gated_mixture_follows_the_gate():
    from face_decision_lab.fusion import gated_mixture

    gate = np.array([1.0, 0.0, 0.5])
    lab_expert = np.array([0.9, 0.9, 0.9])
    video_expert = np.array([0.1, 0.1, 0.1])

    assert gated_mixture(gate, lab_expert, video_expert).tolist() == pytest.approx([0.9, 0.1, 0.5])


def test_gated_mixture_rejects_a_gate_outside_zero_one():
    from face_decision_lab.fusion import gated_mixture

    with pytest.raises(MetricInputError, match="gate values must lie in"):
        gated_mixture(np.array([1.2]), np.array([0.5]), np.array([0.5]))


def test_gated_mixture_rejects_mismatched_lengths():
    from face_decision_lab.fusion import gated_mixture

    with pytest.raises(MetricInputError, match="got 2 gate values, 1 and 2 scores"):
        gated_mixture(np.array([0.5, 0.5]), np.array([0.5]), np.array([0.5, 0.5]))


def test_in_lab_domain_marks_lab_sources_only(make_sample):
    from face_decision_lab.fusion import in_lab_domain

    samples = [make_sample(source=s) for s in ("deepfakeface", "axondata", "faceforensics", "celebdf")]

    assert in_lab_domain(samples).tolist() == [True, True, False, False]
