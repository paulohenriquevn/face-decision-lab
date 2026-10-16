import pytest

from face_decision_lab.errors import MetricInputError
from face_decision_lab.metrics import auc, bpcer_at_apcer, eer, operating_point

# Two bona fide captures, two prints, two replays. Higher score = more likely an attack.
SCORES = [0.1, 0.6, 0.9, 0.3, 0.8, 0.7]
TYPES = ["bona_fide", "bona_fide", "print", "print", "replay", "replay"]


def test_apcer_is_reported_for_the_worst_attack_type():
    point = operating_point(SCORES, TYPES, threshold=0.5)

    # One of two prints (0.3) passes as bona fide; no replay does.
    assert point.apcer_by_type == {"print": 0.5, "replay": 0.0}
    assert point.apcer == 0.5


def test_bpcer_counts_bona_fide_flagged_as_attack():
    point = operating_point(SCORES, TYPES, threshold=0.5)

    assert point.bpcer == 0.5


def test_acer_is_the_mean_of_apcer_and_bpcer():
    point = operating_point(SCORES, TYPES, threshold=0.5)

    assert point.acer == 0.5


def test_bpcer_at_apcer_picks_the_highest_threshold_meeting_the_limit():
    point = bpcer_at_apcer(SCORES, TYPES, max_apcer=0.0)

    # The lowest print scores 0.3, so 0.3 is the highest threshold that catches every attack.
    assert point.threshold == 0.3
    assert point.apcer == 0.0
    assert point.bpcer == 0.5


def test_bpcer_at_apcer_matches_a_brute_force_search():
    scores = [0.05, 0.2, 0.35, 0.4, 0.55, 0.6, 0.62, 0.7, 0.9, 0.95]
    types = [
        "bona_fide",
        "print",
        "bona_fide",
        "replay",
        "print",
        "bona_fide",
        "replay",
        "print",
        "replay",
        "print",
    ]
    candidates = [operating_point(scores, types, t) for t in sorted(set(scores))]
    expected = min((p for p in candidates if p.apcer <= 0.34), key=lambda p: (p.bpcer, -p.threshold))

    point = bpcer_at_apcer(scores, types, max_apcer=0.34)

    assert (point.threshold, point.bpcer) == (expected.threshold, expected.bpcer)


SEPARATED_SCORES = [0.1, 0.2, 0.8, 0.9]
SEPARATED_TYPES = ["bona_fide", "bona_fide", "print", "replay"]


def test_perfect_separation_gives_auc_one():
    assert auc(SEPARATED_SCORES, SEPARATED_TYPES) == 1.0


def test_perfect_separation_gives_eer_zero():
    assert eer(SEPARATED_SCORES, SEPARATED_TYPES) == 0.0


def test_metrics_reject_labels_without_bona_fide():
    with pytest.raises(MetricInputError, match="both bona fide and attack"):
        auc([0.1, 0.9], ["print", "replay"])


def test_metrics_reject_nan_scores():
    with pytest.raises(MetricInputError, match="NaN"):
        operating_point([float("nan"), 0.5], ["bona_fide", "print"], threshold=0.5)


def test_metrics_reject_mismatched_lengths():
    with pytest.raises(MetricInputError, match="scores for"):
        auc([0.1, 0.2, 0.3], ["bona_fide", "print"])
