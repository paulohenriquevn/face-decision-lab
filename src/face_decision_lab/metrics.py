"""Presentation attack detection metrics, following the ISO/IEC 30107-3 definitions.

A score is the probability that a capture is an attack. A capture is classified as an attack when
its score is at or above the threshold.

- APCER of one attack type: share of that type's attacks classified as bona fide.
- APCER (reported): the worst attack type, as ISO/IEC 30107-3 asks.
- BPCER: share of bona fide captures classified as attacks.
- ACER: mean of APCER and BPCER. Not an ISO metric, but common in face anti-spoofing papers.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from .errors import MetricInputError
from .labels import BONA_FIDE


@dataclass(frozen=True)
class OperatingPoint:
    threshold: float
    apcer_by_type: dict[str, float]
    apcer: float
    bpcer: float

    @property
    def acer(self) -> float:
        return (self.apcer + self.bpcer) / 2


def _validated(scores, attack_types) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=float)
    attack_types = np.asarray(attack_types, dtype=object)
    if scores.shape != attack_types.shape or scores.ndim != 1:
        raise MetricInputError(f"got {scores.shape} scores for {attack_types.shape} labels")
    if not np.all(np.isfinite(scores)):
        raise MetricInputError("scores contain NaN or infinity")
    bona_fide = attack_types == BONA_FIDE
    if bona_fide.all() or not bona_fide.any():
        raise MetricInputError("need both bona fide and attack samples")
    return scores, attack_types


def operating_point(scores, attack_types, threshold: float) -> OperatingPoint:
    scores, attack_types = _validated(scores, attack_types)
    flagged = scores >= threshold
    apcer_by_type = {
        kind: float(np.mean(~flagged[attack_types == kind]))
        for kind in sorted(set(attack_types) - {BONA_FIDE})
    }
    return OperatingPoint(
        threshold=float(threshold),
        apcer_by_type=apcer_by_type,
        apcer=max(apcer_by_type.values()),
        bpcer=float(np.mean(flagged[attack_types == BONA_FIDE])),
    )


def bpcer_at_apcer(scores, attack_types, max_apcer: float) -> OperatingPoint:
    """The lowest-BPCER operating point whose worst-type APCER stays at or below max_apcer."""
    scores, attack_types = _validated(scores, attack_types)
    thresholds = np.unique(scores)
    # APCER of each type at every candidate threshold: attacks scoring below it pass as bona fide.
    worst_apcer = np.zeros(len(thresholds))
    for kind in set(attack_types) - {BONA_FIDE}:
        kind_scores = np.sort(scores[attack_types == kind])
        apcer = np.searchsorted(kind_scores, thresholds, side="left") / len(kind_scores)
        worst_apcer = np.maximum(worst_apcer, apcer)
    # BPCER only grows as the threshold drops, so the highest threshold within the APCER limit wins.
    allowed = np.flatnonzero(worst_apcer <= max_apcer)
    if not len(allowed):
        raise MetricInputError(f"no threshold reaches APCER <= {max_apcer}")
    return operating_point(scores, attack_types, thresholds[allowed[-1]])


def auc(scores, attack_types) -> float:
    scores, attack_types = _validated(scores, attack_types)
    return float(roc_auc_score(attack_types != BONA_FIDE, scores))


def eer(scores, attack_types) -> float:
    """Equal error rate between all attacks pooled and bona fide captures."""
    scores, attack_types = _validated(scores, attack_types)
    false_positive, true_positive, _ = roc_curve(attack_types != BONA_FIDE, scores)
    false_negative = 1 - true_positive
    index = int(np.argmin(np.abs(false_negative - false_positive)))
    return float((false_negative[index] + false_positive[index]) / 2)


def threshold_at_bpcer(bona_fide_scores, max_bpcer: float) -> float:
    """The lowest threshold that flags at most max_bpcer of these bona fide scores as attacks.

    Bank face checks fix BPCER first (at most 0.5%, usually 0.1% to 0.2%) and read detection there,
    so the threshold comes from bona fide captures alone. Ties at the limit push it above them all."""
    if not 0 < max_bpcer < 1:
        raise MetricInputError(f"max_bpcer must be between 0 and 1, got {max_bpcer}")
    scores = np.sort(np.asarray(bona_fide_scores, dtype=float))[::-1]
    if not len(scores):
        raise MetricInputError("need bona fide scores to set a threshold")
    if not np.all(np.isfinite(scores)):
        raise MetricInputError("scores contain NaN or infinity")
    allowed = int(np.floor(max_bpcer * len(scores)))
    # Only the `allowed` highest scores may reach the threshold, so it sits just above the next one.
    return float(np.nextafter(scores[allowed], np.inf))
