"""Learned fusion of detector scores: the first policy layer over the detectors.

Each detector here is strong on one domain and weak on the other (the lab's LayerNorm-tuned CLIP on
DeepFakeFace, the official GenD checkpoint on video face swaps). A logistic regression over their
logits, fitted on calibration crops from both domains with every domain and class weighing the
same, learns how far to trust each one; taking the larger score instead lets either detector's false
alarms through.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

from .errors import MetricInputError
from .manifest import FaceSample
from .protocol import EXTERNAL_DATASETS

LOGIT_EPSILON = 1e-6


def score_logits(scores: np.ndarray) -> np.ndarray:
    """Logits of attack probabilities, clipped so scores of exactly 0 or 1 stay finite."""
    scores = np.clip(np.asarray(scores, dtype=float), LOGIT_EPSILON, 1 - LOGIT_EPSILON)
    return np.log(scores / (1 - scores))


def _domain(sample: FaceSample) -> str:
    # The lab sources (DeepFakeFace, AxonData) form one domain; each external dataset is its own.
    return sample.source_dataset if sample.source_dataset in EXTERNAL_DATASETS else "lab"


def in_lab_domain(samples: list[FaceSample]) -> np.ndarray:
    """True for crops from the lab sources: the labels a domain gate learns to predict."""
    return np.array([_domain(s) == "lab" for s in samples])


def gated_mixture(gate: np.ndarray, lab_expert: np.ndarray, video_expert: np.ndarray) -> np.ndarray:
    """gate * lab_expert + (1 - gate) * video_expert, where gate is the probability that a crop
    belongs to the lab domain (photos and generated images) rather than to video.

    A mixture of experts: the gate picks, crop by crop, which detector to listen to, where the
    learned fusion applies one weight to every crop.
    """
    gate, lab_expert, video_expert = (np.asarray(a, dtype=float) for a in (gate, lab_expert, video_expert))
    if not len(gate) == len(lab_expert) == len(video_expert):
        raise MetricInputError(
            f"got {len(gate)} gate values, {len(lab_expert)} and {len(video_expert)} scores"
        )
    if ((gate < 0) | (gate > 1)).any():
        raise MetricInputError("gate values must lie in [0, 1]")
    return gate * lab_expert + (1 - gate) * video_expert


def balancing_weights(samples: list[FaceSample]) -> np.ndarray:
    """Sample weights under which every (domain, bona fide or attack) cell has the same total."""
    keys = [(_domain(s), s.is_bona_fide) for s in samples]
    counts: dict[tuple[str, bool], int] = {}
    for key in keys:
        counts[key] = counts.get(key, 0) + 1
    return np.array([len(samples) / (len(counts) * counts[key]) for key in keys])


class ScoreFusion:
    """Logistic regression from detector logits to the probability of an attack."""

    def __init__(self, seed: int = 0):
        self._model = LogisticRegression(max_iter=2000, random_state=seed)

    def fit(
        self, scores: np.ndarray, is_attack: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> "ScoreFusion":
        """scores: (samples, detectors) attack probabilities."""
        scores, is_attack = np.asarray(scores, dtype=float), np.asarray(is_attack, dtype=bool)
        if scores.ndim != 2 or len(scores) != len(is_attack):
            raise MetricInputError(f"got {scores.shape} scores for {is_attack.shape} labels")
        if is_attack.all() or not is_attack.any():
            raise MetricInputError("fitting needs both bona fide and attack samples")
        self._model.fit(score_logits(scores), is_attack, sample_weight=sample_weight)
        return self

    def attack_scores(self, scores: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(score_logits(scores))[:, 1]

    @property
    def weights(self) -> list[float]:
        """The weight of each detector's logit, in input order."""
        return self._model.coef_[0].tolist()
