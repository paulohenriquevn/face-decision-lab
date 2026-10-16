"""Baselines the decision model has to beat.

LinearProbe follows UniversalFakeDetect (Ojha et al., CVPR 2023): a linear classifier on frozen CLIP
image features. Feature extraction needs a GPU and lives in notebooks/clip_baseline.ipynb; the probe
itself is plain scikit-learn.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .errors import MetricInputError


class LinearProbe:
    """Logistic regression from image features to the probability of an attack."""

    def __init__(self, seed: int = 0):
        # Balanced weights: bona fide is the minority class in DeepFakeFace (one real per three fakes).
        self._model = make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed),
        )

    def fit(self, features: np.ndarray, is_attack: np.ndarray) -> "LinearProbe":
        features, is_attack = np.asarray(features), np.asarray(is_attack, dtype=bool)
        if features.ndim != 2 or len(features) != len(is_attack):
            raise MetricInputError(f"got {features.shape} features for {is_attack.shape} labels")
        if is_attack.all() or not is_attack.any():
            raise MetricInputError("training needs both bona fide and attack samples")
        self._model.fit(features, is_attack)
        return self

    def attack_scores(self, features: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(np.asarray(features))[:, 1]
