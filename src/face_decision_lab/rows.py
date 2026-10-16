"""Turns manifest samples into FastDecisionModel rows and reads its answers back as attack scores."""

from pathlib import Path

from .errors import PredictionError
from .labels import QUESTIONS, STATE
from .manifest import FaceSample


def decision_row(sample: FaceSample, image_root: str | Path) -> dict:
    """One training row: the image, the fixed state, both questions and their gold answers."""
    return {
        "state": STATE,
        "questions": QUESTIONS,
        "gold": {
            "is_bona_fide": "true" if sample.is_bona_fide else "false",
            "attack_type": sample.attack_type,
        },
        "images": [{"path": str(Path(image_root) / sample.image_path)}],
    }


def attack_score(answers: dict) -> float:
    """Probability that the capture is an attack, from a FastDecisionModel.predict result."""
    try:
        bona_fide = float(answers["is_bona_fide"]["probabilities"]["true"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PredictionError(f"answer has no is_bona_fide probability for 'true': {answers!r}") from exc
    if not 0.0 <= bona_fide <= 1.0:
        raise PredictionError(f"is_bona_fide probability {bona_fide} is outside [0, 1]")
    return 1.0 - bona_fide
