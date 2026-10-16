"""Turns manifest samples into FastDecisionModel rows and reads its answers back as attack scores."""

from pathlib import Path

from .errors import PredictionError
from .labels import QUESTIONS, STATE
from .manifest import FaceSample


def local_image(sample: FaceSample, image_root: str | Path) -> dict:
    """Image reference for a crop stored on disk under image_root."""
    return {"path": str(Path(image_root) / sample.image_path)}


def decision_row(sample: FaceSample, image) -> dict:
    """One training row: the image, the fixed state, both questions and their gold answers.

    image is anything FastDecisionModel accepts: a PIL image, a data URL, or a {"bytes", "path"} dict.
    """
    return {
        "state": STATE,
        "questions": QUESTIONS,
        "gold": {
            "is_bona_fide": "true" if sample.is_bona_fide else "false",
            "attack_type": sample.attack_type,
        },
        "images": [image],
    }


def open_image(image):
    """Opens an image reference ({"path"} or {"bytes"} dict) as an RGB PIL image."""
    import io

    from PIL import Image

    if image.get("bytes") is not None:
        source = io.BytesIO(image["bytes"])
    elif image.get("path"):
        source = image["path"]
    else:
        raise ValueError(f"image reference has neither bytes nor path: {image!r}")
    with Image.open(source) as opened:
        return opened.convert("RGB")


def attack_score(answers: dict) -> float:
    """Probability that the capture is an attack, from a FastDecisionModel.predict result."""
    try:
        bona_fide = float(answers["is_bona_fide"]["probabilities"]["true"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PredictionError(f"answer has no is_bona_fide probability for 'true': {answers!r}") from exc
    if not 0.0 <= bona_fide <= 1.0:
        raise PredictionError(f"is_bona_fide probability {bona_fide} is outside [0, 1]")
    return 1.0 - bona_fide
