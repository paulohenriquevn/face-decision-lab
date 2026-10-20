"""Per-crop detector scores saved next to their samples, so intervals and new comparisons are
computed later without scoring again."""

from pathlib import Path

import numpy as np

from .manifest import COLUMNS, FaceSample


def save_scores(path: str | Path, samples: list[FaceSample], scores: dict[str, np.ndarray]) -> None:
    """Writes the samples' manifest columns and one score array per detector to a .npz file."""
    for detector, values in scores.items():
        if len(values) != len(samples):
            raise ValueError(f"detector {detector!r} has {len(values)} scores for {len(samples)} samples")
    columns = {f"sample_{column}": np.array([getattr(s, column) for s in samples]) for column in COLUMNS}
    detectors = {f"score_{name}": np.asarray(values, dtype=np.float64) for name, values in scores.items()}
    np.savez_compressed(path, **columns, **detectors)


def load_scores(path: str | Path) -> tuple[list[FaceSample], dict[str, np.ndarray]]:
    """The samples and per-detector scores written by save_scores."""
    with np.load(path) as data:
        rows = zip(*(data[f"sample_{column}"].tolist() for column in COLUMNS), strict=True)
        samples = [FaceSample(**dict(zip(COLUMNS, row, strict=True))) for row in rows]
        scores = {key.removeprefix("score_"): data[key] for key in data.files if key.startswith("score_")}
    return samples, scores
