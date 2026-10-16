"""A tiny generated dataset that exercises the training pipeline end to end.

The images are colored noise and the labels are arbitrary, so a model trained on them learns
nothing. The point is to check that loading, row building, training, calibration and prediction run
on the chosen GPU before any real face data exists. It holds no face and no personal data.
"""

import random
from pathlib import Path

from .labels import ATTACK_TYPES, BONA_FIDE
from .manifest import FaceSample, write_manifest

SMOKE_TRAIN_SOURCE = "smoke_train"
SMOKE_TEST_SOURCE = "smoke_test"


def write_smoke_dataset(
    output_dir: str | Path,
    subjects_per_source: int = 14,
    frames_per_subject: int = 2,
    image_size: int = 224,
    seed: int = 0,
) -> list[FaceSample]:
    """Writes PNG images and manifest.csv under output_dir and returns the samples."""
    from PIL import Image

    if subjects_per_source < 2:
        raise ValueError(f"need at least 2 subjects per source, got {subjects_per_source}")
    output_dir = Path(output_dir)
    rng = random.Random(seed)
    attacks = sorted(set(ATTACK_TYPES) - {BONA_FIDE})
    samples = []
    for source in (SMOKE_TRAIN_SOURCE, SMOKE_TEST_SOURCE):
        for subject_index in range(subjects_per_source):
            subject = f"{source}_s{subject_index}"
            # Half the subjects are bona fide, the rest cycle through every attack type.
            attack_type = (
                BONA_FIDE if subject_index % 2 == 0 else attacks[(subject_index // 2) % len(attacks)]
            )
            for frame in range(frames_per_subject):
                relative = Path(source) / f"{subject}_{frame}.png"
                path = output_dir / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.frombytes(
                    "RGB", (image_size, image_size), rng.randbytes(image_size * image_size * 3)
                ).save(path)
                samples.append(
                    FaceSample(
                        image_path=relative.as_posix(),
                        attack_type=attack_type,
                        subject_id=subject,
                        video_id=f"{subject}_video",
                        source_dataset=source,
                        generator="noise",
                    )
                )
    write_manifest(samples, output_dir / "manifest.csv")
    return samples
