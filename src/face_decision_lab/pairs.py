"""Batches that keep a real photo and the fakes made from it together.

GenD (arXiv 2508.06248) reports that training on paired real and fake data from the same source is
essential against shortcut learning: when a batch holds a real face and its swap, the cheapest way to
tell them apart is the manipulation itself, not the person, pose or background. DeepFakeFace is paired
by construction (one IMDB-WIKI photo, three fakes made from it), and its video_id encodes the pairing.
"""

import random

from .manifest import FaceSample


def pair_key(sample: FaceSample) -> str:
    """The source a sample was made from. DeepFakeFace ids are "<generator>/<path>", so dropping the
    generator leaves the shared photo path. FaceForensics++ manipulations keep the target video, so
    an original and every fake of it share "ffpp/<target>". Other sources keep their own id."""
    if sample.source_dataset == "faceforensics":
        return f"ffpp/{sample.subject_id.removeprefix('ffpp_')}"
    prefix = f"{sample.generator}/"
    return sample.video_id[len(prefix) :] if sample.video_id.startswith(prefix) else sample.video_id


def paired_batches(samples: list[FaceSample], batch_size: int, seed: int) -> list[list[FaceSample]]:
    """Shuffled batches made of whole pair groups. A group never spans two batches unless it alone
    is larger than batch_size; every sample appears exactly once."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    groups: dict[str, list[FaceSample]] = {}
    for sample in samples:
        groups.setdefault(pair_key(sample), []).append(sample)
    order = sorted(groups)
    random.Random(seed).shuffle(order)
    batches, current = [], []
    for key in order:
        group = groups[key]
        if current and len(current) + len(group) > batch_size:
            batches.append(current)
            current = []
        current.extend(group)
        while len(current) >= batch_size:
            batches.append(current[:batch_size])
            current = current[batch_size:]
    if current:
        batches.append(current)
    return batches
