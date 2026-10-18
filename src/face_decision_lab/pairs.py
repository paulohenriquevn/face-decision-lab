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


def paired_batches(
    samples: list[FaceSample], batch_size: int, seed: int, max_group: int | None = None
) -> list[list[FaceSample]]:
    """Shuffled batches made of pair groups; every sample appears exactly once.

    A group never spans two batches unless it alone is larger than batch_size. With max_group, a
    larger group is shuffled and cut into pieces of at most max_group first: a FaceForensics++ group
    holds every frame of an original and its four manipulations, and whole, one fills a batch, so
    each step would see a single scene. The pieces still mix real and fake frames of one video.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if max_group is not None and max_group <= 0:
        raise ValueError(f"max_group must be positive, got {max_group}")
    rng = random.Random(seed)
    groups: dict[str, list[FaceSample]] = {}
    for sample in samples:
        groups.setdefault(pair_key(sample), []).append(sample)
    pieces = []
    for key in sorted(groups):
        group = groups[key]
        if max_group is None or len(group) <= max_group:
            pieces.append(group)
            continue
        group = list(group)
        random.Random(f"{seed}/{key}").shuffle(group)
        pieces += [group[i : i + max_group] for i in range(0, len(group), max_group)]
    rng.shuffle(pieces)
    batches, current = [], []
    for group in pieces:
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
