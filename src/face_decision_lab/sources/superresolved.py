"""Real crops passed through a super-resolution network, kept as bona fide.

The lab's photo detectors flag the CelebA-HQ photos of DiffusionFace as fake while lowering the
resolution of other real photos does not move their scores (docs/results.md, 2026-10-19). CelebA-HQ
was made with a super-resolution network, so the detail such a network draws is the likely cue.
These copies show the detector that cue on photos labeled real: each training-side real crop is
shrunk FACTOR times and enlarged back by one of NETWORKS.

A copy keeps the person and the pair group of its original, so the subject split and paired
batches treat it as another capture of the same source photo.
"""

from hashlib import sha1

from ..manifest import BONA_FIDE, FaceSample
from ..pairs import pair_key

SOURCE_DATASET = "superresolved"
FACTOR = 4
# Real-ESRGAN x4plus (BSD-3-Clause), a GAN-trained network like the one behind CelebA-HQ, and
# Swin2SR real-world x4 (Apache-2.0), a transformer, so the detector cannot learn one network.
NETWORKS = ("realesrgan_x4plus", "swin2sr_realworld_x4")


def network_for(original: FaceSample) -> str:
    """The network a crop is super-resolved with, fixed by its path."""
    return NETWORKS[int(sha1(original.image_path.encode()).hexdigest(), 16) % len(NETWORKS)]


def super_resolved(original: FaceSample, network: str, image_path: str) -> FaceSample:
    """The sample of original's crop after network, stored at image_path."""
    if not original.is_bona_fide:
        raise ValueError(f"only bona fide crops are super-resolved, got {original.attack_type}")
    if network not in NETWORKS:
        raise ValueError(f"unknown super-resolution network {network!r}; choose one of {NETWORKS}")
    generator = f"sr_{network}"
    return FaceSample(
        image_path=image_path,
        attack_type=BONA_FIDE,
        subject_id=original.subject_id,
        # pair_key drops the generator prefix, which leaves the original's pair group.
        video_id=f"{generator}/{pair_key(original)}",
        source_dataset=SOURCE_DATASET,
        generator=generator,
    )
