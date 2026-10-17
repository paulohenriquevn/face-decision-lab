"""FaceForensics++ (Rössler et al., ICCV 2019), c23 videos.

Access is granted per researcher through the authors' form; the terms allow research use only and
forbid redistribution, so the download URL and the videos stay out of this repository and the crops
built from them go to a private dataset.

Layout written by the authors' download script: original_sequences/youtube/c23/videos/<target>.mp4
and manipulated_sequences/<method>/c23/videos/<target>_<source>.mp4, where the face or expression
of <source> is put into the video <target>. The official train/val/test splits
(faceforensics_splits.json, MIT) list pairs of sequences, so a target and its source share a split.
"""

import json
import re
from importlib import resources

from ..manifest import FaceSample
from .base import SourceItem, SourceLayoutError

SOURCE_DATASET = "faceforensics"
COMPRESSION = "c23"

# Manipulation method folder -> (attack_type, generator).
METHODS = {
    "Deepfakes": ("face_swap", "ffpp_deepfakes"),
    "FaceSwap": ("face_swap", "ffpp_faceswap"),
    "FaceShifter": ("face_swap", "ffpp_faceshifter"),
    "Face2Face": ("face_reenactment", "ffpp_face2face"),
    "NeuralTextures": ("face_reenactment", "ffpp_neuraltextures"),
}

_ORIGINAL = re.compile(rf"original_sequences/youtube/{COMPRESSION}/videos/(\d{{3}})\.mp4")
_MANIPULATED = re.compile(rf"manipulated_sequences/(\w+)/{COMPRESSION}/videos/(\d{{3}})_(\d{{3}})\.mp4")

OFFICIAL_SPLITS: dict[str, list[list[str]]] = json.loads(
    resources.files(__package__).joinpath("faceforensics_splits.json").read_text()
)
_SPLIT_OF_SEQUENCE = {
    sequence: split for split, pairs in OFFICIAL_SPLITS.items() for pair in pairs for sequence in pair
}


def parse_path(path: str) -> SourceItem | None:
    """Maps a downloaded file to a source item. Non-video files return None; unknown layouts raise."""
    if not path.endswith(".mp4"):
        return None
    if match := _ORIGINAL.fullmatch(path):
        target = match.group(1)
        attack_type, generator, method = "bona_fide", "ffpp_original", "original"
        name = target
    elif (match := _MANIPULATED.fullmatch(path)) and match.group(1) in METHODS:
        method, target, source = match.groups()
        attack_type, generator = METHODS[method]
        name = f"{target}_{source}"
    else:
        raise SourceLayoutError(f"unexpected FaceForensics++ path {path!r}")
    return SourceItem(
        path=path,
        attack_type=attack_type,
        subject_id=f"ffpp_{target}",
        video_id=f"ffpp/{method}/{name}",
        source_dataset=SOURCE_DATASET,
        generator=generator,
    )


def official_split(sample: FaceSample | SourceItem) -> str:
    """The official split ("train", "val" or "test") of the target sequence."""
    if sample.source_dataset != SOURCE_DATASET or not sample.subject_id.startswith("ffpp_"):
        raise ValueError(f"not a FaceForensics++ sample: {sample.subject_id!r} ({sample.source_dataset})")
    return _SPLIT_OF_SEQUENCE[sample.subject_id.removeprefix("ffpp_")]
