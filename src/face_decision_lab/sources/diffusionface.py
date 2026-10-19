"""DiffusionFace (Chen et al., arXiv 2403.18471), used only as an unseen test set.

Part 1 is open on Zenodo (record 10865300, CC BY 4.0; the authors ask for research use only). It
has one tar per folder, each holding <folder>/<number>.png at 256x256 (Inpaint keeps its images in
Inpaint/images/, next to the masks and JSON files it skips). Real/ holds the CelebA-HQ photos;
every other folder holds images made by one diffusion method, and an image's number is that of
the real photo it was made from or paired with. MM_CelebA.tar holds no images, only the captions
behind the text prompts, so it is skipped whole. Unconditional generators (ADM, DDIM, DDPM,
PNDM, LDM) share the numbering without depicting the same person.

The authors publish no test split, and the lab never trains on DiffusionFace, so the build keeps a
fixed sample of numbers from every folder (`sampled`).
"""

import re
from hashlib import sha1

from .base import SourceItem, SourceLayoutError

SOURCE_DATASET = "diffusionface"

# Folder -> attack_type. Stable Diffusion image-to-image at denoising strength 0.3 to 0.7
# regenerates the whole photo, so it counts as synthetic; inpainting replaces the face region,
# mapped to face_swap as DeepFakeFace's inpainting is.
FOLDERS = {
    "Real": "bona_fide",
    "ADM": "fully_synthetic",
    "DDIM": "fully_synthetic",
    "DDPM": "fully_synthetic",
    "PNDM": "fully_synthetic",
    "LDM": "fully_synthetic",
    "SDv15_DS0.3": "fully_synthetic",
    "SDv15_DS0.5": "fully_synthetic",
    "SDv15_DS0.7": "fully_synthetic",
    "SDv21_DS0.3": "fully_synthetic",
    "SDv21_DS0.5": "fully_synthetic",
    "SDv21_DS0.7": "fully_synthetic",
    "DiffSwap": "face_swap",
    "Inpaint": "face_swap",
}

_MEMBER = re.compile(r"(?P<folder>[^/]+)/(?:images/)?(?P<number>\d+)\.png")
# Files next to the images that are not images of a face.
_SKIPPED = re.compile(r"MM_CelebA/.*|Inpaint/(?:masks/\d+\.png|jsons/\d+\.json)|DiffSwap/target_source\.csv")


def sampled(name: str, every: int) -> bool:
    """True for about one image number in every, the same numbers in every folder."""
    number = name.rsplit("/", 1)[-1].split(".", 1)[0]
    return int(sha1(number.encode()).hexdigest(), 16) % every == 0


def parse_member(name: str) -> SourceItem | None:
    """Maps a tar member to a source item. Directories return None."""
    if name.endswith("/") or "/" not in name or _SKIPPED.fullmatch(name):
        return None
    folder = name.split("/", 1)[0]
    if folder not in FOLDERS:
        raise SourceLayoutError(f"unexpected DiffusionFace member {name!r}: unknown folder")
    match = _MEMBER.fullmatch(name)
    if match is None:
        raise SourceLayoutError(f"unexpected DiffusionFace member {name!r}: expected <folder>/<number>.png")
    return SourceItem(
        path=name,
        attack_type=FOLDERS[folder],
        subject_id=f"{SOURCE_DATASET}_{match['number']}",
        video_id=f"{SOURCE_DATASET}/{folder}/{match['number']}",
        source_dataset=SOURCE_DATASET,
        generator=f"{SOURCE_DATASET}_{folder.lower()}",
    )
