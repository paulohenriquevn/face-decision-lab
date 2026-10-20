"""DiFF, Diffusion Facial Forgery (Cheng et al., ACM MM 2024), used only as an unseen test set.

The synthesized images are public on the authors' Google Drive (Apache 2.0 per the repository);
the pristine images and prompts are granted per researcher through the authors' form, for academic
research only and without redistribution. The crops built from them stay in a private repository.

The test split has one zip per method, grouped in the paper's four subsets: text-to-image (T2I),
image-to-image (I2I), face swapping (FS) and face editing (FE). Every zip holds
<method>/id_<number>/.../<image>, where id_<number> is the DiFF identity whose prompts produced
the image; how deep the image sits below it differs by method. The LoRA zip also carries macOS
resource forks (__MACOSX/, ._*), which are skipped.
"""

import re

from .base import IMAGE_EXTENSIONS, SourceItem, SourceLayoutError

SOURCE_DATASET = "diff"

# Method folder -> (subset, attack_type). DiFF's face editing changes attributes or expression of a
# real face, which the lab's taxonomy calls face_reenactment; DCFace sits in the FS subset as in
# the paper, although it generates new identities rather than swapping a given one.
METHODS = {
    "Midjourney": ("t2i", "fully_synthetic"),
    "SDXL": ("t2i", "fully_synthetic"),
    "FreeDoM_T": ("t2i", "fully_synthetic"),
    "HPS": ("t2i", "fully_synthetic"),
    "SDXL_Refine": ("i2i", "fully_synthetic"),
    "LoRA": ("i2i", "fully_synthetic"),
    "DreamBooth": ("i2i", "fully_synthetic"),
    "FreeDoM_I": ("i2i", "fully_synthetic"),
    "DiffFace": ("fs", "face_swap"),
    "DCFace": ("fs", "face_swap"),
    "Imagic": ("fe", "face_reenactment"),
    "CoDiff": ("fe", "face_reenactment"),
    "cycle_diff": ("fe", "face_reenactment"),
}

SUBSETS = ("t2i", "i2i", "fs", "fe")
_MEMBER = re.compile(r"(?P<method>[^/]+)/(?P<identity>id_\d+)/.+")


def generator_name(method: str) -> str:
    subset, _ = METHODS[method]
    return f"diff_{subset}_{method.lower()}"


def subset(generator: str) -> str:
    """The paper's subset (t2i, i2i, fs or fe) of a DiFF generator name."""
    parts = generator.split("_")
    if len(parts) < 3 or parts[0] != SOURCE_DATASET or parts[1] not in SUBSETS:
        raise ValueError(f"{generator!r} is not a DiFF generator name")
    return parts[1]


def parse_member(name: str) -> SourceItem | None:
    """Maps a zip member to a source item. Directories and macOS metadata return None."""
    file_name = name.rsplit("/", 1)[-1]
    if name.endswith("/") or name.startswith("__MACOSX/") or file_name.startswith("._"):
        return None
    method = name.split("/", 1)[0]
    if method not in METHODS:
        raise SourceLayoutError(f"unexpected DiFF member {name!r}: unknown method folder")
    match = _MEMBER.fullmatch(name)
    if match is None:
        raise SourceLayoutError(f"unexpected DiFF member {name!r}: expected <method>/id_<number>/...")
    if not name.lower().endswith(IMAGE_EXTENSIONS):
        raise SourceLayoutError(f"unexpected DiFF member {name!r}: not an image")
    _, attack_type = METHODS[method]
    return SourceItem(
        path=name,
        attack_type=attack_type,
        subject_id=f"{SOURCE_DATASET}_{match['identity']}",
        video_id=f"{SOURCE_DATASET}/{name.rsplit('.', 1)[0]}",
        source_dataset=SOURCE_DATASET,
        generator=generator_name(method),
    )


_PRISTINE_TEST = re.compile(r"DiFF_real/test/(?P<number>\d+)\.png")


def parse_pristine_member(name: str) -> SourceItem | None:
    """Maps a member of the pristine archive the authors grant per researcher: DiFF_real/test/<n>.png
    holds the pristine test images; the train and val folders are skipped, as the lab never trains on
    DiFF. The archive names no identity, so every image is its own subject."""
    if name.endswith("/") or name.startswith(("DiFF_real/train/", "DiFF_real/val/")):
        return None
    match = _PRISTINE_TEST.fullmatch(name)
    if match is None:
        raise SourceLayoutError(f"unexpected DiFF pristine member {name!r}")
    return SourceItem(
        path=name,
        attack_type="bona_fide",
        subject_id=f"{SOURCE_DATASET}_real_{match['number']}",
        video_id=f"{SOURCE_DATASET}/real/{match['number']}",
        source_dataset=SOURCE_DATASET,
        generator=f"{SOURCE_DATASET}_real",
    )
