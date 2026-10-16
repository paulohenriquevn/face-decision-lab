"""DeepFakeFace (OpenRL/DeepFakeFace on Hugging Face, arXiv 2309.02218).

Four zips with the same 30,000 relative paths, taken from IMDB-WIKI:

- wiki.zip: the real photos
- insight.zip: InsightFace face swaps
- inpainting.zip: Stable Diffusion inpainting of the face region
- text2img.zip: Stable Diffusion 1.5 images generated from scratch

A path such as `wiki/61/19242061_1982-08-16_2009.jpg` starts with the IMDB-WIKI person id. It becomes
the subject, so a person's real photo and the three fakes made from it stay on one side of a split.
"""

import re

from .base import IMAGE_EXTENSIONS, SourceItem, SourceLayoutError

SOURCE_DATASET = "deepfakeface"
REPO_ID = "OpenRL/DeepFakeFace"

GENERATORS = {
    "wiki": "bona_fide",
    "insight": "face_swap",
    "inpainting": "face_swap",
    "text2img": "fully_synthetic",
}

_MEMBER = re.compile(r"^(?P<generator>[a-z0-9]+)/(?P<folder>\d{2})/(?P<person>\d+)_[^/]+$")


def parse_member(name: str) -> SourceItem | None:
    """Maps a zip member to a source item. Directories return None; unknown layouts raise."""
    if name.endswith("/"):
        return None
    match = _MEMBER.match(name)
    if not match or not name.lower().endswith(IMAGE_EXTENSIONS):
        raise SourceLayoutError(f"unexpected DeepFakeFace member {name!r}")
    generator = match["generator"]
    if generator not in GENERATORS:
        raise SourceLayoutError(f"unknown DeepFakeFace generator {generator!r} in {name!r}")
    relative = name.split("/", 1)[1]
    return SourceItem(
        path=name,
        attack_type=GENERATORS[generator],
        subject_id=f"imdbwiki_{match['person']}",
        # One image per "video": the stem is shared by the real photo and its fakes, so the generator
        # keeps the ids distinct while the subject keeps them together.
        video_id=f"{generator}/{relative}",
        source_dataset=SOURCE_DATASET,
        generator=generator,
    )
