"""FFHQ (Karras et al., CVPR 2019), real photos only, added to training as bona fide.

70,000 Flickr portraits, aligned and cropped around the face, each its own person as far as the
dataset says (it publishes no identities). The build reads the 256 px copy on the Hub
(`bitmind/ffhq-256`, parquet), whose image names keep the original numbering `00000.png` to
`69999.png`. NVIDIA releases the dataset under CC BY-NC-SA 4.0; the images keep their Flickr
licenses. The lab keeps a fixed sample (`sampled`) so the real side grows without swamping the fakes.
"""

import re
from hashlib import sha1

from .base import SourceItem, SourceLayoutError

SOURCE_DATASET = "ffhq"
_NAME = re.compile(r"(?P<number>\d{5})\.png")


def sampled(name: str, keep: int, out_of: int) -> bool:
    """True for about keep in every out_of image names, the same names on every run."""
    return int(sha1(name.encode()).hexdigest(), 16) % out_of < keep


def parse_name(name: str) -> SourceItem:
    """The source item of one FFHQ image, by its file name."""
    match = _NAME.fullmatch(name)
    if match is None:
        raise SourceLayoutError(f"unexpected FFHQ image name {name!r}: expected <5 digits>.png")
    number = match["number"]
    return SourceItem(
        path=name,
        attack_type="bona_fide",
        subject_id=f"{SOURCE_DATASET}_{number}",
        video_id=f"{SOURCE_DATASET}/{number}",
        source_dataset=SOURCE_DATASET,
        generator=f"{SOURCE_DATASET}_real",
    )
