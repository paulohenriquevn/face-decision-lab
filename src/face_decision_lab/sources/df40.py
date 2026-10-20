"""DF40 (Yan et al., arXiv 2406.13495): 40 forgery methods.

Access is granted per researcher through the authors' form; the terms allow non-commercial
research only and forbid redistribution, so the download links stay in a Colab secret and the
crops go to a private dataset. The frames come already cropped by the authors (DeepfakeBench
preprocessing); the build crops them again with the lab's pipeline so every source shares one
framing.

Thirty-one methods are "known": the authors made them from FaceForensics++ sequences and give
training frames for them. Face swapping and reenactment frames sit in
<method>/frames/<target>_<source>/<n>.png, entire-face synthesis in <method>/<sequence>/<name>.png,
and each is assigned the official FF++ split of its sequence (`official_split`). The FF++ real
frames the authors cropped the same way come in a separate archive. Nine methods are "unknown",
test only, and each ships its own real images, so each is scored against its own reals
(`method`). e4e is the exception: its inversions are of FF++ frames and its reals are the FF++
real frames of the test sequences. The Celeb-DF domain of DF40 is left out: the authors' link to
its real frames points to a script, and Celeb-DF is the lab's video benchmark.
"""

import re
from hashlib import sha1

from ..manifest import FaceSample
from .base import IMAGE_EXTENSIONS, SourceItem, SourceLayoutError
from .faceforensics import split_of_sequence

SOURCE_DATASET = "df40"

# Training archive -> attack_type. Face editing has no attack type of its own; like DiFF's FE
# subset it maps to face_reenactment, a real face digitally changed.
KNOWN_METHODS = {
    **dict.fromkeys(
        ("blendface", "e4s", "facedancer", "faceswap", "fsgan", "inswap", "mobileswap", "simswap", "uniface"),
        "face_swap",
    ),
    **dict.fromkeys(
        (
            "danet",
            "facevid2vid",
            "fomm",
            "hyperreenact",
            "lia",
            "mcnet",
            "MRAA",
            "one_shot_free",
            "pirender",
            "sadtalker",
            "tpsm",
            "wav2lip",
        ),
        "face_reenactment",
    ),
    **dict.fromkeys(
        ("ddim", "DiT", "pixart", "RDDM", "SiT", "sd2.1", "StyleGAN2", "StyleGAN3", "StyleGANXL", "VQGAN"),
        "fully_synthetic",
    ),
}
UNKNOWN_METHODS = {
    "deepfacelab": "face_swap",
    "heygen": "face_reenactment",
    "MidJourney": "fully_synthetic",
    "whichfaceisreal": "fully_synthetic",
    "CollabDiff": "face_reenactment",
    "e4e": "face_reenactment",
    "stargan": "face_reenactment",
    "starganv2": "face_reenactment",
    "styleclip": "face_reenactment",
}

# A few sadtalker clips use "##" instead of "_" after the sequence ("000##CfmNXPMjChs_50010").
_SWAP_OR_REENACT = re.compile(
    r"(?:[^/]+/)?frames/(?P<video>(?P<sequence>\d{3})(?:_|##)[^/]+)/(?P<frame>[^/]+)"
)
_SYNTHETIC = re.compile(r"[^/]+/(?P<sequence>\d{3})/(?P<frame>[^/]+)")
_REAL = re.compile(r"FaceForensics\+\+/original_sequences/youtube/c23/frames/(?P<sequence>\d{3})/[^/]+")
_E4E = re.compile(r"e4e/e4e/ff/(?P<sequence>\d{3})/[^/]+")
# Unknown archives: <folder>/{fake,real}/... with a few nested variants.
_UNKNOWN = re.compile(r"(?P<folder>[^/]+)/(?P<label>fake|real)/(?:(?:fake|real|frames)/)?(?P<rest>.+)")


def _skipped(name: str) -> bool:
    parts = name.split("/")
    return (
        name.endswith("/")
        or parts[0] == "__MACOSX"
        or parts[-1].startswith(".")
        or "landmarks" in parts
        # MidJourney ships 247 of its fakes as GIF, which OpenCV cannot decode.
        or not name.lower().endswith(IMAGE_EXTENSIONS)
    )


def _generator(method: str) -> str:
    return f"{SOURCE_DATASET}_{method.lower()}"


def sampled(name: str, every: int) -> bool:
    """True for about one member in every; fixed by the member's name."""
    return int(sha1(name.encode()).hexdigest(), 16) % every == 0


def parse_train_member(archive: str, name: str) -> SourceItem | None:
    """Maps a member of a known method's training archive (archive is its stem, e.g. "sd2.1")."""
    if archive not in KNOWN_METHODS:
        raise SourceLayoutError(f"unexpected DF40 training archive {archive!r}")
    if _skipped(name):
        return None
    attack_type = KNOWN_METHODS[archive]
    pattern = _SYNTHETIC if attack_type == "fully_synthetic" else _SWAP_OR_REENACT
    match = pattern.fullmatch(name)
    if match is None:
        raise SourceLayoutError(f"unexpected DF40 member {name!r} in archive {archive!r}")
    video = match["video"] if "video" in match.groupdict() else match["sequence"]
    return SourceItem(
        path=f"{archive}/{name}",
        attack_type=attack_type,
        subject_id=f"ffpp_{match['sequence']}",
        video_id=f"{SOURCE_DATASET}/{archive}/{video}",
        source_dataset=SOURCE_DATASET,
        generator=_generator(archive),
    )


def parse_real_member(name: str) -> SourceItem | None:
    """Maps a member of the authors' archive of FF++ real frames."""
    if _skipped(name):
        return None
    match = _REAL.fullmatch(name)
    if match is None:
        raise SourceLayoutError(f"unexpected DF40 real member {name!r}")
    return SourceItem(
        path=name,
        attack_type="bona_fide",
        subject_id=f"ffpp_{match['sequence']}",
        video_id=f"{SOURCE_DATASET}/ffpp_real/{match['sequence']}",
        source_dataset=SOURCE_DATASET,
        generator=_generator("ffpp_real"),
    )


def parse_unknown_member(archive: str, name: str) -> SourceItem | None:
    """Maps a member of an unknown method's archive: its fakes and the reals it ships."""
    if archive not in UNKNOWN_METHODS:
        raise SourceLayoutError(f"unexpected DF40 unknown archive {archive!r}")
    if _skipped(name):
        return None
    if archive == "e4e":
        if name.startswith("e4e/e4e/cdf/"):
            return None
        match = _E4E.fullmatch(name)
        if match is None:
            raise SourceLayoutError(f"unexpected DF40 member {name!r} in archive 'e4e'")
        return SourceItem(
            path=f"{archive}/{name}",
            attack_type=UNKNOWN_METHODS[archive],
            subject_id=f"ffpp_{match['sequence']}",
            video_id=f"{SOURCE_DATASET}/e4e/{match['sequence']}",
            source_dataset=SOURCE_DATASET,
            generator=_generator(archive),
        )
    match = _UNKNOWN.fullmatch(name)
    if match is None:
        raise SourceLayoutError(f"unexpected DF40 member {name!r} in archive {archive!r}")
    real = match["label"] == "real"
    # Frames of one clip share their folder; single images are their own video.
    video = match["rest"].rsplit("/", 1)[0] if "/" in match["rest"] else match["rest"].rsplit(".", 1)[0]
    method = f"{archive}_real" if real else archive
    return SourceItem(
        path=f"{archive}/{name}",
        attack_type="bona_fide" if real else UNKNOWN_METHODS[archive],
        subject_id=f"{SOURCE_DATASET}_{method.lower()}_{video}",
        video_id=f"{SOURCE_DATASET}/{method}/{video}",
        source_dataset=SOURCE_DATASET,
        generator=_generator(method),
    )


def method(sample: FaceSample) -> str:
    """The DF40 method a sample belongs to, its reals included ("midjourney" for both
    df40_midjourney and df40_midjourney_real)."""
    name = sample.generator.removeprefix(f"{SOURCE_DATASET}_")
    return name if name == "ffpp_real" else name.removesuffix("_real")


def unknown_group(sample: FaceSample) -> str:
    """The unknown method a test sample is scored in: its own method, or e4e for the FF++ real
    frames, which are e4e's reals."""
    name = method(sample)
    return "e4e" if name == "ffpp_real" else name


def official_split(sample: FaceSample | SourceItem) -> str:
    """The official FF++ split of the sequence a sample was made from."""
    if not sample.subject_id.startswith("ffpp_"):
        raise ValueError(f"{sample.subject_id!r} has no FaceForensics++ sequence")
    return split_of_sequence(sample.subject_id.removeprefix("ffpp_"))
