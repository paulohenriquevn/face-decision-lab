"""AxonData face anti-spoofing sample (AxonData/face-anti-spoofing-dataset on Hugging Face, CC BY-NC 4.0).

A free sample of a commercial dataset: a few dozen videos per attack instrument plus bona fide
selfies. The card does not say which participant appears in which file, so a file is its own subject
unless its folder names the identity (`Mask_6`, `id1`, `id_06`): for a mask, the identity is the face
the mask imitates. Two files of one participant can still land on both sides of a split; with
samples this small, treat AxonData metrics as a format check, not a measurement.
"""

from pathlib import PurePosixPath

from .base import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, SourceItem, SourceLayoutError

SOURCE_DATASET = "axondata"
REPO_ID = "AxonData/face-anti-spoofing-dataset"

# Top-level folder -> (attack_type, generator, folder depth that names the identity or None).
FOLDERS = {
    "Selfies": ("bona_fide", "selfie", None),
    "Replay_display_attacks/Real": ("bona_fide", "selfie", None),
    "Replay_display_attacks/Screen": ("replay", "display_replay", None),
    "Replay_mobile_attacks": ("replay", "mobile_replay", None),
    "Cutout_attacks": ("mask", "cutout_mask", None),
    "3D_paper_mask": ("mask", "paper_mask_3d", None),
    "Wrapped_3D_paper_mask": ("mask", "wrapped_paper_mask_3d", 1),
    "Latex_mask": ("mask", "latex_mask", 1),
    "Silicone_mask": ("mask", "silicone_mask", 1),
    "Textile 3D Face Mask Attack Sample": ("mask", "textile_mask", 1),
}


def parse_path(path: str) -> SourceItem | None:
    """Maps a repository file to a source item. Non-media files return None; unknown folders raise."""
    if not path.lower().endswith(IMAGE_EXTENSIONS + VIDEO_EXTENSIONS):
        return None
    # The repository has a folder named "3D_paper_mask " with a trailing space.
    parts = [part.strip() for part in PurePosixPath(path).parts]
    for depth in (2, 1):
        key = "/".join(parts[:depth])
        if key in FOLDERS and len(parts) > depth:
            attack_type, generator, identity_depth = FOLDERS[key]
            break
    else:
        raise SourceLayoutError(f"unexpected AxonData path {path!r}")
    if identity_depth is not None and len(parts) > depth + identity_depth:
        identity = parts[depth + identity_depth - 1]
        subject = f"axon_{generator}_{identity}"
    else:
        subject = f"axon_{generator}_{PurePosixPath(path).stem}"
    return SourceItem(
        path=path,
        attack_type=attack_type,
        subject_id=subject.replace(" ", "_"),
        video_id="axon/" + "/".join(parts).replace(" ", "_"),
        source_dataset=SOURCE_DATASET,
        generator=generator,
    )
