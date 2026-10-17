"""Celeb-DF v2 (Li et al., CVPR 2020), used only as an unseen test set.

Access is granted per researcher through the authors' form; the terms allow non-commercial research
only and forbid redistribution, so the archive never leaves the researcher's own storage and the
crops built from it stay in a private repository.

Layout of Celeb-DF-v2.zip: Celeb-real/id<person>_<clip>.mp4 (YouTube interviews of celebrities),
YouTube-real/<clip>.mp4 (other YouTube videos), Celeb-synthesis/id<target>_id<source>_<clip>.mp4
(the face of <source> swapped into the clip id<target>_<clip>), and List_of_testing_videos.txt,
the official 518-video test list with one "<label> <path>" line per video, label 1 = real.
"""

import re

from .base import SourceItem, SourceLayoutError

SOURCE_DATASET = "celebdf"
TEST_LIST = "List_of_testing_videos.txt"

# Folder -> (attack_type, generator, file name pattern whose first group names the subject).
FOLDERS = {
    "Celeb-real": ("bona_fide", "celeb_real", re.compile(r"(id\d+)_\d{4}\.mp4")),
    "YouTube-real": ("bona_fide", "youtube_real", re.compile(r"(\d{5})\.mp4")),
    # The target is the person whose clip, body and background the fake keeps.
    "Celeb-synthesis": ("face_swap", "celebdf_synthesis", re.compile(r"(id\d+)_id\d+_\d{4}\.mp4")),
}


def parse_member(name: str) -> SourceItem | None:
    """Maps a zip member to a source item. The test list and directories return None."""
    if name.endswith("/") or name == TEST_LIST:
        return None
    folder, _, file_name = name.partition("/")
    if folder not in FOLDERS:
        raise SourceLayoutError(f"unexpected Celeb-DF member {name!r}")
    attack_type, generator, pattern = FOLDERS[folder]
    match = pattern.fullmatch(file_name)
    if match is None:
        raise SourceLayoutError(f"unexpected Celeb-DF member {name!r}: expected {pattern.pattern}")
    subject = match.group(1)
    if folder == "YouTube-real":
        subject = f"youtube_{subject}"
    return SourceItem(
        path=name,
        attack_type=attack_type,
        subject_id=f"celebdf_{subject}",
        video_id=f"celebdf/{name.removesuffix('.mp4')}",
        source_dataset=SOURCE_DATASET,
        generator=generator,
    )


def official_test_videos(test_list: str) -> set[str]:
    """Member paths of the official test videos, checking each label against its folder."""
    videos = set()
    for line in test_list.splitlines():
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 2 or fields[0] not in ("0", "1"):
            raise SourceLayoutError(f"malformed Celeb-DF test list line {line!r}")
        label, path = fields
        item = parse_member(path)
        if item is None or (label == "1") != (item.attack_type == "bona_fide"):
            raise SourceLayoutError(f"Celeb-DF test list labels {path!r} as {label}; its folder disagrees")
        videos.add(path)
    return videos
