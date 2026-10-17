import pytest

from face_decision_lab.sources import faceforensics
from face_decision_lab.sources.base import SourceLayoutError

# Paths as the FaceForensics++ download script writes them (c23 videos).
ORIGINAL = "original_sequences/youtube/c23/videos/071.mp4"
SWAP = "manipulated_sequences/Deepfakes/c23/videos/071_054.mp4"


def test_original_video_is_bona_fide():
    item = faceforensics.parse_path(ORIGINAL)

    assert (item.attack_type, item.generator) == ("bona_fide", "ffpp_original")


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("Deepfakes", ("face_swap", "ffpp_deepfakes")),
        ("FaceSwap", ("face_swap", "ffpp_faceswap")),
        ("FaceShifter", ("face_swap", "ffpp_faceshifter")),
        ("Face2Face", ("face_reenactment", "ffpp_face2face")),
        ("NeuralTextures", ("face_reenactment", "ffpp_neuraltextures")),
    ],
)
def test_each_manipulation_method_maps_to_its_label(method, expected):
    item = faceforensics.parse_path(f"manipulated_sequences/{method}/c23/videos/071_054.mp4")

    assert (item.attack_type, item.generator) == expected


def test_manipulated_video_belongs_to_the_person_in_the_target_video():
    original = faceforensics.parse_path(ORIGINAL)
    swap = faceforensics.parse_path(SWAP)

    assert original.subject_id == swap.subject_id == "ffpp_071"


def test_original_and_manipulated_videos_have_distinct_video_ids():
    assert faceforensics.parse_path(ORIGINAL).video_id != faceforensics.parse_path(SWAP).video_id


def test_parse_path_skips_files_that_are_not_videos():
    assert faceforensics.parse_path("manipulated_sequences/Deepfakes/c23/videos/") is None


def test_parse_path_rejects_an_unknown_method():
    with pytest.raises(SourceLayoutError, match="unexpected FaceForensics\\+\\+ path"):
        faceforensics.parse_path("manipulated_sequences/StyleGAN/c23/videos/071_054.mp4")


def test_parse_path_rejects_a_misnamed_manipulated_video():
    with pytest.raises(SourceLayoutError, match="unexpected FaceForensics\\+\\+ path"):
        faceforensics.parse_path("manipulated_sequences/Deepfakes/c23/videos/071.mp4")


def test_official_split_follows_the_target_sequence():
    # ["071", "054"] is the first pair of the official train split.
    assert faceforensics.official_split(faceforensics.parse_path(SWAP)) == "train"


def test_official_split_of_a_test_sequence():
    # ["953", "974"] is the first pair of the official test split.
    item = faceforensics.parse_path("manipulated_sequences/Face2Face/c23/videos/953_974.mp4")

    assert faceforensics.official_split(item) == "test"


def test_official_splits_cover_each_of_the_1000_sequences_once():
    sequences = [s for pairs in faceforensics.OFFICIAL_SPLITS.values() for pair in pairs for s in pair]

    assert len(sequences) == len(set(sequences)) == 1000


def test_official_split_rejects_a_sample_from_another_dataset(make_sample):
    with pytest.raises(ValueError, match="not a FaceForensics\\+\\+ sample"):
        faceforensics.official_split(make_sample(subject="celebdf_id0", source="celebdf"))
