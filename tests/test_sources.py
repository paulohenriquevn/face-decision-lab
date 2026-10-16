import pytest

from face_decision_lab.labels import ATTACK_TYPES
from face_decision_lab.sources import axondata, deepfakeface
from face_decision_lab.sources.base import SourceLayoutError

# Real paths from the AxonData/face-anti-spoofing-dataset listing (October 2026).
AXON_PATHS = {
    "Selfies/Selfie_1.jpg": ("bona_fide", "selfie"),
    "Replay_display_attacks/Real/1.jpg": ("bona_fide", "selfie"),
    "Replay_display_attacks/Screen/IMG_5915.MOV": ("replay", "display_replay"),
    "Replay_mobile_attacks/Galaxy_a54-Pixel7/20241126_030439.mp4": ("replay", "mobile_replay"),
    "Cutout_attacks/104.MOV": ("mask", "cutout_mask"),
    "3D_paper_mask /133.mp4": ("mask", "paper_mask_3d"),
    "Wrapped_3D_paper_mask/id_06/PXL_20230411_164048136.mp4": ("mask", "wrapped_paper_mask_3d"),
    "Latex_mask/Mask_6/1.mp4": ("mask", "latex_mask"),
    "Silicone_mask/Mask 3/108.mp4": ("mask", "silicone_mask"),
    "Textile 3D Face Mask Attack Sample/id1/Samsung_S23/20250528_181931.mp4": ("mask", "textile_mask"),
}


@pytest.mark.parametrize("generator", ["wiki", "insight", "inpainting", "text2img"])
def test_deepfakeface_maps_each_generator_to_a_known_label(generator):
    item = deepfakeface.parse_member(f"{generator}/61/19242061_1982-08-16_2009.jpg")

    assert item.attack_type in ATTACK_TYPES


def test_deepfakeface_real_photo_is_bona_fide():
    assert deepfakeface.parse_member("wiki/61/19242061_1982-08-16_2009.jpg").attack_type == "bona_fide"


def test_deepfakeface_real_photo_and_its_fakes_share_a_subject():
    names = [f"{g}/61/19242061_1982-08-16_2009.jpg" for g in deepfakeface.GENERATORS]

    subjects = {deepfakeface.parse_member(name).subject_id for name in names}

    assert subjects == {"imdbwiki_19242061"}


def test_deepfakeface_real_photo_and_its_fakes_have_distinct_video_ids():
    names = [f"{g}/61/19242061_1982-08-16_2009.jpg" for g in deepfakeface.GENERATORS]

    assert len({deepfakeface.parse_member(name).video_id for name in names}) == 4


def test_deepfakeface_skips_directories():
    assert deepfakeface.parse_member("wiki/61/") is None


def test_deepfakeface_rejects_an_unknown_generator():
    with pytest.raises(SourceLayoutError, match="unknown DeepFakeFace generator 'stylegan'"):
        deepfakeface.parse_member("stylegan/61/19242061_1982-08-16_2009.jpg")


def test_deepfakeface_rejects_an_unexpected_layout():
    with pytest.raises(SourceLayoutError, match="unexpected DeepFakeFace member"):
        deepfakeface.parse_member("wiki/19242061.jpg")


@pytest.mark.parametrize(("path", "expected"), AXON_PATHS.items())
def test_axondata_maps_each_folder_to_its_label_and_instrument(path, expected):
    item = axondata.parse_path(path)

    assert (item.attack_type, item.generator) == expected


def test_axondata_groups_videos_of_one_mask_identity_into_one_subject():
    first = axondata.parse_path("Latex_mask/Mask_6/1.mp4")
    second = axondata.parse_path("Latex_mask/Mask_6/2.mp4")

    assert first.subject_id == second.subject_id


def test_axondata_keeps_different_mask_identities_apart():
    first = axondata.parse_path("Latex_mask/Mask_6/1.mp4")
    second = axondata.parse_path("Latex_mask/Mask_8/1.MOV")

    assert first.subject_id != second.subject_id


def test_axondata_ids_contain_no_spaces():
    item = axondata.parse_path("Silicone_mask/Mask 3/108.mp4")

    assert " " not in item.subject_id + item.video_id


def test_axondata_skips_non_media_files():
    assert axondata.parse_path("README.md") is None


def test_axondata_rejects_an_unknown_folder():
    with pytest.raises(SourceLayoutError, match="unexpected AxonData path"):
        axondata.parse_path("Resin_mask/1.mp4")


def test_axondata_marks_videos_as_videos():
    assert axondata.parse_path("Cutout_attacks/104.MOV").is_video
