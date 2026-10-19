import pytest

from face_decision_lab.sources import diffusionface
from face_decision_lab.sources.base import SourceLayoutError

# Member names as they appear in the DiffusionFace part 1 tars on Zenodo (October 2026).
MEMBERS = {
    "Real/23010.png": ("diffusionface_real", "bona_fide"),
    "ADM/23010.png": ("diffusionface_adm", "fully_synthetic"),
    "DDIM/23010.png": ("diffusionface_ddim", "fully_synthetic"),
    "DDPM/23010.png": ("diffusionface_ddpm", "fully_synthetic"),
    "PNDM/23010.png": ("diffusionface_pndm", "fully_synthetic"),
    "LDM/23010.png": ("diffusionface_ldm", "fully_synthetic"),
    "SDv15_DS0.3/23010.png": ("diffusionface_sdv15_ds0.3", "fully_synthetic"),
    "SDv21_DS0.7/23010.png": ("diffusionface_sdv21_ds0.7", "fully_synthetic"),
    "DiffSwap/23010.png": ("diffusionface_diffswap", "face_swap"),
    "Inpaint/images/23010.png": ("diffusionface_inpaint", "face_swap"),
}


@pytest.mark.parametrize("name", sorted(MEMBERS))
def test_every_folder_maps_to_its_generator_and_attack_type(name):
    item = diffusionface.parse_member(name)

    assert (item.generator, item.attack_type) == MEMBERS[name]


def test_a_fake_shares_the_subject_of_the_real_photo_with_its_number():
    real = diffusionface.parse_member("Real/23010.png")
    fake = diffusionface.parse_member("SDv15_DS0.5/23010.png")

    assert real.subject_id == fake.subject_id == "diffusionface_23010"
    assert real.video_id != fake.video_id


@pytest.mark.parametrize(
    "name",
    [
        "Real/",
        "Inpaint/masks/23010.png",
        "Inpaint/jsons/23010.json",
        "DiffSwap/target_source.csv",
        "MM_CelebA/text/celeba-caption/20820.txt",
        "MM_CelebA/train/filenames.pickle",
    ],
)
def test_directories_masks_and_metadata_are_skipped(name):
    # MM_CelebA.tar holds only the captions and file lists behind the text prompts, no images.
    assert diffusionface.parse_member(name) is None


def test_inpainting_shares_the_subject_of_its_real_photo():
    item = diffusionface.parse_member("Inpaint/images/23010.png")

    assert item.subject_id == "diffusionface_23010"


def test_unexpected_member_of_a_known_folder_is_rejected():
    with pytest.raises(SourceLayoutError, match="expected <folder>/<number>.png"):
        diffusionface.parse_member("Inpaint/previews/23010.png")


def test_unknown_folder_is_rejected():
    with pytest.raises(SourceLayoutError, match="unexpected DiffusionFace member 'StyleGAN/1.png'"):
        diffusionface.parse_member("StyleGAN/1.png")


def test_member_that_is_not_a_numbered_png_is_rejected():
    with pytest.raises(SourceLayoutError, match="expected <folder>/<number>.png"):
        diffusionface.parse_member("Real/notes.txt")


def test_sampled_keeps_the_same_numbers_in_every_folder():
    kept = {n for n in range(5000) if diffusionface.sampled(f"Real/{n}.png", every=20)}

    assert kept == {n for n in range(5000) if diffusionface.sampled(f"ADM/{n}.png", every=20)}
    assert 150 < len(kept) < 350
