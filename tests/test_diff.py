import pytest

from face_decision_lab.sources import diff
from face_decision_lab.sources.base import SourceLayoutError

# Member names as they appear in the DiFF test zips on the authors' Google Drive (October 2026).
MEMBERS = {
    "Midjourney/id_2160/000.png": ("diff_t2i_midjourney", "fully_synthetic"),
    "SDXL/id_0002/images/M1_015/sd2_gene_1.png": ("diff_t2i_sdxl", "fully_synthetic"),
    "FreeDoM_T/id_1893/clip_face_3/8939/FreeDom_Gene.png": ("diff_t2i_freedom_t", "fully_synthetic"),
    "HPS/id_1893/image_1/2.jpg/align_sd_gene_000_000.png": ("diff_t2i_hps", "fully_synthetic"),
    "SDXL_Refine/id_0002/images_refine/M1_015/sd2_gene_1_refine.png": (
        "diff_i2i_sdxl_refine",
        "fully_synthetic",
    ),
    "LoRA/id_0002/lora_0_1.png": ("diff_i2i_lora", "fully_synthetic"),
    "DreamBooth/id_2246/1024/img_db_000.png": ("diff_i2i_dreambooth", "fully_synthetic"),
    "FreeDoM_I/id_2160/arcface_face/1.jpg/FreeDom_Gene.png": ("diff_i2i_freedom_i", "fully_synthetic"),
    "DiffFace/id_0002/5.jpg/id_1892/img_id_001_002.png": ("diff_fs_diffface", "face_swap"),
    "DCFace/id_0626/025.jpg": ("diff_fs_dcface", "face_swap"),
    "Imagic/id_1893/2.jpg_0/img_000.png": ("diff_fe_imagic", "face_reenactment"),
    "CoDiff/id_2508/0.jpg/id_1320_1.png/0C_interText_optDM_4_alpha=1.0.png": (
        "diff_fe_codiff",
        "face_reenactment",
    ),
    "cycle_diff/id_1893/seed_1/2.jpg_0/gene.png": ("diff_fe_cycle_diff", "face_reenactment"),
}


@pytest.mark.parametrize("name", sorted(MEMBERS))
def test_every_test_method_maps_to_its_generator_and_attack_type(name):
    item = diff.parse_member(name)

    assert (item.generator, item.attack_type) == MEMBERS[name]


def test_generator_names_the_subset_of_the_paper():
    subsets = {diff.subset(generator) for generator, _ in MEMBERS.values()}

    assert subsets == {"t2i", "i2i", "fs", "fe"}


def test_subject_is_the_diff_identity_folder():
    item = diff.parse_member("SDXL/id_0002/images/M1_015/sd2_gene_1.png")

    assert item.subject_id == "diff_id_0002"


def test_every_image_is_its_own_video():
    first = diff.parse_member("Midjourney/id_2160/000.png")
    second = diff.parse_member("Midjourney/id_2160/001.png")

    assert first.video_id != second.video_id
    assert first.source_dataset == "diff"


@pytest.mark.parametrize(
    "name",
    [
        "LoRA/",
        "LoRA/id_0002/",
        "__MACOSX/LoRA/._id_0002",
        "__MACOSX/LoRA/id_0002/._lora_0_1.png",
        "LoRA/._id_0002",
    ],
)
def test_directories_and_macos_metadata_are_skipped(name):
    assert diff.parse_member(name) is None


def test_unknown_method_folder_is_rejected():
    with pytest.raises(SourceLayoutError, match="unexpected DiFF member 'StyleGAN/id_0002/0.png'"):
        diff.parse_member("StyleGAN/id_0002/0.png")


def test_member_outside_an_identity_folder_is_rejected():
    with pytest.raises(SourceLayoutError, match="expected <method>/id_<number>/"):
        diff.parse_member("Midjourney/000.png")


def test_non_image_member_is_rejected():
    with pytest.raises(SourceLayoutError, match="not an image"):
        diff.parse_member("Midjourney/id_2160/prompts.txt")
