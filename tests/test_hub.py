import pytest

from face_decision_lab.hub import FIELDS, dataset_card, dataset_rows


def test_dataset_rows_put_the_image_path_under_the_build_root(make_sample):
    rows = dataset_rows([make_sample(path="images/a.jpg")], "/build")

    assert rows["image"] == ["/build/images/a.jpg"]


def test_dataset_rows_carry_every_manifest_field_except_the_path(make_sample):
    rows = dataset_rows([make_sample()], "/build")

    assert set(rows) == {"image", *FIELDS}
    assert "image_path" not in rows


def test_dataset_card_counts_rows_per_generator_and_label(make_sample):
    samples = [make_sample(source="deepfakeface", generator="insight", attack_type="face_swap")] * 2

    card = dataset_card(samples, {"insight": {"saved": 2}})

    assert "| deepfakeface | insight | face_swap | 2 |" in card


def test_dataset_card_marks_the_dataset_private(make_sample):
    assert "Private on purpose" in dataset_card([make_sample(source="axondata")], {})


def test_dataset_card_lists_only_the_sources_present(make_sample):
    card = dataset_card([make_sample(source="celebdf", generator="celeb_real")], {})

    assert "Celeb-DF" in card
    assert "OpenRL/DeepFakeFace" not in card


def test_dataset_card_rejects_a_source_without_a_license_row(make_sample):
    with pytest.raises(ValueError, match="no license row for source_dataset 'ds_unknown'"):
        dataset_card([make_sample(source="ds_unknown")], {})


def test_dataset_card_has_a_license_row_for_faceforensics(make_sample):
    assert "FaceForensics++" in dataset_card([make_sample(source="faceforensics")], {})


def test_dataset_card_has_a_license_row_for_diff(make_sample):
    card = dataset_card([make_sample(source="diff", generator="diff_t2i_sdxl")], {})

    assert "DiFF" in card


def test_dataset_card_has_a_license_row_for_diffusionface(make_sample):
    card = dataset_card([make_sample(source="diffusionface", generator="diffusionface_real")], {})

    assert "DiffusionFace" in card


def test_dataset_card_has_a_license_row_for_super_resolved_reals(make_sample):
    card = dataset_card([make_sample(source="superresolved", generator="sr_realesrgan_x4plus")], {})

    assert "Real-ESRGAN or Swin2SR" in card
