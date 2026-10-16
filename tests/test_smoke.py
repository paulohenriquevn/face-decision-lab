import pytest
from PIL import Image

from face_decision_lab.labels import ATTACK_TYPES, BONA_FIDE
from face_decision_lab.manifest import load_manifest
from face_decision_lab.smoke import SMOKE_TEST_SOURCE, SMOKE_TRAIN_SOURCE, write_smoke_dataset
from face_decision_lab.splits import cross_dataset_split, group_split


def test_smoke_manifest_loads_back_as_the_returned_samples(tmp_path):
    samples = write_smoke_dataset(tmp_path)

    assert load_manifest(tmp_path / "manifest.csv") == samples


def test_smoke_images_exist_at_the_requested_size(tmp_path):
    samples = write_smoke_dataset(tmp_path, image_size=64)

    with Image.open(tmp_path / samples[0].image_path) as image:
        assert image.size == (64, 64)


def test_smoke_dataset_supports_the_notebook_splits(tmp_path):
    samples = write_smoke_dataset(tmp_path)

    train, test = cross_dataset_split(samples, {SMOKE_TRAIN_SOURCE}, {SMOKE_TEST_SOURCE})
    train, calibration = group_split(train, test_fraction=0.25, seed=0)

    assert train and calibration and test


def test_smoke_test_side_holds_bona_fide_and_attacks(tmp_path):
    samples = write_smoke_dataset(tmp_path)

    test_types = {s.attack_type for s in samples if s.source_dataset == SMOKE_TEST_SOURCE}

    assert BONA_FIDE in test_types and len(test_types) > 1


def test_smoke_dataset_rejects_a_single_subject(tmp_path):
    with pytest.raises(ValueError, match="at least 2 subjects"):
        write_smoke_dataset(tmp_path, subjects_per_source=1)


def test_smoke_train_side_holds_every_label(tmp_path):
    samples = write_smoke_dataset(tmp_path)

    train_types = {s.attack_type for s in samples if s.source_dataset == SMOKE_TRAIN_SOURCE}

    assert train_types == set(ATTACK_TYPES)
