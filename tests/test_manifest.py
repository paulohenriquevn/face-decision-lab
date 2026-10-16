import pytest

from face_decision_lab.errors import ManifestError
from face_decision_lab.manifest import load_manifest, write_manifest

HEADER = "image_path,attack_type,subject_id,video_id,source_dataset,generator\n"


def test_manifest_round_trips_samples(tmp_path, make_sample):
    # Arrange
    samples = [make_sample(), make_sample(attack_type="replay", subject="s2", video="v2", path="b.png")]
    path = tmp_path / "manifest.csv"

    # Act
    write_manifest(samples, path)

    # Assert
    assert load_manifest(path) == samples


def test_manifest_rejects_unknown_attack_type(tmp_path):
    path = tmp_path / "manifest.csv"
    path.write_text(HEADER + "a.png,deepfake,s1,v1,ds_a,camera\n")

    with pytest.raises(ManifestError, match="line 2: attack_type 'deepfake'"):
        load_manifest(path)


def test_manifest_rejects_empty_subject(tmp_path):
    path = tmp_path / "manifest.csv"
    path.write_text(HEADER + "a.png,print,,v1,ds_a,camera\n")

    with pytest.raises(ManifestError, match=r"line 2: empty or missing columns \['subject_id'\]"):
        load_manifest(path)


def test_manifest_rejects_header_without_required_columns(tmp_path):
    path = tmp_path / "manifest.csv"
    path.write_text("image_path,attack_type\na.png,print\n")

    with pytest.raises(ManifestError, match="header lacks columns"):
        load_manifest(path)


def test_manifest_rejects_file_without_samples(tmp_path):
    path = tmp_path / "manifest.csv"
    path.write_text(HEADER)

    with pytest.raises(ManifestError, match="no samples"):
        load_manifest(path)
