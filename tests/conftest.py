import pytest

from face_decision_lab.manifest import FaceSample


@pytest.fixture
def make_sample():
    def make(
        attack_type="bona_fide", subject="s1", video="v1", source="ds_a", path="a.png", generator="camera"
    ):
        return FaceSample(
            image_path=path,
            attack_type=attack_type,
            subject_id=subject,
            video_id=video,
            source_dataset=source,
            generator=generator,
        )

    return make
