import pytest

from face_decision_lab.sources import celebdf
from face_decision_lab.sources.base import SourceLayoutError

# Member names and test-list lines as they appear in Celeb-DF-v2.zip (October 2026).
TEST_LIST = """1 YouTube-real/00170.mp4
1 Celeb-real/id0_0000.mp4
0 Celeb-synthesis/id0_id16_0000.mp4
"""


def test_celeb_real_video_is_bona_fide():
    item = celebdf.parse_member("Celeb-real/id0_0000.mp4")

    assert (item.attack_type, item.generator) == ("bona_fide", "celeb_real")


def test_youtube_real_video_is_bona_fide():
    item = celebdf.parse_member("YouTube-real/00170.mp4")

    assert (item.attack_type, item.generator) == ("bona_fide", "youtube_real")


def test_synthesis_video_is_a_face_swap():
    item = celebdf.parse_member("Celeb-synthesis/id0_id16_0000.mp4")

    assert (item.attack_type, item.generator) == ("face_swap", "celebdf_synthesis")


def test_synthesis_video_belongs_to_the_person_in_the_target_video():
    real = celebdf.parse_member("Celeb-real/id0_0000.mp4")
    fake = celebdf.parse_member("Celeb-synthesis/id0_id16_0000.mp4")

    assert real.subject_id == fake.subject_id == "celebdf_id0"


def test_youtube_videos_are_their_own_subjects():
    first = celebdf.parse_member("YouTube-real/00170.mp4")
    second = celebdf.parse_member("YouTube-real/00171.mp4")

    assert first.subject_id != second.subject_id


def test_every_video_has_its_own_video_id():
    paths = ["Celeb-real/id0_0000.mp4", "Celeb-synthesis/id0_id16_0000.mp4", "YouTube-real/00170.mp4"]

    assert len({celebdf.parse_member(path).video_id for path in paths}) == 3


def test_parse_member_skips_the_test_list_and_directories():
    assert celebdf.parse_member("List_of_testing_videos.txt") is None
    assert celebdf.parse_member("Celeb-real/") is None


def test_parse_member_rejects_an_unknown_folder():
    with pytest.raises(SourceLayoutError, match="unexpected Celeb-DF member 'Celeb-other/id0_0000.mp4'"):
        celebdf.parse_member("Celeb-other/id0_0000.mp4")


def test_parse_member_rejects_a_misnamed_synthesis_video():
    with pytest.raises(SourceLayoutError, match="unexpected Celeb-DF member"):
        celebdf.parse_member("Celeb-synthesis/id0_0000.mp4")


def test_official_test_videos_are_read_from_the_list():
    assert celebdf.official_test_videos(TEST_LIST) == {
        "YouTube-real/00170.mp4",
        "Celeb-real/id0_0000.mp4",
        "Celeb-synthesis/id0_id16_0000.mp4",
    }


def test_official_test_list_label_must_match_the_folder():
    with pytest.raises(SourceLayoutError, match="labels 'Celeb-synthesis/id0_id16_0000.mp4' as 1"):
        celebdf.official_test_videos("1 Celeb-synthesis/id0_id16_0000.mp4\n")


def test_official_test_list_rejects_a_malformed_line():
    with pytest.raises(SourceLayoutError, match="malformed Celeb-DF test list line 'Celeb-real/"):
        celebdf.official_test_videos("Celeb-real/id0_0000.mp4\n")
