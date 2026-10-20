import pytest

from face_decision_lab.sources import df40
from face_decision_lab.sources.base import SourceLayoutError

# Member names as they appear in the DF40 zips on the authors' Google Drive (October 2026). The
# folder the authors' email calls "testing data" holds the FF++-domain training frames; the one it
# calls "training data" holds the test frames, including the unknown methods.
TRAIN_MEMBERS = {
    ("simswap", "simswap/frames/402_453/326.png"): ("df40_simswap", "face_swap", "402"),
    ("uniface", "frames/350_349/182.png"): ("df40_uniface", "face_swap", "350"),
    ("fomm", "fomm/frames/137_165/012.png"): ("df40_fomm", "face_reenactment", "137"),
    ("sadtalker", "sadtalker/frames/645_5J6jAC6XxAI_50008/003.png"): (
        "df40_sadtalker",
        "face_reenactment",
        "645",
    ),
    ("sadtalker", "sadtalker/frames/000##CfmNXPMjChs_50010/034.png"): (
        "df40_sadtalker",
        "face_reenactment",
        "000",
    ),
    ("sd2.1", "sd2.1/001/001_325.png"): ("df40_sd2.1", "fully_synthetic", "001"),
    ("RDDM", "RDDM/001/sample-10885.png"): ("df40_rddm", "fully_synthetic", "001"),
    ("StyleGAN2", "StyleGAN2/001/seed0188.png"): ("df40_stylegan2", "fully_synthetic", "001"),
}


@pytest.mark.parametrize(("archive", "name"), sorted(TRAIN_MEMBERS))
def test_training_frames_map_to_their_method_and_ffpp_target(archive, name):
    item = df40.parse_train_member(archive, name)

    generator, attack_type, target = TRAIN_MEMBERS[(archive, name)]
    assert (item.generator, item.attack_type, item.subject_id) == (generator, attack_type, f"ffpp_{target}")
    assert item.source_dataset == "df40"


def test_frames_of_one_fake_video_share_a_video_id():
    first = df40.parse_train_member("simswap", "simswap/frames/402_453/326.png")
    second = df40.parse_train_member("simswap", "simswap/frames/402_453/327.png")

    assert first.video_id == second.video_id == "df40/simswap/402_453"


@pytest.mark.parametrize(
    "name", ["simswap/landmarks/402_453/326.npy", "simswap/frames/402_453/", "simswap/.DS_Store"]
)
def test_landmarks_directories_and_metadata_are_skipped(name):
    assert df40.parse_train_member("simswap", name) is None


def test_an_unknown_training_archive_is_rejected():
    with pytest.raises(SourceLayoutError, match="unexpected DF40 training archive 'MidJourney'"):
        df40.parse_train_member("MidJourney", "MidJourney/fake/1.png")


def test_an_unexpected_training_member_is_rejected():
    with pytest.raises(SourceLayoutError, match="unexpected DF40 member 'simswap/clips/402_453/1.png'"):
        df40.parse_train_member("simswap", "simswap/clips/402_453/1.png")


def test_real_ffpp_frames_are_bona_fide_of_their_sequence():
    item = df40.parse_real_member("FaceForensics++/original_sequences/youtube/c23/frames/071/277.png")

    assert (item.attack_type, item.generator, item.subject_id) == ("bona_fide", "df40_ffpp_real", "ffpp_071")
    assert item.video_id == "df40/ffpp_real/071"


TEST_MEMBERS = {
    ("CollabDiff", "CollabDiff/fake/0/0_x_1.png"): ("df40_collabdiff", "face_reenactment"),
    ("CollabDiff", "CollabDiff/real/0.jpg"): ("df40_collabdiff_real", "bona_fide"),
    ("deepfacelab", "deepfacelab/fake/frames/000_003/012.png"): ("df40_deepfacelab", "face_swap"),
    ("deepfacelab", "deepfacelab/real/frames/000/012.png"): ("df40_deepfacelab_real", "bona_fide"),
    ("heygen", "heygen_new/fake/frames/3/001.png"): ("df40_heygen", "face_reenactment"),
    ("heygen", "heygen_new/real/Clip+RfHY644z7aI+P0+C1+F2537-2828/001.png"): (
        "df40_heygen_real",
        "bona_fide",
    ),
    ("MidJourney", "MidJourney/fake/13592447708_Single_person_9f.png"): (
        "df40_midjourney",
        "fully_synthetic",
    ),
    ("MidJourney", "MidJourney/real/11180.jpg"): ("df40_midjourney_real", "bona_fide"),
    ("stargan", "stargan/fake/fake/1.jpg"): ("df40_stargan", "face_reenactment"),
    ("stargan", "stargan/real/real/1.jpg"): ("df40_stargan_real", "bona_fide"),
    ("starganv2", "starganv2/fake/1.jpg"): ("df40_starganv2", "face_reenactment"),
    ("styleclip", "styleclip/real/1.jpg"): ("df40_styleclip_real", "bona_fide"),
    ("whichfaceisreal", "whichfaceisreal/fake/1.png"): ("df40_whichfaceisreal", "fully_synthetic"),
    ("whichfaceisreal", "whichfaceisreal/real/1.jpeg"): ("df40_whichfaceisreal_real", "bona_fide"),
    ("e4e", "e4e/e4e/ff/953/350.jpg"): ("df40_e4e", "face_reenactment"),
}


@pytest.mark.parametrize(("archive", "name"), sorted(TEST_MEMBERS))
def test_unknown_methods_map_their_fakes_and_their_own_reals(archive, name):
    item = df40.parse_unknown_member(archive, name)

    assert (item.generator, item.attack_type) == TEST_MEMBERS[(archive, name)]


@pytest.mark.parametrize(
    ("archive", "name"),
    [
        ("styleclip", "__MACOSX/styleclip/fake/._1.jpg"),
        ("styleclip", "styleclip/.DS_Store"),
        ("heygen", "heygen_new/fake/landmarks/3/001.npy"),
        ("e4e", "e4e/e4e/cdf/Celeb-real/inversions/1.jpg"),
        ("e4e", "e4e/e4e/ff/latents.pt"),
        ("MidJourney", "MidJourney/fake/13592447708_Single_person_2703e6e5_01.gif"),
    ],
)
def test_unknown_archives_skip_metadata_landmarks_and_the_celebdf_domain(archive, name):
    assert df40.parse_unknown_member(archive, name) is None


def test_e4e_inversions_belong_to_their_ffpp_sequence():
    item = df40.parse_unknown_member("e4e", "e4e/e4e/ff/953/350.jpg")

    assert item.subject_id == "ffpp_953"


def test_method_groups_each_unknown_fake_with_its_own_reals(make_sample):
    fake = make_sample(attack_type="fully_synthetic", source="df40", generator="df40_midjourney")
    real = make_sample(source="df40", generator="df40_midjourney_real")
    ffpp_real = make_sample(source="df40", generator="df40_ffpp_real")

    assert df40.method(fake) == df40.method(real) == "midjourney"
    assert df40.method(ffpp_real) == "ffpp_real"


@pytest.mark.parametrize(("target", "split"), [("071", "train"), ("953", "test")])
def test_official_split_follows_the_ffpp_target(target, split):
    item = df40.parse_train_member("simswap", f"simswap/frames/{target}_000/001.png")

    assert df40.official_split(item) == split


def test_official_split_needs_an_ffpp_sequence(make_sample):
    with pytest.raises(ValueError, match="no FaceForensics\\+\\+ sequence"):
        df40.official_split(make_sample(subject="df40_midjourney_11180", source="df40"))


def test_sampled_keeps_about_one_in_every():
    kept = sum(df40.sampled(f"simswap/frames/402_453/{n:03d}.png", every=8) for n in range(4000))

    assert 400 < kept < 600


def test_unknown_group_scores_e4e_against_the_ffpp_real_frames(make_sample):
    real = make_sample(source="df40", generator="df40_ffpp_real")
    e4e = make_sample(attack_type="face_reenactment", source="df40", generator="df40_e4e")
    midjourney_real = make_sample(source="df40", generator="df40_midjourney_real")

    assert df40.unknown_group(real) == df40.unknown_group(e4e) == "e4e"
    assert df40.unknown_group(midjourney_real) == "midjourney"
