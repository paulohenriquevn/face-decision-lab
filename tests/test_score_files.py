import numpy as np
import pytest

from face_decision_lab.score_files import load_scores, save_scores


def test_scores_round_trip_with_their_samples(tmp_path, make_sample):
    samples = [
        make_sample(
            attack_type="bona_fide",
            subject="s1",
            video="v1",
            source="diff",
            path="a.jpg",
            generator="diff_real",
        ),
        make_sample(
            attack_type="face_swap",
            subject="s2",
            video="v2",
            source="diff",
            path="b.jpg",
            generator="diff_fs",
        ),
    ]
    scores = {"v2": np.array([0.1, 0.9]), "gend_pe": np.array([0.2, 0.7])}

    save_scores(tmp_path / "scores.npz", samples, scores)
    loaded_samples, loaded_scores = load_scores(tmp_path / "scores.npz")

    assert loaded_samples == samples
    assert loaded_scores.keys() == scores.keys()
    assert all(np.array_equal(loaded_scores[k], scores[k]) for k in scores)


def test_scores_must_cover_every_sample(tmp_path, make_sample):
    with pytest.raises(ValueError, match="detector 'v2' has 1 scores for 2 samples"):
        save_scores(
            tmp_path / "scores.npz", [make_sample(), make_sample(path="b.jpg")], {"v2": np.array([0.5])}
        )
