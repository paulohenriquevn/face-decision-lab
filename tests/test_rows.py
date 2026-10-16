import pytest

from face_decision_lab.errors import PredictionError
from face_decision_lab.labels import ATTACK_TYPES, STATE
from face_decision_lab.rows import attack_score, decision_row


def test_bona_fide_sample_answers_true_to_is_bona_fide(make_sample):
    row = decision_row(make_sample(attack_type="bona_fide"), "/data")

    assert row["gold"] == {"is_bona_fide": "true", "attack_type": "bona_fide"}


def test_attack_sample_answers_false_and_names_its_type(make_sample):
    row = decision_row(make_sample(attack_type="face_swap"), "/data")

    assert row["gold"] == {"is_bona_fide": "false", "attack_type": "face_swap"}


def test_row_points_to_the_image_under_the_root(make_sample):
    row = decision_row(make_sample(path="ffpp/x/0001.png"), "/data")

    assert row["images"] == [{"path": "/data/ffpp/x/0001.png"}]


def test_row_state_does_not_reveal_the_source_dataset_or_label(make_sample):
    row = decision_row(make_sample(attack_type="replay", source="oulu_npu"), "/data")

    assert row["state"] == STATE
    assert "oulu" not in str(row["state"]).lower()
    assert "replay" not in str(row["state"]).lower()


def test_every_attack_type_is_an_option_of_the_choice_question(make_sample):
    row = decision_row(make_sample(), "/data")

    assert set(row["questions"]["attack_type"]["criteria"]) == set(ATTACK_TYPES)


def test_attack_score_is_the_complement_of_bona_fide_probability():
    answers = {"is_bona_fide": {"probabilities": {"true": 0.2, "false": 0.8}}}

    assert attack_score(answers) == pytest.approx(0.8)


def test_attack_score_rejects_an_answer_without_true_probability():
    with pytest.raises(PredictionError, match="no is_bona_fide probability"):
        attack_score({"is_bona_fide": {"probabilities": {"yes": 0.2}}})


def test_attack_score_rejects_a_probability_outside_unit_interval():
    with pytest.raises(PredictionError, match="outside"):
        attack_score({"is_bona_fide": {"probabilities": {"true": 1.5}}})
