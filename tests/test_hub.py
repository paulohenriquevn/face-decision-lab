from face_decision_lab.hub import FIELDS, dataset_card, dataset_rows


def test_dataset_rows_put_the_image_path_under_the_build_root(make_sample):
    rows = dataset_rows([make_sample(path="images/a.jpg")], "/build")

    assert rows["image"] == ["/build/images/a.jpg"]


def test_dataset_rows_carry_every_manifest_field_except_the_path(make_sample):
    rows = dataset_rows([make_sample()], "/build")

    assert set(rows) == {"image", *FIELDS}
    assert "image_path" not in rows


def test_dataset_card_counts_rows_per_generator_and_label(make_sample):
    samples = [make_sample(generator="insight", attack_type="face_swap")] * 2

    card = dataset_card(samples, {"insight": {"saved": 2}})

    assert "| ds_a | insight | face_swap | 2 |" in card


def test_dataset_card_marks_the_dataset_private(make_sample):
    assert "Private on purpose" in dataset_card([make_sample()], {})
