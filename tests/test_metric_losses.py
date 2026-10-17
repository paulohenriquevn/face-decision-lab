import math

import pytest
import torch
from torch.nn import functional

from face_decision_lab.metric_losses import alignment_loss, gend_loss, uniformity_loss


def _unit(rows):
    return functional.normalize(torch.tensor(rows, dtype=torch.float32), dim=-1)


def test_alignment_is_zero_when_each_class_shares_one_point():
    features = _unit([[1, 0], [1, 0], [0, 1], [0, 1]])
    labels = torch.tensor([0, 0, 1, 1])

    assert alignment_loss(features, labels).item() == pytest.approx(0.0)


def test_alignment_measures_squared_distance_within_a_class():
    # Two orthogonal unit vectors are sqrt(2) apart, so their squared distance is 2.
    features = _unit([[1, 0], [0, 1]])
    labels = torch.tensor([0, 0])

    assert alignment_loss(features, labels).item() == pytest.approx(2.0)


def test_alignment_ignores_pairs_across_classes():
    features = _unit([[1, 0], [0, 1]])
    labels = torch.tensor([0, 1])

    assert alignment_loss(features, labels).item() == pytest.approx(0.0)


def test_uniformity_is_lower_for_spread_features_than_for_a_cluster():
    spread = _unit([[1, 0], [-1, 0], [0, 1], [0, -1]])
    cluster = _unit([[1, 0], [1, 0.01], [1, -0.01], [1, 0.02]])

    assert uniformity_loss(spread).item() < uniformity_loss(cluster).item()


def test_uniformity_of_identical_points_is_zero():
    # exp(0) = 1 for every pair, and log(1) = 0.
    features = _unit([[1, 0], [1, 0], [1, 0]])

    assert uniformity_loss(features).item() == pytest.approx(0.0)


def test_uniformity_of_two_antipodal_points():
    # Squared distance 4, temperature 2: log(exp(-8)) = -8.
    features = _unit([[1, 0], [-1, 0]])

    assert uniformity_loss(features).item() == pytest.approx(-8.0)


def test_gend_loss_combines_the_three_terms_with_the_paper_weights():
    features = _unit([[1, 0], [0, 1], [1, 1]])
    labels = torch.tensor([0, 0, 1])
    logits = torch.tensor([[2.0, 0.0], [0.5, 0.5], [0.0, 3.0]])

    expected = (
        functional.cross_entropy(logits, labels)
        + 0.1 * alignment_loss(features, labels)
        + 0.5 * uniformity_loss(features)
    )

    assert gend_loss(logits, features, labels).item() == pytest.approx(expected.item())


def test_gend_loss_sends_gradient_to_the_features():
    features = torch.randn(6, 4, requires_grad=True)
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    logits = torch.randn(6, 2)

    gend_loss(logits, functional.normalize(features, dim=-1), labels).backward()

    assert features.grad is not None and torch.isfinite(features.grad).all()
    assert not math.isclose(features.grad.abs().sum().item(), 0.0)
