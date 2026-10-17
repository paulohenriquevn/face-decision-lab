"""The GenD training objective (arXiv 2508.06248): cross entropy plus alignment and uniformity
(Wang and Isola, ICML 2020) on L2-normalized features.

loss = cross_entropy + alpha * alignment + beta * uniformity, with alpha = 0.1 and beta = 0.5 for
CLIP ViT-L/14 in the paper. Alignment pulls features of the same class together; uniformity spreads
all features over the hypersphere so the classes do not collapse onto a few directions.
"""

import torch
from torch.nn import functional

GEND_ALPHA = 0.1
GEND_BETA = 0.5


def alignment_loss(features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Mean squared distance between every pair of same-class features in the batch.

    Zero when no class has two samples in the batch. Features must already be L2-normalized.
    """
    same_class = labels.unsqueeze(0) == labels.unsqueeze(1)
    pairs = torch.triu(same_class, diagonal=1)
    if not pairs.any():
        return features.sum() * 0.0
    distances = torch.cdist(features, features).pow(2)
    return distances[pairs].mean()


def uniformity_loss(features: torch.Tensor, temperature: float = 2.0) -> torch.Tensor:
    """log E[exp(-t * ||x - y||^2)] over every distinct pair in the batch. Lower is more uniform."""
    if features.shape[0] < 2:
        return features.sum() * 0.0
    return torch.pdist(features).pow(2).mul(-temperature).exp().mean().log()


def gend_loss(
    logits: torch.Tensor,
    features: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = GEND_ALPHA,
    beta: float = GEND_BETA,
) -> torch.Tensor:
    """Cross entropy on two-class logits plus the weighted alignment and uniformity terms."""
    return (
        functional.cross_entropy(logits, labels)
        + alpha * alignment_loss(features, labels)
        + beta * uniformity_loss(features)
    )
