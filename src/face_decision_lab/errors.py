"""Typed errors raised at the lab's boundaries."""


class ManifestError(ValueError):
    """A manifest file or one of its rows is invalid."""


class SplitError(ValueError):
    """A train/test split cannot be built without leaking or emptying a side."""


class MetricInputError(ValueError):
    """Scores or labels given to a metric are inconsistent."""


class PredictionError(RuntimeError):
    """The decision model returned an answer the lab cannot score."""
