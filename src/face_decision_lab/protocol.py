"""The evaluation protocol every model in the lab is scored under, and its report.

The decision model and each baseline import these values instead of setting their own, so their
numbers cover the same subjects, the same test crops and the same operating point.
"""

from .labels import BONA_FIDE
from .manifest import FaceSample
from .metrics import auc, bpcer_at_apcer, eer, operating_point
from .splits import capped, standard_split

SEED = 3407
TEST_FRACTION = 0.2
CALIBRATION_FRACTION = 0.1
MAX_TEST_SAMPLES = 4000
MAX_APCER = 0.05


def protocol_split(samples: list[FaceSample]) -> tuple[list[FaceSample], list[FaceSample], list[FaceSample]]:
    """Train, calibration and the capped test subset that every model is scored on."""
    train, calibration, test = standard_split(samples, TEST_FRACTION, CALIBRATION_FRACTION, SEED)
    return train, calibration, capped(test, MAX_TEST_SAMPLES, SEED)


def evaluation_report(samples: list[FaceSample], scores: list[float]) -> dict:
    """Metrics for attack scores (probability of attack) on the given samples."""
    types = [sample.attack_type for sample in samples]
    point = bpcer_at_apcer(scores, types, MAX_APCER)
    by_generator = [BONA_FIDE if sample.is_bona_fide else sample.generator for sample in samples]
    return {
        "samples": len(samples),
        "auc": auc(scores, types),
        "eer": eer(scores, types),
        "max_apcer": MAX_APCER,
        "bpcer_at_max_apcer": point.bpcer,
        "threshold": point.threshold,
        "apcer_by_attack_type": point.apcer_by_type,
        "apcer_by_generator": operating_point(scores, by_generator, point.threshold).apcer_by_type,
    }
