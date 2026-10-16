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


# Protocols by name. "subject_holdout" tests new people on known generators. The "unseen_*" ones
# also remove one generator from training and test only on it, which measures generalization to a
# manipulation method the model never saw: the question that separates detectors in the literature.
PROTOCOLS = {
    "subject_holdout": frozenset(),
    "unseen_text2img": frozenset({"text2img"}),
    "unseen_insight": frozenset({"insight"}),
    "unseen_inpainting": frozenset({"inpainting"}),
}


def protocol_split(
    samples: list[FaceSample], protocol: str = "subject_holdout"
) -> tuple[list[FaceSample], list[FaceSample], list[FaceSample]]:
    """Train, calibration and the capped test subset that every model is scored on."""
    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown protocol {protocol!r}; choose one of {sorted(PROTOCOLS)}")
    held_out = PROTOCOLS[protocol]
    train, calibration, test = standard_split(samples, TEST_FRACTION, CALIBRATION_FRACTION, SEED)
    if held_out:
        train = [s for s in train if s.generator not in held_out]
        calibration = [s for s in calibration if s.generator not in held_out]
        test = [s for s in test if s.is_bona_fide or s.generator in held_out]
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
