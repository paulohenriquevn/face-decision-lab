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

# Cross-dataset protocols by name, each with the dataset it holds out. Training and calibration are
# the subject_holdout ones, built from every other dataset; the test side is every crop of the held
# out dataset, uncapped, so all frames of each video are there for the video-level score.
CROSS_DATASET_PROTOCOLS = {
    "cross_celebdf": "celebdf",
}

# FaceForensics++ protocols follow its official splits. "ffpp" is the in-domain test; the others
# test on Celeb-DF, the standard cross-dataset benchmark, after training on FaceForensics++ alone or
# together with the lab data. Only the four methods of the original paper enter, as in the
# literature these numbers are compared with; FaceShifter is left out.
FFPP_PROTOCOLS = ("ffpp", "ffpp_to_celebdf", "lab_ffpp_to_celebdf")
FFPP_GENERATORS = frozenset(
    {"ffpp_original", "ffpp_deepfakes", "ffpp_faceswap", "ffpp_face2face", "ffpp_neuraltextures"}
)

# Datasets with their own test role; the lab's subject split never draws from them, since mixed in
# they would change the subject shuffle and the split would no longer be the one every other model
# was scored on.
EXTERNAL_DATASETS = frozenset({"celebdf", "faceforensics"})


def external_datasets(protocol: str) -> frozenset[str]:
    """The external datasets a protocol needs loaded next to the lab data."""
    if protocol in CROSS_DATASET_PROTOCOLS:
        return frozenset({CROSS_DATASET_PROTOCOLS[protocol]})
    if protocol == "ffpp":
        return frozenset({"faceforensics"})
    if protocol in FFPP_PROTOCOLS:
        return frozenset({"faceforensics", "celebdf"})
    return frozenset()


def protocol_split(
    samples: list[FaceSample], protocol: str = "subject_holdout"
) -> tuple[list[FaceSample], list[FaceSample], list[FaceSample]]:
    """Train, calibration and the test side that every model is scored on."""
    if protocol in CROSS_DATASET_PROTOCOLS:
        return _cross_dataset_split(samples, CROSS_DATASET_PROTOCOLS[protocol])
    if protocol in FFPP_PROTOCOLS:
        return _ffpp_split(samples, protocol)
    if protocol not in PROTOCOLS:
        known = sorted(PROTOCOLS) + sorted(CROSS_DATASET_PROTOCOLS) + sorted(FFPP_PROTOCOLS)
        raise ValueError(f"unknown protocol {protocol!r}; choose one of {known}")
    samples = [s for s in samples if s.source_dataset not in EXTERNAL_DATASETS]
    held_out = PROTOCOLS[protocol]
    train, calibration, test = standard_split(samples, TEST_FRACTION, CALIBRATION_FRACTION, SEED)
    if held_out:
        train = [s for s in train if s.generator not in held_out]
        calibration = [s for s in calibration if s.generator not in held_out]
        test = [s for s in test if s.is_bona_fide or s.generator in held_out]
    return train, calibration, capped(test, MAX_TEST_SAMPLES, SEED)


def _from(samples: list[FaceSample], dataset: str) -> list[FaceSample]:
    chosen = [s for s in samples if s.source_dataset == dataset]
    if not chosen:
        raise ValueError(f"no {dataset!r} samples; load that dataset alongside the lab one")
    return chosen


def _cross_dataset_split(
    samples: list[FaceSample], held_out: str
) -> tuple[list[FaceSample], list[FaceSample], list[FaceSample]]:
    test = _from(samples, held_out)
    train, calibration, _ = protocol_split(samples, "subject_holdout")
    return train, calibration, test


def _ffpp_split(
    samples: list[FaceSample], protocol: str
) -> tuple[list[FaceSample], list[FaceSample], list[FaceSample]]:
    from .sources.faceforensics import official_split

    ffpp = [s for s in _from(samples, "faceforensics") if s.generator in FFPP_GENERATORS]
    sides = {name: [s for s in ffpp if official_split(s) == name] for name in ("train", "val", "test")}
    if protocol == "ffpp":
        return sides["train"], sides["val"], sides["test"]
    celebdf = _from(samples, "celebdf")
    if protocol == "ffpp_to_celebdf":
        return sides["train"], sides["val"], celebdf
    lab_train, lab_calibration, _ = protocol_split(samples, "subject_holdout")
    return lab_train + sides["train"], lab_calibration + sides["val"], celebdf


def video_scores(samples: list[FaceSample], scores: list[float]) -> tuple[list[str], list[float]]:
    """One attack type and one score per video: the mean of its frame scores."""
    by_video: dict[str, list[float]] = {}
    types: dict[str, str] = {}
    for sample, score in zip(samples, scores, strict=True):
        by_video.setdefault(sample.video_id, []).append(score)
        types[sample.video_id] = sample.attack_type
    videos = sorted(by_video)
    return [types[v] for v in videos], [sum(by_video[v]) / len(by_video[v]) for v in videos]


def evaluation_report(samples: list[FaceSample], scores: list[float]) -> dict:
    """Metrics for attack scores (probability of attack) on the given samples."""
    types = [sample.attack_type for sample in samples]
    point = bpcer_at_apcer(scores, types, MAX_APCER)
    by_generator = [BONA_FIDE if sample.is_bona_fide else sample.generator for sample in samples]
    video_types, per_video = video_scores(samples, scores)
    return {
        "samples": len(samples),
        "auc": auc(scores, types),
        "eer": eer(scores, types),
        "max_apcer": MAX_APCER,
        "bpcer_at_max_apcer": point.bpcer,
        "threshold": point.threshold,
        "apcer_by_attack_type": point.apcer_by_type,
        "apcer_by_generator": operating_point(scores, by_generator, point.threshold).apcer_by_type,
        # Frame scores averaged per video, the usual video-level number in deepfake papers.
        "videos": len(per_video),
        "video_auc": auc(per_video, video_types),
    }
