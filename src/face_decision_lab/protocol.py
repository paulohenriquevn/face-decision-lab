"""The evaluation protocol every model in the lab is scored under, and its report.

The decision model and each baseline import these values instead of setting their own, so their
numbers cover the same subjects, the same test crops and the same operating point.
"""

from collections.abc import Callable

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
    "cross_diff": "diff",
}

# FaceForensics++ protocols follow its official splits. "ffpp" is the in-domain test; the others
# test on Celeb-DF, the standard cross-dataset benchmark, after training on FaceForensics++ alone or
# together with the lab data. Only the four methods of the original paper enter, as in the
# literature these numbers are compared with; FaceShifter is left out.
FFPP_PROTOCOLS = ("ffpp", "ffpp_to_celebdf", "lab_ffpp_to_celebdf", "lab_ffpp_real")
FFPP_GENERATORS = frozenset(
    {"ffpp_original", "ffpp_deepfakes", "ffpp_faceswap", "ffpp_face2face", "ffpp_neuraltextures"}
)

# "lab_ffpp_real" adds only the FF++ originals to the lab's subject_holdout sides, as more bona fide
# video, and tests on the subject_holdout test side: a photo detector that must stop flagging real
# video as an attack without learning FF++'s manipulations.

# Datasets with their own test role; the lab's subject split never draws from them, since mixed in
# they would change the subject shuffle and the split would no longer be the one every other model
# was scored on.
EXTERNAL_DATASETS = frozenset({"celebdf", "diff", "faceforensics"})


def external_datasets(protocol: str) -> frozenset[str]:
    """The external datasets a protocol needs loaded next to the lab data."""
    if protocol in CROSS_DATASET_PROTOCOLS:
        return frozenset({CROSS_DATASET_PROTOCOLS[protocol]})
    if protocol in ("ffpp", "lab_ffpp_real"):
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
    if protocol == "lab_ffpp_real":
        lab_train, lab_calibration, lab_test = protocol_split(samples, "subject_holdout")
        originals = {name: [s for s in side if s.is_bona_fide] for name, side in sides.items()}
        return lab_train + originals["train"], lab_calibration + originals["val"], lab_test
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


def _mean_by(samples: list[FaceSample], scores: list[float]) -> dict[str, float]:
    totals: dict[str, list[float]] = {}
    for sample, score in zip(samples, scores, strict=True):
        totals.setdefault(sample.generator, []).append(score)
    return {generator: sum(values) / len(values) for generator, values in sorted(totals.items())}


def auc_by_group(
    samples: list[FaceSample], scores: list[float], group: Callable[[str], str] = lambda generator: generator
) -> dict[str, float]:
    """AUC of every bona fide sample against the attacks of each group, where group maps a generator
    to its group name. Papers on multi-method benchmarks (DiFF's T2I, I2I, FS and FE) report this."""
    bona_fide = [(s, score) for s, score in zip(samples, scores, strict=True) if s.is_bona_fide]
    by_group: dict[str, list[tuple[FaceSample, float]]] = {}
    for sample, score in zip(samples, scores, strict=True):
        if not sample.is_bona_fide:
            by_group.setdefault(group(sample.generator), []).append((sample, score))
    result = {}
    for name, attacks in sorted(by_group.items()):
        pairs = bona_fide + attacks
        result[name] = auc([score for _, score in pairs], [s.attack_type for s, _ in pairs])
    return result


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
        # Where a detector's scores sit for each source, real ones included: a detector can rank well
        # and still score a whole real source as suspicious.
        "mean_score_by_generator": _mean_by(samples, scores),
        "auc_by_generator": auc_by_group(samples, scores),
        "videos": len(per_video),
        "video_auc": auc(per_video, video_types),
    }
