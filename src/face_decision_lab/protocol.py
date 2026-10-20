"""The evaluation protocol every model in the lab is scored under, and its report.

The decision model and each baseline import these values instead of setting their own, so their
numbers cover the same subjects, the same test crops and the same operating point.
"""

from collections.abc import Callable

from .errors import MetricInputError
from .labels import BONA_FIDE
from .manifest import FaceSample
from .metrics import auc, bpcer_at_apcer, eer, operating_point, threshold_at_bpcer
from .pairs import pair_key
from .splits import capped, standard_split

SEED = 3407
TEST_FRACTION = 0.2
CALIBRATION_FRACTION = 0.1
MAX_TEST_SAMPLES = 4000
MAX_APCER = 0.05
# BPCER limits a bank's face check works at: at most 0.5%, usually 0.1% to 0.2%.
BPCER_LIMITS = (0.001, 0.002, 0.005)


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
    "cross_diffusionface": "diffusionface",
}

# FaceForensics++ protocols follow its official splits. "ffpp" is the in-domain test; the others
# test on Celeb-DF, the standard cross-dataset benchmark, after training on FaceForensics++ alone or
# together with the lab data. Only the four methods of the original paper enter, as in the
# literature these numbers are compared with; FaceShifter is left out.
FFPP_PROTOCOLS = ("ffpp", "ffpp_to_celebdf", "lab_ffpp_to_celebdf", "lab_ffpp_real", "ffpp_deepfakes_to_diff")
FFPP_GENERATORS = frozenset(
    {"ffpp_original", "ffpp_deepfakes", "ffpp_faceswap", "ffpp_face2face", "ffpp_neuraltextures"}
)

# "ffpp_deepfakes_to_diff" is the cross-dataset DiFF protocol of FSFM and FS-VFM: training on the
# FF++ originals and DeepFakes alone, testing on every DiFF crop.

# "lab_ffpp_real" adds only the FF++ originals to the lab's subject_holdout sides, as more bona fide
# video, and tests on the subject_holdout test side: a photo detector that must stop flagging real
# video as an attack without learning FF++'s manipulations.

# DF40 protocols. "df40_ff" trains on the 31 known methods' frames of FF++ training sequences
# (calibration: validation sequences) and tests on the 9 unknown methods, each with its own reals;
# e4e's inversions of FF++ test sequences are scored against DF40's real frames of those. "lab_df40"
# adds df40_ff's training and calibration sides to lab_ffpp_real's and keeps its test side.
DF40_PROTOCOLS = ("df40_ff", "lab_df40")

# Protocols that add super-resolved copies of their base protocol's training-side reals, labeled
# bona fide (sources/superresolved.py), and keep its calibration and test sides.
SUPER_RESOLVED_PROTOCOLS = {"lab_ffpp_real_sr": "lab_ffpp_real"}

# Protocols that add every photo of some real-photo datasets to their base protocol's training side,
# as bona fide, and keep its calibration and test sides. The detectors flag real photos from sources
# they never saw (DiffusionFace's CelebA-HQ, docs/results.md); these measure whether more kinds of
# real photo in training fix that, on real sources still left out.
EXTRA_REALS_PROTOCOLS = {"lab_ffpp_real_ffhq": ("lab_ffpp_real", ("ffhq",))}

# Datasets with their own test role; the lab's subject split never draws from them, since mixed in
# they would change the subject shuffle and the split would no longer be the one every other model
# was scored on.
EXTERNAL_DATASETS = frozenset(
    {"celebdf", "df40", "diff", "diffusionface", "faceforensics", "ffhq", "superresolved"}
)


def external_datasets(protocol: str) -> frozenset[str]:
    """The external datasets a protocol needs loaded next to the lab data."""
    if protocol in CROSS_DATASET_PROTOCOLS:
        return frozenset({CROSS_DATASET_PROTOCOLS[protocol]})
    if protocol in SUPER_RESOLVED_PROTOCOLS:
        return external_datasets(SUPER_RESOLVED_PROTOCOLS[protocol]) | {"superresolved"}
    if protocol in EXTRA_REALS_PROTOCOLS:
        base, datasets = EXTRA_REALS_PROTOCOLS[protocol]
        return external_datasets(base) | set(datasets)
    if protocol == "df40_ff":
        return frozenset({"df40"})
    if protocol == "lab_df40":
        return frozenset({"faceforensics", "df40"})
    if protocol in ("ffpp", "lab_ffpp_real"):
        return frozenset({"faceforensics"})
    if protocol == "ffpp_deepfakes_to_diff":
        return frozenset({"faceforensics", "diff"})
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
    if protocol in SUPER_RESOLVED_PROTOCOLS:
        return _super_resolved_split(samples, SUPER_RESOLVED_PROTOCOLS[protocol])
    if protocol in EXTRA_REALS_PROTOCOLS:
        return _extra_reals_split(samples, *EXTRA_REALS_PROTOCOLS[protocol])
    if protocol in DF40_PROTOCOLS:
        return _df40_split(samples, protocol)
    if protocol not in PROTOCOLS:
        known = sorted(PROTOCOLS) + sorted(CROSS_DATASET_PROTOCOLS) + sorted(FFPP_PROTOCOLS)
        known += sorted(SUPER_RESOLVED_PROTOCOLS) + sorted(EXTRA_REALS_PROTOCOLS) + list(DF40_PROTOCOLS)
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
    if protocol == "ffpp_deepfakes_to_diff":
        kept = {"ffpp_original", "ffpp_deepfakes"}
        train, calibration = ([s for s in sides[name] if s.generator in kept] for name in ("train", "val"))
        return train, calibration, _from(samples, "diff")
    if protocol == "lab_ffpp_real":
        lab_train, lab_calibration, lab_test = protocol_split(samples, "subject_holdout")
        originals = {name: [s for s in side if s.is_bona_fide] for name, side in sides.items()}
        return lab_train + originals["train"], lab_calibration + originals["val"], lab_test
    celebdf = _from(samples, "celebdf")
    if protocol == "ffpp_to_celebdf":
        return sides["train"], sides["val"], celebdf
    lab_train, lab_calibration, _ = protocol_split(samples, "subject_holdout")
    return lab_train + sides["train"], lab_calibration + sides["val"], celebdf


def _super_resolved_split(
    samples: list[FaceSample], base: str
) -> tuple[list[FaceSample], list[FaceSample], list[FaceSample]]:
    train, calibration, test = protocol_split(samples, base)
    # A copy of a calibration or test photo would leak that photo into training.
    trained = {pair_key(s) for s in train if s.is_bona_fide}
    copies = [s for s in _from(samples, "superresolved") if pair_key(s) in trained]
    return train + copies, calibration, test


def _extra_reals_split(
    samples: list[FaceSample], base: str, datasets: tuple[str, ...]
) -> tuple[list[FaceSample], list[FaceSample], list[FaceSample]]:
    train, calibration, test = protocol_split(samples, base)
    added = [s for dataset in datasets for s in _from(samples, dataset)]
    not_real = [s.image_path for s in added if not s.is_bona_fide]
    if not_real:
        raise ValueError(f"real-photo datasets {datasets} must hold bona fide crops only: {not_real[:3]}")
    return train + added, calibration, test


def _df40_split(
    samples: list[FaceSample], protocol: str
) -> tuple[list[FaceSample], list[FaceSample], list[FaceSample]]:
    from .sources import df40

    known = {f"df40_{m.lower()}" for m in df40.KNOWN_METHODS} | {"df40_ffpp_real"}
    sides: dict[str, list[FaceSample]] = {"train": [], "val": [], "test": []}
    for sample in _from(samples, "df40"):
        if sample.generator in known:
            sides[df40.official_split(sample)].append(sample)
        elif sample.generator == "df40_e4e":
            if df40.official_split(sample) == "test":
                sides["test"].append(sample)
        else:
            sides["test"].append(sample)
    # Known methods' frames of test sequences are not an unknown method; only the reals stay, for e4e.
    test = [s for s in sides["test"] if s.generator not in known or s.generator == "df40_ffpp_real"]
    if protocol == "df40_ff":
        return sides["train"], sides["val"], test
    lab_train, lab_calibration, lab_test = protocol_split(samples, "lab_ffpp_real")
    return lab_train + sides["train"], lab_calibration + sides["val"], lab_test


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


def auc_within_groups(
    samples: list[FaceSample], scores: list[float], group: Callable[[FaceSample], str]
) -> dict[str, float]:
    """AUC of each group's attacks against that group's own bona fide samples, where group maps a
    sample, real or fake, to its group. DF40's unknown methods each ship their own reals."""
    by_group: dict[str, list[tuple[FaceSample, float]]] = {}
    for sample, score in zip(samples, scores, strict=True):
        by_group.setdefault(group(sample), []).append((sample, score))
    result = {}
    for name, members in sorted(by_group.items()):
        kinds = {s.is_bona_fide for s, _ in members}
        if kinds != {True, False}:
            raise MetricInputError(f"group {name!r} needs bona fide and attack samples")
        result[name] = auc([score for _, score in members], [s.attack_type for s, _ in members])
    return result


def bootstrap_interval(
    samples: list[FaceSample],
    scores: list[float],
    statistic: Callable[[list[FaceSample], list[float]], float],
    rounds: int = 1000,
    level: float = 0.95,
    seed: int = SEED,
) -> tuple[float, float]:
    """Percentile interval of statistic(samples, scores) over bootstrap resamples of whole subjects.

    Crops of one subject (a person, or one image when a dataset names no identity) are not
    independent, so the resample draws subjects with replacement and keeps all their crops."""
    if not 0 < level < 1:
        raise ValueError(f"level must be between 0 and 1, got {level}")
    import numpy as np

    by_subject: dict[str, list[int]] = {}
    for index, sample in enumerate(samples):
        by_subject.setdefault(sample.subject_id, []).append(index)
    groups = list(by_subject.values())
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(rounds):
        chosen = [index for g in rng.integers(0, len(groups), len(groups)) for index in groups[g]]
        values.append(statistic([samples[i] for i in chosen], [scores[i] for i in chosen]))
    tail = (1 - level) / 2 * 100
    low, high = np.percentile(values, [tail, 100 - tail])
    return float(low), float(high)


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


def detection_at_bpcer(
    calibration: list[FaceSample],
    calibration_scores: list[float],
    test: list[FaceSample],
    test_scores: list[float],
    max_bpcers: tuple[float, ...] = BPCER_LIMITS,
) -> dict[str, dict]:
    """Detection (1 - APCER over all attacks) on the test side at thresholds fixed on the
    calibration side's bona fide crops, one entry per BPCER limit.

    A threshold set on the test's own reals would be optimistic, so the test reals only report the
    BPCER the fixed threshold actually gives there."""
    calibration_pairs = list(zip(calibration, calibration_scores, strict=True))
    test_pairs = list(zip(test, test_scores, strict=True))
    calibration_reals = [score for s, score in calibration_pairs if s.is_bona_fide]
    test_reals = [score for s, score in test_pairs if s.is_bona_fide]
    attacks = [(s.generator, score) for s, score in test_pairs if not s.is_bona_fide]
    if not calibration_reals:
        raise MetricInputError("the calibration side has no bona fide crop to set a threshold on")
    if not attacks:
        raise MetricInputError("the test side has no attack to detect")
    by_generator: dict[str, list[float]] = {}
    for generator, score in attacks:
        by_generator.setdefault(generator, []).append(score)

    def share_at_or_above(values, threshold):
        return sum(value >= threshold for value in values) / len(values)

    report = {}
    for max_bpcer in max_bpcers:
        threshold = threshold_at_bpcer(calibration_reals, max_bpcer)
        report[f"{max_bpcer:g}"] = {
            "threshold": threshold,
            "calibration_bpcer": share_at_or_above(calibration_reals, threshold),
            "test_bpcer": share_at_or_above(test_reals, threshold) if test_reals else None,
            "detection": share_at_or_above([score for _, score in attacks], threshold),
            "detection_by_generator": {
                generator: share_at_or_above(values, threshold)
                for generator, values in sorted(by_generator.items())
            },
        }
    return report
